#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>


struct SegmentOutput {
    int _id; // 结果类别id
    float _confidence; // 结果置信度
    cv::Rect2f _box; // 矩形框
    cv::Mat _boxMask; // 矩形框内mask, 节省内存空间和加快速度
};


std::vector<float> Letterbox(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size) {
    auto in_h = static_cast<float>(src.rows);
    auto in_w = static_cast<float>(src.cols);
    float out_h = out_size.height;
    float out_w = out_size.width;

    float scale = (std::min)(out_w / in_w, out_h / in_h);

    int mid_h = static_cast<int>(in_h * scale);
    int mid_w = static_cast<int>(in_w * scale);

    cv::resize(src, dst, cv::Size(mid_w, mid_h));

    int top = (static_cast<int>(out_h) - mid_h) / 2;
    int down = (static_cast<int>(out_h) - mid_h + 1) / 2;
    int left = (static_cast<int>(out_w) - mid_w) / 2;
    int right = (static_cast<int>(out_w) - mid_w + 1) / 2;

    cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    std::vector<float> pad_info{ static_cast<float>(left), static_cast<float>(top), scale };
    return pad_info;
}

cv::Rect toBox(const cv::Mat& input, const cv::Rect& range)
{
    const float cx = input.at<float>(0);
    const float cy = input.at<float>(1);
    const float ow = input.at<float>(2);
    const float oh = input.at<float>(3);
    cv::Rect box;
    box.x = cvRound(cx - 0.5f * ow);
    box.y = cvRound(cy - 0.5f * oh);
    box.width = cvRound(ow);
    box.height = cvRound(oh);
    return box & range;
}

void draw_result(cv::Mat& image, std::vector<SegmentOutput>& results)
{
    cv::Mat mask = image.clone();
    for (const SegmentOutput& result : results) {
        cv::rectangle(image, result._box, cv::Scalar(0, 255, 0), 2, 8);
        mask(result._box).setTo(cv::Scalar(0, 0, 255), result._boxMask);
    }
    cv::addWeighted(image, 0.5, mask, 0.8, 1, image); // 把mask加在原图上面
}

int main() {

    const float conf_threshold = 0.5;
    const float nms_threshold = 0.5;

    torch::jit::script::Module model;
    model = torch::jit::load("yolov8/yolov8s-seg.torchscript");
    model.to(torch::kCUDA);
    model.eval();

    cv::Mat image = cv::imread("yolov8/zidane.jpg");
    cv::Mat resize_image;
    std::vector<float> pad_info = Letterbox(image, resize_image, cv::Size(640, 640));

    torch::Tensor image_tensor = torch::from_blob(resize_image.data, { resize_image.rows, resize_image.cols, 3 }, torch::kByte).to(torch::kCUDA);
    image_tensor = image_tensor.toType(torch::kFloat32).div(255);
    image_tensor = image_tensor.permute({ 2, 0, 1 });
    image_tensor = image_tensor.unsqueeze(0);

    std::vector<torch::jit::IValue> inputs{ image_tensor };
    auto net_outputs = model.forward(inputs).toTuple();


    at::Tensor main_output = net_outputs->elements()[0].toTensor().to(torch::kCPU);
    at::Tensor mask_output = net_outputs->elements()[1].toTensor().to(torch::kCPU);

    cv::Mat detect_buffer = cv::Mat(main_output.sizes()[1], main_output.sizes()[2], CV_32F, (float*)main_output.data_ptr()).t();

    cv::Mat segment_buffer(32, 25600, CV_32F);
    std::memcpy((void*)segment_buffer.data, mask_output.data_ptr(), sizeof(float) * 32 * 160 * 160);


    std::vector<cv::Rect> mask_boxes;
    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Mat> masks;
    for (int i = 0; i < detect_buffer.rows; ++i) {
        const cv::Mat result = detect_buffer.row(i);
        /// 处理检测部分的结果
        // 取置信度最大的那个标签
        const cv::Mat classes_scores = result.colRange(4, main_output.sizes()[1] - 32);
        cv::Point class_id_point;
        double score;
        cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        if (score > conf_threshold) {
            // score\in [0, 1), 置信度太小的结果可以舍弃掉
            class_ids.push_back(class_id_point.x);
            confidences.push_back(score);
            // 预测框是在640x640的图片上预测的, 但是分割结果只有160x160
            const float mask_scale = 0.25f; // 160/640 = 0.25

            const cv::Mat detection_box = result.colRange(0, 4);
            const cv::Rect mask_box = toBox(detection_box * mask_scale,
                cv::Rect(0, 0, 160, 160));
            const cv::Rect image_box = toBox(detection_box,
                cv::Rect(0, 0, image.cols, image.rows));
            mask_boxes.push_back(mask_box);
            boxes.push_back(image_box);

            /// 处理分割部分的结果
            masks.push_back(result.colRange(main_output.sizes()[1] - 32, main_output.sizes()[1]));
        }
    }
    // NMS, 消除具有较低置信度的冗余重叠框
    std::vector<int> nms_indexes;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_indexes);

    std::vector<SegmentOutput> segmentOutputs;
    for (const int index : nms_indexes) {
        SegmentOutput segmentOutput;
        segmentOutput._id = class_ids[index];
        segmentOutput._confidence = confidences[index];
        segmentOutput._box = boxes[index];
        // sigmoid运算
        cv::Mat m;
        cv::exp(-masks[index] * segment_buffer, m);
        m = 1.0f / (1.0f + m);
        m = m.reshape(1, 160); // 1x25600 -> 160x160
        cv::resize(m(mask_boxes[index]) > 0.5f,
            segmentOutput._boxMask, segmentOutput._box.size());

        segmentOutputs.push_back(segmentOutput);
    }
    draw_result(resize_image, segmentOutputs);

    cv::imshow("resize_image", resize_image);
    cv::waitKey(0);
    cv::destroyAllWindows();


}