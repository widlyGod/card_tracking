//
// Created by 邓昊晴 on 14/6/2020.
//

#include "YoloV5.h"

bool YoloV5::hasGPU = false;
YoloV5 *YoloV5::detector = nullptr;

YoloV5::YoloV5(AAssetManager *mgr, const char *param, const char *bin) {
    Net = new ncnn::Net();
    Net->load_param(mgr, param);
    Net->load_model(mgr, bin);
}

YoloV5::~YoloV5() {
    delete Net;
}

std::vector <BoxInfo>
YoloV5::detect(JNIEnv *env, jobject image, float threshold, float nms_threshold) {
    AndroidBitmapInfo img_size;
    AndroidBitmap_getInfo(env, image, &img_size);
//    ncnn::Mat in_net = ncnn::Mat::from_android_bitmap_resize(env,image,ncnn::Mat::PIXEL_BGR2RGB,input_size/2,input_size/2);
    ncnn::Mat in_net = ncnn::Mat::from_android_bitmap_resize(env, image, ncnn::Mat::PIXEL_RGBA2RGB,
                                                             input_size / 2, input_size / 2);
    float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    float mean[3] = {0, 0, 0};
    in_net.substract_mean_normalize(mean, norm);
    auto ex = Net->create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
//    ex.set_vulkan_compute(hasGPU);
    ex.input(0, in_net);
    std::vector <BoxInfo> result;
    for (const auto &layer: layers) {
        ncnn::Mat blob;
        ex.extract(layer.name.c_str(), blob);
        auto boxes = decode_infer(blob, layer.stride, {(int) img_size.width, (int) img_size.height},
                                  input_size, num_class, layer.anchors, threshold);
        result.insert(result.begin(), boxes.begin(), boxes.end());
    }
    nms(result, nms_threshold);
    return result;
}

inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

std::vector <BoxInfo>
YoloV5::decode_infer(ncnn::Mat &data, int stride, const cv::Size &frame_size, int net_size,
                     int num_classes, const std::vector <cv::Size> &anchors, float threshold) {
    std::vector <BoxInfo> result;
    int grid_size = data.h;
    float cx, cy, w, h;
    for (int i = 0; i < grid_size; i++) {
        const float *values = data.row(i);

        BoxInfo object;
        object.label = values[0];
        object.score = values[1];

        object.x1 = values[2] * frame_size.width;
        object.y1 = values[3] * frame_size.height;
        object.x2 = values[4] * frame_size.width;
        object.y2 = values[5] * frame_size.height;
        result.push_back(object);
    }

    return result;
}

void YoloV5::nms(std::vector <BoxInfo> &input_boxes, float NMS_THRESH) {
    std::sort(input_boxes.begin(), input_boxes.end(),
              [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            } else {
                j++;
            }
        }
    }
}
