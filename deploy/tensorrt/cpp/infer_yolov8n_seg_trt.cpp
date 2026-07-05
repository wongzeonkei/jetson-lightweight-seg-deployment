#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace nvinfer1;

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

template <typename T>
struct TRTDeleter {
    void operator()(T* obj) const {
        if (obj) {
            delete obj;
        }
    }
};

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)          \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

static const std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
};

struct LetterboxInfo {
    float scale;
    int pad_x;
    int pad_y;
};

struct Detection {
    int class_id;
    std::string class_name;
    float score;
    cv::Rect box;
    int mask_pixels;
};

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

int64_t volume(const Dims& dims) {
    int64_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        v *= dims.d[i];
    }
    return v;
}

std::string dimsToString(const Dims& dims) {
    std::string s = "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        s += std::to_string(dims.d[i]);
        if (i + 1 < dims.nbDims) {
            s += ", ";
        }
    }
    s += "]";
    return s;
}

std::vector<char> readBinaryFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open engine file: " + path);
    }

    file.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

cv::Mat letterbox(const cv::Mat& img, int input_size, LetterboxInfo& info) {
    int h = img.rows;
    int w = img.cols;

    float scale = std::min(static_cast<float>(input_size) / w,
                           static_cast<float>(input_size) / h);

    int resized_w = static_cast<int>(std::round(w * scale));
    int resized_h = static_cast<int>(std::round(h * scale));

    int dw = input_size - resized_w;
    int dh = input_size - resized_h;

    int left = static_cast<int>(std::round(dw / 2.0f - 0.1f));
    int right = static_cast<int>(std::round(dw / 2.0f + 0.1f));
    int top = static_cast<int>(std::round(dh / 2.0f - 0.1f));
    int bottom = static_cast<int>(std::round(dh / 2.0f + 0.1f));

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat padded;
    cv::copyMakeBorder(
        resized,
        padded,
        top,
        bottom,
        left,
        right,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114)
    );

    info.scale = scale;
    info.pad_x = left;
    info.pad_y = top;

    return padded;
}

std::vector<float> preprocessImage(const cv::Mat& bgr, int input_size, LetterboxInfo& info) {
    cv::Mat padded = letterbox(bgr, input_size, info);

    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    cv::Mat f32;
    rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);

    std::vector<float> input(1 * 3 * input_size * input_size);

    int channel_size = input_size * input_size;

    for (int y = 0; y < input_size; ++y) {
        for (int x = 0; x < input_size; ++x) {
            cv::Vec3f pixel = f32.at<cv::Vec3f>(y, x);
            input[0 * channel_size + y * input_size + x] = pixel[0];
            input[1 * channel_size + y * input_size + x] = pixel[1];
            input[2 * channel_size + y * input_size + x] = pixel[2];
        }
    }

    return input;
}

cv::Rect2f xywhToRect(float cx, float cy, float w, float h) {
    float x1 = cx - w / 2.0f;
    float y1 = cy - h / 2.0f;
    return cv::Rect2f(x1, y1, w, h);
}

cv::Mat cropMask(const cv::Mat& mask, const cv::Rect2f& box) {
    cv::Mat cropped = cv::Mat::zeros(mask.size(), mask.type());

    int x1 = std::max(0, std::min(static_cast<int>(box.x), mask.cols - 1));
    int y1 = std::max(0, std::min(static_cast<int>(box.y), mask.rows - 1));
    int x2 = std::max(0, std::min(static_cast<int>(box.x + box.width), mask.cols));
    int y2 = std::max(0, std::min(static_cast<int>(box.y + box.height), mask.rows));

    if (x2 > x1 && y2 > y1) {
        mask(cv::Rect(x1, y1, x2 - x1, y2 - y1)).copyTo(
            cropped(cv::Rect(x1, y1, x2 - x1, y2 - y1))
        );
    }

    return cropped;
}

std::vector<Detection> postprocess(
    const std::vector<float>& output0,
    const std::vector<float>& output1,
    const cv::Mat& original,
    const LetterboxInfo& info,
    cv::Mat& vis,
    float conf_thres = 0.25f,
    float iou_thres = 0.45f,
    int input_size = 320
) {
    const int num_channels = 116;
    const int num_candidates = 2100;
    const int num_classes = 80;
    const int num_masks = 32;
    const int proto_h = 80;
    const int proto_w = 80;
    const int proto_area = proto_h * proto_w;

    std::vector<cv::Rect> nms_boxes;
    std::vector<cv::Rect2f> boxes_input;
    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> mask_coeffs;

    for (int i = 0; i < num_candidates; ++i) {
        float cx = output0[0 * num_candidates + i];
        float cy = output0[1 * num_candidates + i];
        float w = output0[2 * num_candidates + i];
        float h = output0[3 * num_candidates + i];

        int best_cls = -1;
        float best_score = -1.0f;

        for (int c = 0; c < num_classes; ++c) {
            float score = output0[(4 + c) * num_candidates + i];
            if (score > best_score) {
                best_score = score;
                best_cls = c;
            }
        }

        if (best_score < conf_thres) {
            continue;
        }

        cv::Rect2f rect = xywhToRect(cx, cy, w, h);

        int x = static_cast<int>(std::round(rect.x));
        int y = static_cast<int>(std::round(rect.y));
        int rw = static_cast<int>(std::round(rect.width));
        int rh = static_cast<int>(std::round(rect.height));

        nms_boxes.emplace_back(x, y, rw, rh);
        boxes_input.push_back(rect);
        scores.push_back(best_score);
        class_ids.push_back(best_cls);

        std::vector<float> coeff(num_masks);
        for (int m = 0; m < num_masks; ++m) {
            coeff[m] = output0[(84 + m) * num_candidates + i];
        }
        mask_coeffs.push_back(coeff);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(nms_boxes, scores, conf_thres, iou_thres, indices);

    vis = original.clone();
    std::vector<Detection> detections;

    int h0 = original.rows;
    int w0 = original.cols;

    for (int idx : indices) {
        cv::Rect2f box_in = boxes_input[idx];

        float x1 = std::max(0.0f, std::min(box_in.x, static_cast<float>(input_size)));
        float y1 = std::max(0.0f, std::min(box_in.y, static_cast<float>(input_size)));
        float x2 = std::max(0.0f, std::min(box_in.x + box_in.width, static_cast<float>(input_size)));
        float y2 = std::max(0.0f, std::min(box_in.y + box_in.height, static_cast<float>(input_size)));

        cv::Rect2f clipped_box(x1, y1, x2 - x1, y2 - y1);

        cv::Mat mask80(proto_h, proto_w, CV_32FC1);

        for (int p = 0; p < proto_area; ++p) {
            float value = 0.0f;
            for (int m = 0; m < num_masks; ++m) {
                value += mask_coeffs[idx][m] * output1[m * proto_area + p];
            }
            mask80.at<float>(p / proto_w, p % proto_w) = sigmoid(value);
        }

        cv::Mat mask320;
        cv::resize(mask80, mask320, cv::Size(input_size, input_size), 0, 0, cv::INTER_LINEAR);
        mask320 = cropMask(mask320, clipped_box);

        cv::Mat mask_bin;
        cv::threshold(mask320, mask_bin, 0.5, 1.0, cv::THRESH_BINARY);
        mask_bin.convertTo(mask_bin, CV_8UC1);

        int x_unpad1 = info.pad_x;
        int y_unpad1 = info.pad_y;
        int x_unpad2 = input_size - info.pad_x;
        int y_unpad2 = input_size - info.pad_y;

        x_unpad1 = std::max(0, std::min(x_unpad1, input_size - 1));
        y_unpad1 = std::max(0, std::min(y_unpad1, input_size - 1));
        x_unpad2 = std::max(0, std::min(x_unpad2, input_size));
        y_unpad2 = std::max(0, std::min(y_unpad2, input_size));

        cv::Mat mask_unpad = mask_bin(cv::Rect(
            x_unpad1,
            y_unpad1,
            x_unpad2 - x_unpad1,
            y_unpad2 - y_unpad1
        ));

        cv::Mat mask_orig;
        cv::resize(mask_unpad, mask_orig, cv::Size(w0, h0), 0, 0, cv::INTER_NEAREST);

        float bx1 = (x1 - info.pad_x) / info.scale;
        float by1 = (y1 - info.pad_y) / info.scale;
        float bx2 = (x2 - info.pad_x) / info.scale;
        float by2 = (y2 - info.pad_y) / info.scale;

        int ix1 = std::max(0, std::min(static_cast<int>(bx1), w0 - 1));
        int iy1 = std::max(0, std::min(static_cast<int>(by1), h0 - 1));
        int ix2 = std::max(0, std::min(static_cast<int>(bx2), w0 - 1));
        int iy2 = std::max(0, std::min(static_cast<int>(by2), h0 - 1));

        int cls_id = class_ids[idx];
        std::string cls_name = cls_id >= 0 && cls_id < static_cast<int>(COCO_CLASSES.size())
            ? COCO_CLASSES[cls_id]
            : std::to_string(cls_id);

        cv::Scalar color(
            (37 * (cls_id + 1)) % 255,
            (17 * (cls_id + 3)) % 255,
            (29 * (cls_id + 5)) % 255
        );

        cv::Mat colored = cv::Mat::zeros(vis.size(), vis.type());
        colored.setTo(color, mask_orig);

        cv::addWeighted(vis, 1.0, colored, 0.45, 0.0, vis);
        cv::rectangle(vis, cv::Point(ix1, iy1), cv::Point(ix2, iy2), color, 2);

        std::string label = cls_name + " " + cv::format("%.2f", scores[idx]);
        cv::putText(
            vis,
            label,
            cv::Point(ix1, std::max(0, iy1 - 5)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        );

        int mask_pixels = cv::countNonZero(mask_orig);

        Detection det;
        det.class_id = cls_id;
        det.class_name = cls_name;
        det.score = scores[idx];
        det.box = cv::Rect(ix1, iy1, std::max(0, ix2 - ix1), std::max(0, iy2 - iy1));
        det.mask_pixels = mask_pixels;
        detections.push_back(det);
    }

    return detections;
}

int main(int argc, char** argv) {
    std::string engine_path = "deploy/tensorrt/engines/yolov8n_seg_320_fp16.engine";
    std::string image_path = "data/samples/bus.jpg";
    std::string output_path = "outputs/tensorrt/cpp/yolov8n_seg_320_bus_trt_cpp.jpg";

    if (argc >= 2) engine_path = argv[1];
    if (argc >= 3) image_path = argv[2];
    if (argc >= 4) output_path = argv[3];

    const int input_size = 320;

    Logger logger;

    std::cout << "Engine: " << engine_path << std::endl;
    std::cout << "Image: " << image_path << std::endl;
    std::cout << "Output: " << output_path << std::endl;

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return 1;
    }

    LetterboxInfo info{};
    std::vector<float> input = preprocessImage(img, input_size, info);

    std::cout << "Original shape: " << img.cols << "x" << img.rows << std::endl;
    std::cout << "Input shape: 1x3x320x320" << std::endl;
    std::cout << "scale: " << info.scale << ", pad_x: " << info.pad_x << ", pad_y: " << info.pad_y << std::endl;

    std::vector<char> engine_data = readBinaryFile(engine_path);

    std::unique_ptr<IRuntime, TRTDeleter<IRuntime>> runtime(
        createInferRuntime(logger)
    );

    if (!runtime) {
        std::cerr << "Failed to create TensorRT runtime." << std::endl;
        return 1;
    }

    std::unique_ptr<ICudaEngine, TRTDeleter<ICudaEngine>> engine(
        runtime->deserializeCudaEngine(engine_data.data(), engine_data.size())
    );

    if (!engine) {
        std::cerr << "Failed to deserialize engine." << std::endl;
        return 1;
    }

    std::unique_ptr<IExecutionContext, TRTDeleter<IExecutionContext>> context(
        engine->createExecutionContext()
    );

    if (!context) {
        std::cerr << "Failed to create execution context." << std::endl;
        return 1;
    }

    int nb_bindings = engine->getNbBindings();
    std::cout << "Number of bindings: " << nb_bindings << std::endl;

    int input_idx = -1;
    int output0_idx = -1;
    int output1_idx = -1;

    for (int i = 0; i < nb_bindings; ++i) {
        std::string name = engine->getBindingName(i);
        Dims dims = engine->getBindingDimensions(i);

        std::cout << "Binding " << i
                  << " name=" << name
                  << " is_input=" << engine->bindingIsInput(i)
                  << " dims=" << dimsToString(dims)
                  << std::endl;

        if (name == "images") input_idx = i;
        if (name == "output0") output0_idx = i;
        if (name == "output1") output1_idx = i;
    }

    if (input_idx < 0 || output0_idx < 0 || output1_idx < 0) {
        std::cerr << "Failed to find required bindings: images/output0/output1" << std::endl;
        return 1;
    }

    Dims input_dims = engine->getBindingDimensions(input_idx);
    Dims output0_dims = engine->getBindingDimensions(output0_idx);
    Dims output1_dims = engine->getBindingDimensions(output1_idx);

    size_t input_count = static_cast<size_t>(volume(input_dims));
    size_t output0_count = static_cast<size_t>(volume(output0_dims));
    size_t output1_count = static_cast<size_t>(volume(output1_dims));

    std::cout << "input_count: " << input_count << std::endl;
    std::cout << "output0_count: " << output0_count << std::endl;
    std::cout << "output1_count: " << output1_count << std::endl;

    std::vector<float> output0(output0_count);
    std::vector<float> output1(output1_count);

    void* buffers[3] = {nullptr, nullptr, nullptr};

    CHECK_CUDA(cudaMalloc(&buffers[input_idx], input_count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers[output0_idx], output0_count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers[output1_idx], output1_count * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(
        buffers[input_idx],
        input.data(),
        input_count * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    for (int i = 0; i < 10; ++i) {
        bool ok = context->executeV2(buffers);
        if (!ok) {
            std::cerr << "TensorRT warmup executeV2 failed." << std::endl;
            return 1;
        }
    }

    const int repeat = 100;
    std::vector<double> times;
    times.reserve(repeat);

    for (int i = 0; i < repeat; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();

        CHECK_CUDA(cudaMemcpy(
            buffers[input_idx],
            input.data(),
            input_count * sizeof(float),
            cudaMemcpyHostToDevice
        ));

        bool ok = context->executeV2(buffers);
        if (!ok) {
            std::cerr << "TensorRT executeV2 failed." << std::endl;
            return 1;
        }

        CHECK_CUDA(cudaMemcpy(
            output0.data(),
            buffers[output0_idx],
            output0_count * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        CHECK_CUDA(cudaMemcpy(
            output1.data(),
            buffers[output1_idx],
            output1_count * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        CHECK_CUDA(cudaDeviceSynchronize());

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);
    }

    double mean_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    std::vector<double> sorted = times;
    std::sort(sorted.begin(), sorted.end());
    double median_ms = sorted[sorted.size() / 2];

    std::cout << "C++ TensorRT latency over " << repeat << " runs:" << std::endl;
    std::cout << " mean ms: " << mean_ms << std::endl;
    std::cout << " median ms: " << median_ms << std::endl;
    std::cout << " min ms: " << sorted.front() << std::endl;
    std::cout << " max ms: " << sorted.back() << std::endl;

    cv::Mat vis;
    std::vector<Detection> detections = postprocess(output0, output1, img, info, vis);

    std::cout << "Detections:" << std::endl;
    for (const auto& d : detections) {
        std::cout << "{class_id: " << d.class_id
                  << ", class_name: " << d.class_name
                  << ", score: " << d.score
                  << ", box: [" << d.box.x << ", " << d.box.y
                  << ", " << d.box.x + d.box.width
                  << ", " << d.box.y + d.box.height
                  << "], mask_pixels: " << d.mask_pixels
                  << "}" << std::endl;
    }

    bool saved = cv::imwrite(output_path, vis);
    if (!saved) {
        std::cerr << "Failed to save output image: " << output_path << std::endl;
        return 1;
    }

    std::cout << "Saved visualization: " << output_path << std::endl;

    CHECK_CUDA(cudaFree(buffers[input_idx]));
    CHECK_CUDA(cudaFree(buffers[output0_idx]));
    CHECK_CUDA(cudaFree(buffers[output1_idx]));

    return 0;
}
