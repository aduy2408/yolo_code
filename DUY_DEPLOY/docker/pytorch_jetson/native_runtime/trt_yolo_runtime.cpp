#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kInputSize = 640;
constexpr int kChannels = 3;
constexpr int kClasses = 80;
constexpr float kPadValue = 114.0F / 255.0F;

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << message << '\n';
        }
    }
};

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
using TrtPtr = std::unique_ptr<T, void (*)(T*)>;

template <typename T>
TrtPtr<T> own(T* pointer) {
    return {pointer, [](T* value) {
                if (value) {
                    value->destroy();
                }
            }};
}

struct Letterbox {
    std::vector<float> chw;
    float scale;
    int pad_x;
    int pad_y;
};

Letterbox preprocess(const cv::Mat& bgr) {
    const float scale = std::min(static_cast<float>(kInputSize) / bgr.cols,
                                 static_cast<float>(kInputSize) / bgr.rows);
    const int resized_width = std::round(bgr.cols * scale);
    const int resized_height = std::round(bgr.rows * scale);
    const int pad_x = (kInputSize - resized_width) / 2;
    const int pad_y = (kInputSize - resized_height) / 2;

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(resized_width, resized_height), 0, 0, cv::INTER_LINEAR);
    cv::Mat canvas(kInputSize, kInputSize, CV_8UC3,
                   cv::Scalar(static_cast<uint8_t>(114), static_cast<uint8_t>(114), static_cast<uint8_t>(114)));
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, resized_width, resized_height)));

    std::vector<float> chw(kChannels * kInputSize * kInputSize, kPadValue);
    const size_t plane = static_cast<size_t>(kInputSize) * kInputSize;
    for (int y = 0; y < kInputSize; ++y) {
        const auto* row = canvas.ptr<cv::Vec3b>(y);
        for (int x = 0; x < kInputSize; ++x) {
            const auto pixel = row[x];
            const size_t offset = static_cast<size_t>(y) * kInputSize + x;
            chw[offset] = pixel[2] / 255.0F;
            chw[plane + offset] = pixel[1] / 255.0F;
            chw[2 * plane + offset] = pixel[0] / 255.0F;
        }
    }
    return {std::move(chw), scale, pad_x, pad_y};
}

struct Detection {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
};

float intersection_over_union(const Detection& a, const Detection& b) {
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float intersection = std::max(0.0F, x2 - x1) * std::max(0.0F, y2 - y1);
    const float area_a = std::max(0.0F, a.x2 - a.x1) * std::max(0.0F, a.y2 - a.y1);
    const float area_b = std::max(0.0F, b.x2 - b.x1) * std::max(0.0F, b.y2 - b.y1);
    return intersection / std::max(area_a + area_b - intersection, 1e-12F);
}

std::vector<Detection> postprocess(const float* output, int image_width, int image_height,
                                   float scale, int pad_x, int pad_y, float confidence_threshold,
                                   float nms_iou, int max_detections) {
    constexpr int kCandidates = 8400;
    std::vector<Detection> candidates;
    candidates.reserve(kCandidates);
    for (int index = 0; index < kCandidates; ++index) {
        const float center_x = output[index];
        const float center_y = output[kCandidates + index];
        const float width = output[2 * kCandidates + index];
        const float height = output[3 * kCandidates + index];
        int best_class = 0;
        float best_score = output[4 * kCandidates + index];
        for (int class_id = 1; class_id < kClasses; ++class_id) {
            const float score = output[(4 + class_id) * kCandidates + index];
            if (score > best_score) {
                best_score = score;
                best_class = class_id;
            }
        }
        if (best_score < confidence_threshold) {
            continue;
        }
        Detection detection{
            (center_x - width / 2.0F - pad_x) / scale,
            (center_y - height / 2.0F - pad_y) / scale,
            (center_x + width / 2.0F - pad_x) / scale,
            (center_y + height / 2.0F - pad_y) / scale,
            best_score,
            best_class,
        };
        detection.x1 = std::clamp(detection.x1, 0.0F, static_cast<float>(image_width));
        detection.y1 = std::clamp(detection.y1, 0.0F, static_cast<float>(image_height));
        detection.x2 = std::clamp(detection.x2, 0.0F, static_cast<float>(image_width));
        detection.y2 = std::clamp(detection.y2, 0.0F, static_cast<float>(image_height));
        candidates.push_back(detection);
    }

    std::sort(candidates.begin(), candidates.end(), [](const Detection& left, const Detection& right) {
        return left.score > right.score;
    });
    std::vector<Detection> selected;
    selected.reserve(std::min(max_detections, static_cast<int>(candidates.size())));
    for (const Detection& candidate : candidates) {
        bool suppressed = false;
        for (const Detection& kept : selected) {
            if (candidate.class_id == kept.class_id && intersection_over_union(candidate, kept) > nms_iou) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            selected.push_back(candidate);
            if (static_cast<int>(selected.size()) == max_detections) {
                break;
            }
        }
    }
    return selected;
}

std::vector<char> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open engine: " + path);
    }
    const auto size = file.tellg();
    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0);
    file.read(data.data(), size);
    return data;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 6) {
        std::cerr << "Usage: trt_yolo_runtime ENGINE IMAGE [CONF=0.001] [NMS_IOU=0.5] [MAX_DET=300]\n";
        return 2;
    }
    try {
        const float confidence_threshold = argc >= 4 ? std::stof(argv[3]) : 0.001F;
        const float nms_iou = argc >= 5 ? std::stof(argv[4]) : 0.5F;
        const int max_detections = argc >= 6 ? std::stoi(argv[5]) : 300;
        const cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Cannot read image: " + std::string(argv[2]));
        }
        const Letterbox input = preprocess(image);
        const std::vector<char> engine_bytes = read_file(argv[1]);
        Logger logger;
        auto runtime = own(nvinfer1::createInferRuntime(logger));
        auto engine = own(runtime->deserializeCudaEngine(engine_bytes.data(), engine_bytes.size()));
        if (!engine) {
            throw std::runtime_error("TensorRT failed to deserialize engine");
        }
        auto context = own(engine->createExecutionContext());
        if (!context) {
            throw std::runtime_error("TensorRT failed to create execution context");
        }
        if (engine->getNbIOTensors() != 2) {
            throw std::runtime_error("Expected one input and one output tensor");
        }

        const char* input_name = nullptr;
        const char* output_name = nullptr;
        for (int index = 0; index < engine->getNbIOTensors(); ++index) {
            const char* name = engine->getIOTensorName(index);
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                input_name = name;
            } else {
                output_name = name;
            }
        }
        if (!input_name || !output_name) {
            throw std::runtime_error("Missing input/output tensor");
        }
        const auto input_shape = engine->getTensorShape(input_name);
        const auto output_shape = engine->getTensorShape(output_name);
        if (input_shape.nbDims != 4 || input_shape.d[0] != 1 || input_shape.d[1] != kChannels ||
            input_shape.d[2] != kInputSize || input_shape.d[3] != kInputSize ||
            engine->getTensorDataType(input_name) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("Expected FLOAT input with shape [1,3,640,640]");
        }
        if (output_shape.nbDims != 3 || output_shape.d[0] != 1 || output_shape.d[1] != 4 + kClasses ||
            output_shape.d[2] != 8400 || engine->getTensorDataType(output_name) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("Expected FLOAT YOLO output with shape [1,84,8400]");
        }

        constexpr size_t input_elements = static_cast<size_t>(kChannels) * kInputSize * kInputSize;
        constexpr size_t output_elements = static_cast<size_t>(84) * 8400;
        float* device_input = nullptr;
        float* device_output = nullptr;
        cudaStream_t stream = nullptr;
        check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
        check_cuda(cudaMalloc(&device_input, input_elements * sizeof(float)), "cudaMalloc(input)");
        check_cuda(cudaMalloc(&device_output, output_elements * sizeof(float)), "cudaMalloc(output)");
        check_cuda(cudaMemcpyAsync(device_input, input.chw.data(), input_elements * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync(input)");
        if (!context->setTensorAddress(input_name, device_input) ||
            !context->setTensorAddress(output_name, device_output) || !context->enqueueV3(stream)) {
            throw std::runtime_error("TensorRT enqueue failed");
        }
        std::vector<float> output(output_elements);
        check_cuda(cudaMemcpyAsync(output.data(), device_output, output.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync(output)");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        const auto detections = postprocess(output.data(), image.cols, image.rows, input.scale, input.pad_x,
                                            input.pad_y, confidence_threshold, nms_iou, max_detections);
        for (const Detection& detection : detections) {
            std::cout << detection.class_id << ' ' << detection.score << ' ' << detection.x1 << ' '
                      << detection.y1 << ' ' << detection.x2 << ' ' << detection.y2 << '\n';
        }
        cudaFree(device_input);
        cudaFree(device_output);
        cudaStreamDestroy(stream);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
