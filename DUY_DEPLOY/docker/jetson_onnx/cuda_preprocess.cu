#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

namespace {

__device__ float clamp_float(float value, float low, float high) {
    return fminf(fmaxf(value, low), high);
}

__device__ float read_bilinear(const unsigned char* source, int width, int height, int channels,
                               float x, float y, int channel) {
    x = clamp_float(x, 0.0F, static_cast<float>(width - 1));
    y = clamp_float(y, 0.0F, static_cast<float>(height - 1));
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int x1 = min(x0 + 1, width - 1);
    const int y1 = min(y0 + 1, height - 1);
    const float wx = x - x0;
    const float wy = y - y0;
    const auto at = [&](int px, int py) {
        return static_cast<float>(source[(py * width + px) * channels + channel]);
    };
    const float top = at(x0, y0) + wx * (at(x1, y0) - at(x0, y0));
    const float bottom = at(x0, y1) + wx * (at(x1, y1) - at(x0, y1));
    return (top + wy * (bottom - top)) / 255.0F;
}

__global__ void preprocess_kernel(const unsigned char* source, float* output, int source_width,
                                  int source_height, int output_size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixels = output_size * output_size;
    if (index >= pixels) {
        return;
    }
    const int y = index / output_size;
    const int x = index - y * output_size;
    const float scale = fminf(static_cast<float>(output_size) / source_width,
                              static_cast<float>(output_size) / source_height);
    const int resized_width = static_cast<int>(roundf(source_width * scale));
    const int resized_height = static_cast<int>(roundf(source_height * scale));
    const int pad_x = (output_size - resized_width) / 2;
    const int pad_y = (output_size - resized_height) / 2;
    const int plane = pixels;
    if (x < pad_x || x >= pad_x + resized_width || y < pad_y || y >= pad_y + resized_height) {
        output[index] = 114.0F / 255.0F;
        output[plane + index] = 114.0F / 255.0F;
        output[2 * plane + index] = 114.0F / 255.0F;
        return;
    }
    const float source_x = ((x - pad_x + 0.5F) * source_width / resized_width) - 0.5F;
    const float source_y = ((y - pad_y + 0.5F) * source_height / resized_height) - 0.5F;
    // Input is BGR; TensorRT receives RGB in CHW order.
    output[index] = read_bilinear(source, source_width, source_height, 3, source_x, source_y, 2);
    output[plane + index] = read_bilinear(source, source_width, source_height, 3, source_x, source_y, 1);
    output[2 * plane + index] = read_bilinear(source, source_width, source_height, 3, source_x, source_y, 0);
}

__global__ void pack_kernel(const unsigned char* source, float* output, int width, int height) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixels = width * height;
    if (index >= pixels) {
        return;
    }
    const int pixel = index * 3;
    output[index] = source[pixel + 2] / 255.0F;
    output[pixels + index] = source[pixel + 1] / 255.0F;
    output[2 * pixels + index] = source[pixel] / 255.0F;
}

__global__ void decode_kernel(const float* prediction, float* boxes, float* scores, int* classes,
                              int num_classes, float confidence) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int candidates = 8400;
    if (index >= candidates) {
        return;
    }
    int best_class = 0;
    float best_score = prediction[4 * candidates + index];
    for (int class_id = 1; class_id < num_classes; ++class_id) {
        const float score = prediction[(4 + class_id) * candidates + index];
        if (score > best_score) {
            best_score = score;
            best_class = class_id;
        }
    }
    scores[index] = best_score;
    classes[index] = best_class;
    const float center_x = prediction[index];
    const float center_y = prediction[candidates + index];
    const float width = prediction[2 * candidates + index];
    const float height = prediction[3 * candidates + index];
    boxes[index] = center_x - width / 2.0F;
    boxes[candidates + index] = center_y - height / 2.0F;
    boxes[2 * candidates + index] = center_x + width / 2.0F;
    boxes[3 * candidates + index] = center_y + height / 2.0F;
    if (best_score < confidence) {
        scores[index] = 0.0F;
    }
}

}  // namespace

extern "C" int launch_yolo_decode(const float* prediction, float* boxes, float* scores, int* classes,
                                  int num_classes, float confidence, void* stream) {
    if (!prediction || !boxes || !scores || !classes || num_classes <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    decode_kernel<<<(8400 + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        prediction, boxes, scores, classes, num_classes, confidence);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int launch_yolo_pack(const unsigned char* source, float* output, int width, int height, void* stream) {
    if (!source || !output || width <= 0 || height <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int pixels = width * height;
    pack_kernel<<<(pixels + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        source, output, width, height);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int launch_yolo_preprocess(const unsigned char* source, float* output, int source_width,
                                      int source_height, int output_size, void* stream) {
    if (!source || !output || source_width <= 0 || source_height <= 0 || output_size <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int pixels = output_size * output_size;
    preprocess_kernel<<<(pixels + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        source, output, source_width, source_height, output_size);
    return static_cast<int>(cudaGetLastError());
}
