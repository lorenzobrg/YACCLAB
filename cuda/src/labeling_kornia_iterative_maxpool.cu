// Copyright (c) 2026, the YACCLAB contributors, as
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Kornia-inspired Connected Components Labeling:
// iterative 3x3 max propagation on foreground pixels.

#include <opencv2/cudafeatures2d.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "labeling_algorithms.h"
#include "register.h"

#define BLOCK_ROWS 16
#define BLOCK_COLS 16

namespace {

constexpr int KORNIA_NUM_ITERATIONS = 100;

__global__ void Initialization(const cv::cuda::PtrStepSzb img, cv::cuda::PtrStepSzi labels) {
    const unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned img_index = row * (img.step / img.elem_size) + col;
    const unsigned labels_index = row * (labels.step / labels.elem_size) + col;

    if (row < labels.rows && col < labels.cols) {
        if (img[img_index] > 0) {
            labels[labels_index] = static_cast<int>(labels_index + 1);
        }
        else {
            labels[labels_index] = 0;
        }
    }
}

__global__ void MaxPoolIteration(const cv::cuda::PtrStepSzi labels_in, cv::cuda::PtrStepSzi labels_out) {
    const unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= labels_in.rows || col >= labels_in.cols) {
        return;
    }

    const unsigned stride = labels_in.step / labels_in.elem_size;
    const unsigned idx = row * stride + col;
    const int current = labels_in[idx];

    if (current == 0) {
        labels_out[idx] = 0;
        return;
    }

    int max_label = current;

    for (int dr = -1; dr <= 1; ++dr) {
        const int rr = static_cast<int>(row) + dr;
        if (rr < 0 || rr >= labels_in.rows) {
            continue;
        }
        for (int dc = -1; dc <= 1; ++dc) {
            const int cc = static_cast<int>(col) + dc;
            if (cc < 0 || cc >= labels_in.cols) {
                continue;
            }
            const int neighbor = labels_in[rr * stride + cc];
            if (neighbor > max_label) {
                max_label = neighbor;
            }
        }
    }

    labels_out[idx] = max_label;
}

}  // namespace

class KORNIA : public GpuLabeling2D<Connectivity2D::CONN_8> {
private:
    dim3 grid_size_;
    dim3 block_size_;
    cv::cuda::GpuMat d_img_labels_tmp_;

public:
    KORNIA() {}

    void PerformLabeling() {
        d_img_labels_.create(d_img_.size(), CV_32SC1);
        d_img_labels_tmp_.create(d_img_.size(), CV_32SC1);

        LaunchKornia(BLOCK_COLS, BLOCK_ROWS);
        cudaDeviceSynchronize();
    }

private:
    double Alloc() {
        perf_.start();
        d_img_labels_.create(d_img_.size(), CV_32SC1);
        d_img_labels_tmp_.create(d_img_.size(), CV_32SC1);
        cudaDeviceSynchronize();
        return perf_.stop();
    }

    double Dealloc() {
        perf_.start();
        d_img_labels_tmp_.release();
        perf_.stop();
        return perf_.last();
    }

    void LaunchKornia(int block_cols, int block_rows) {
        grid_size_ = dim3((d_img_.cols + block_cols - 1) / block_cols, (d_img_.rows + block_rows - 1) / block_rows, 1);
        block_size_ = dim3(block_cols, block_rows, 1);

        Initialization<<<grid_size_, block_size_>>>(d_img_, d_img_labels_);

        for (int i = 0; i < KORNIA_NUM_ITERATIONS; ++i) {
            MaxPoolIteration<<<grid_size_, block_size_>>>(d_img_labels_, d_img_labels_tmp_);
            d_img_labels_.swap(d_img_labels_tmp_);
        }
    }

public:
    void PerformLabelingWithSteps() {
        const double alloc_timing = Alloc();

        perf_.start();
        LaunchKornia(BLOCK_COLS, BLOCK_ROWS);
        cudaDeviceSynchronize();
        perf_.stop();
        perf_.store(Step(StepType::ALL_SCANS), perf_.last());

        const double dealloc_timing = Dealloc();
        perf_.store(Step(StepType::ALLOC_DEALLOC), alloc_timing + dealloc_timing);
    }

    void PerformLabelingBlocksize(int x, int y, int z) override {
        d_img_labels_.create(d_img_.size(), CV_32SC1);
        d_img_labels_tmp_.create(d_img_.size(), CV_32SC1);

        LaunchKornia(x, y);
        cudaDeviceSynchronize();
    }
};

REGISTER_LABELING(KORNIA);
REGISTER_KERNELS(KORNIA, Initialization, MaxPoolIteration)
