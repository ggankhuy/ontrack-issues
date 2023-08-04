hipcc --std=c++14 transpoce-rocm.cpp
nvcc --expt-relaxed-constexpr transpoce-cuda.cu
