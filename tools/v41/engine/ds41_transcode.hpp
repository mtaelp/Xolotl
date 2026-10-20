// ds41_transcode.hpp - PTQ1_0 (ternary, type 143) -> Q2_0 (type 42) on the device, value for value (see the .cu).
#pragma once

#include <cstddef>
#include <cstdint>

/// Bytes of the Q2_0 copy of `ptq1_bytes` of PTQ1_0 blocks (28 B / 128 weights -> 2 x 18 B).
size_t ds41_q2_0_bytes_from_ptq1(size_t ptq1_bytes);
/// Transcodes `ptq1_bytes` of PTQ1_0 blocks at `src` (device) into Q2_0 blocks at `dst` (device) on `stream`
/// (a cudaStream_t).  Whole matrices or rows: block b becomes Q2_0 blocks 2b and 2b+1.
void ds41_ptq1_to_q2_0(const void* src, void* dst, size_t ptq1_bytes, void* stream);
/// The same for `count` matrices of `ptq1_bytes` each: matrix c from src + c * src_stride to dst + c * dst_stride
/// (bytes) - one launch for a run of staged experts.
void ds41_ptq1_to_q2_0_strided(const void* src, size_t src_stride, void* dst, size_t dst_stride, size_t ptq1_bytes,
                               int count, void* stream);
/// --gpu2: dst[rows[r] * H + i] += src[r * H + i] for r < nrows (all device pointers; src may be mapped host memory) on
/// `stream` (a cudaStream_t).
void ds41_add_rows_f32(float* dst, const int32_t* rows, const float* src, int nrows, int64_t H, void* stream);
/// --gpu2 worker: one tiny block spinning ~`us` microseconds at `clock_khz` (keeps the card's clocks up), on `stream`.
void ds41_spin_us(double us, int clock_khz, void* stream);
