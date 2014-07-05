#ifndef _UNDO_CHEN_H_
#define _UNDO_CHEN_H_

#include <stdint.h>

__device__ inline unsigned char
undo_chen_byte_gpu(const unsigned char c, const unsigned char cp,
    const unsigned char s)
{
  return (unsigned char) (((int) (c ^ s) - (int) cp) % 256);
}

/**
 * Reverses Chen transformation (b_i = (b'_i ^ x'_i - b'_{i-1}) % 256.
 * A block processes a row of the input image.
 * A thread process a complete pixel consisting of three components (R, G, B).
 */
__global__
void
undo_chen_gpu(unsigned char *in, unsigned char *out,
    const size_t pixels_per_row)
{
  // Shared memory layout:
  // (R,G,B),(R,G,B),...,(R,G,B)
  // Each thread processes one pixel, i.e., three bytes.
  // First element of array is used to store right-most pixel from previous row.
  extern __shared__ unsigned char s_row[];
  const int row = blockIdx.x;
  const size_t off = row * 3 * pixels_per_row;

  // check if thread is out-of-bounds
  if (threadIdx.x >= pixels_per_row) {
    return;
  }

  // fetch row from global memory and store it in shared memory
  // fetch first third of row, then second third, and finally third third
  // from global memory.
#pragma unroll 3
  for (int i = 0; i < 3; i++) {
    // leave space for previous row pixel
    s_row[3 + i * pixels_per_row + threadIdx.x] = in[off + i * pixels_per_row
        + threadIdx.x];
  }
  // fetch first pixel (right-most element from previous row or key init
  // values)
  if (row > 0) {
    if (threadIdx.x <= 2) {
      s_row[threadIdx.x] = in[off - 3 + threadIdx.x];
    }
  } else {
    if (threadIdx.x <= 2) {
      s_row[threadIdx.x] = const_p0[threadIdx.x];
    }
  }

  __syncthreads();
  // process R, then G, then B bytes
#pragma unroll 3
  for (int i = 0; i < 3; i++) {
    // get right sequence (sx, sy, sz) and find correct row in sequence.
    const unsigned char *s = const_seqs[i] + row * pixels_per_row;
    // thread 0 is responsible for 1st pixel
    // thread 1 is responsible for 2nd pixel, and so on
    size_t index = 3 + 3 * threadIdx.x + i;
    unsigned char seq_element = s[threadIdx.x];
    unsigned char new_byte = undo_chen_byte_gpu(s_row[index], s_row[index - 3],
        seq_element);
    // let every thead obtain its byte before overwriting it (subsequent thread
    // needs old value that predecessor thread updates)
    __syncthreads();
    s_row[index] = new_byte;
    // no __syncthreads required here, since color channel transformation
    // are independent
  }

  __syncthreads();
  // write first third, then second third, and finally third third to global
  // memory.
#pragma unroll 3
  for (int i = 0; i < 3; i++) {
    // do not write previous pixel to array
    out[off + i * pixels_per_row + threadIdx.x] = s_row[3 + i * pixels_per_row
        + threadIdx.x];
  }
}

#endif // #ifndef _UNDO_CHEN_H_
