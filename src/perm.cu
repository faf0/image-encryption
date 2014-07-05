#ifndef _PERM_H_
#define _PERM_H_

#include <stdint.h>

#include "libppm.h"

#define BIT_SYNC 0

/**
 * Returns a byte which has bit bpos from b set at position outpos.
 * The other bits are zero.
 */
__device__ inline unsigned char
set_bit_gpu(unsigned char b, int bpos, int outpos)
{
  return ((b >> bpos) & 1) << outpos;
}

__device__ inline void
s_permute(const unsigned char *s_in, unsigned char *s_out, const int *v,
    size_t pixels_per_row, const bool encrypt)
{
  // each thread processes 3 * 8 = 24 bits
  // let N be the number of bytes per row per channel
  // (which is equal to the number of threads per block)
  // first thread processes bits 0..23
  // last thread processes bits (N - 1) * 24..N - 1
#pragma unroll 24
  for (int i = 0; i < (3 * 8); i++) {
    // get in and out BIT positions
    const size_t index = (3 * 8) * threadIdx.x + i;
    // out-of-bounds possible when accessing v and in_bounds is false
    size_t in_col = encrypt ? index : v[index];
    size_t out_col = encrypt ? v[index] : index;
    // map BIT to BYTE position
    // Memory layout:
    // (R,G,B),(R,G,B),...,(R,G,B) with bits_per_row / (3 * 8) (R,G,B) pixels
    int in_pixel = in_col / (3 * 8);
    int in_channel = (in_col / 8) % 3;
    int in_bit = in_col % 8;
    int out_pixel = out_col / (3 * 8);
    int out_channel = (out_col / 8) % 3;
    int out_bit = out_col % 8;
    unsigned char updated_out_byte;

#if BIT_SYNC
    // out-of-bounds access impossible, if enough shared memory exists
    updated_out_byte = set_bit_gpu(s_in[3 * in_pixel + in_channel], in_bit, out_bit);

    // Each thread writes 1st bit, then 2nd bit, and so on, if bit was potentially changed.
    // Prevents conflicting concurrent access on byte.
#pragma unroll 8
    for (int b = 0; b < 8; b++) {
      if (out_bit == b) {
        s_out[3 * out_pixel + out_channel] |= updated_out_byte;
      }
      __syncthreads();
    }
#else
    // Solution using atomicOr
    // number of shared memory bytes must be multiple of sizeof(unsigned int)
    size_t out_off = 3 * out_pixel + out_channel;
    size_t out_off_int = out_off / sizeof(unsigned int);
    size_t pos = out_off % sizeof(unsigned int);
    unsigned int out_uint;
    
    // out-of-bounds access impossible, if enough shared memory exists
    updated_out_byte = set_bit_gpu(s_in[3 * in_pixel + in_channel], in_bit, out_bit);
    out_uint = updated_out_byte << (pos * 8);

    atomicOr((unsigned int *) s_out + out_off_int, out_uint);
#endif
  }
#if !BIT_SYNC
  // make sure shared memory has been written, before writing it to global memory.
  __syncthreads();
#endif
}

/**
 * Permutes a complete row on a bit-level.
 */
__global__
void
permute_gpu(const unsigned char *in, unsigned char *out, const int *h,
    const int *v, size_t rows, size_t bits_per_row, size_t bytes_shared_mem,
    const bool encrypt)
{
  // Shared memory layout:
  // (R,G,B),(R,G,B),...,(R,G,B) with bits_per_row / (3 * 8) (R,G,B) pixels
  // Each thread processes one pixel.
  size_t bytes_per_row = bits_per_row / 8;
  size_t pixels_per_row = bytes_per_row / 3;
  extern __shared__ unsigned char s_in[];
  unsigned char *s_out = &s_in[bytes_shared_mem / 2];
  const int row = blockIdx.x;

  // in_bounds = (3 * 8) * threadIdx.x < bits_per_row;
  // thus, in_bounds = threadIdx.x < pixels_per_row
  if (threadIdx.x >= pixels_per_row) {
    return;
  }

  // row < rows holds (if grid size equals the number of pixels per row)
  // fetch row from global memory and store it in shared memory
  const size_t in_off = encrypt ? row * bytes_per_row : h[row] * bytes_per_row;
  const size_t out_off = encrypt ? h[row] * bytes_per_row : row * bytes_per_row;

  // fetch first third of row, second third, and third third of row
  // initialize s_out
#pragma unroll 3
  for (int i = 0; i < 3; i++) {
    s_in[i * pixels_per_row + threadIdx.x] = in[in_off + i * pixels_per_row
        + threadIdx.x];
    s_out[i * pixels_per_row + threadIdx.x] = 0;
  }
  __syncthreads();

  // permute the bits of row in shared memory
  s_permute(s_in, s_out, v, pixels_per_row, encrypt);

  // write permuted row to global memory
#pragma unroll 3
  for (int i = 0; i < 3; i++) {
    out[out_off + i * pixels_per_row + threadIdx.x] = s_out[i * pixels_per_row
        + threadIdx.x];
  }
}

#endif // #ifndef _PERM_H_
