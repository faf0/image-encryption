#ifndef _APPLY_CHEN_H_
#define _APPLY_CHEN_H_

#include <stdint.h>

/**
 * Applies Chen transformation on complete image from global memory
 * and stores result back in global memory.
 * Assumes that every pixel is encrypted using the same key.
 * Therefore, the Chen sequence is an input parameter and is identical
 * for all images.
 * Each thread in a block processes exactly one image.
 * To be able to use the same Chen sequence, all images are assumed
 * to have the same dimensions.
 */
__global__
void
apply_chen_gpu(size_t number_images, unsigned char *inouts,
    const size_t rows, const size_t cols)
{
  // Each thread processes one image
  extern __shared__ unsigned char seq_row[];
  unsigned char *image = inouts + threadIdx.x * 3 * rows * cols;

  // thread must process an image
  if (threadIdx.x >= number_images) {
    return;
  }

  for (int ch = 0; ch < 3; ch++) {
    const unsigned char *s = const_seqs[ch];
    unsigned char pre_pixel_byte = const_p0[ch];

    for (int row = 0; row < rows; row++) {
      size_t row_off = row * cols;
      // fetch sequence row to shared memory
      for (size_t col_off = 0; col_off < cols; col_off += blockDim.x) {
        const size_t index = col_off + threadIdx.x;

        if (index < cols) {
          seq_row[index] = s[row_off + index];
        }
      }
      __syncthreads();

      for (int col = 0; col < cols; col++) {
        size_t byte_index = 3 * row_off + col + ch;
        unsigned char current_byte = image[byte_index];
        unsigned char new_byte;

        new_byte = (unsigned char) (((int) current_byte + (int) pre_pixel_byte)
            % 256) ^ seq_row[col];
        image[byte_index] = new_byte;
        pre_pixel_byte = new_byte;
      }
    }
  }
}

#endif // #ifndef _APPLY_CHEN_H_
