#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

#include <sys/time.h>
#include <time.h>

#include <iostream>
#include <iomanip>

#include "libppm.h"
#include "enc_cpu.h"

#ifndef __ENABLE_GPU__
#define __ENABLE_GPU__ 1
#endif

#if __ENABLE_GPU__
__constant__ unsigned char const_p0[3];
__constant__ unsigned char *const_seqs[3];

#include "perm.cu"
#include "apply_chen.cu"
#include "undo_chen.cu"
#endif

#define MAX_IMAGE_WIDTH 1024
#define NUMBER_IMAGES 448
#define WARP_SIZE (32)
#define CEIL_TO_WARP_SIZE(i) ((( (i) + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE)
// FIXME set to prevent NaNs. Different in original algorithm.
#define CHEN_SCALING (1.0e7f)
#define TEST_RUNS 20
#define PRECISION 10
#define IMG_OUT "../img-out/"
#define CSV_DELIMITER ";"

#if __ENABLE_GPU__
static float
permute_gpu_launch(const PPMImage *in, PPMImage *out, const int *h,
    const int *v, const bool encrypt)
{
  size_t image_bytes = sizeof(PPMPixel) * in->x * in->y;
  size_t bits_per_row = 3 * 8 * in->x;
  size_t rows = in->y;
  unsigned char *device_in = 0;
  unsigned char *device_out = 0;
  int *device_h = 0;
  int *device_v = 0;
  size_t h_bytes = sizeof(int) * rows;
  size_t v_bytes = sizeof(int) * bits_per_row;
  size_t grid_size;
  size_t block_size;
  size_t bytes_shared_mem;
  float time_ms;
  cudaEvent_t start, stop;

  cudaMalloc((void**) &device_in, image_bytes);
  cudaMalloc((void**) &device_out, image_bytes);
  cudaMalloc((void**) &device_h, h_bytes);
  cudaMalloc((void**) &device_v, v_bytes);

  if (!device_in || !device_out || !device_h || !device_v) {
    fprintf(stderr, "cudaMalloc failed for device arrays!");
    exit(EXIT_FAILURE);
  }

  cudaMemcpy(device_in, in->data, image_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_out, out->data, image_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_h, h, h_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_v, v, v_bytes, cudaMemcpyHostToDevice);

  // create number rows blocks with ceil_for_warps(number pixels per row)
  // threads.
  grid_size = rows;
  block_size = CEIL_TO_WARP_SIZE(in->x);
  // we need shared memory for every input and output array row byte
  bytes_shared_mem = 2 * sizeof(unsigned char) * (bits_per_row / 8);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  permute_gpu<<<grid_size, block_size, bytes_shared_mem>>>(device_in,
      device_out, device_h, device_v, rows, bits_per_row, bytes_shared_mem,
      encrypt);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms, start, stop);

  cudaMemcpy(out->data, device_out, image_bytes, cudaMemcpyDeviceToHost);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(device_in);
  cudaFree(device_out);
  cudaFree(device_h);
  cudaFree(device_v);

  return time_ms;
}

static float
undo_chen_gpu_launch(PPMImage *inout, const sym_key *key,
    const ChenSequence seq[])
{
  size_t seq_length = inout->x * inout->y;
  size_t rows = inout->y;
  // pixels_per_row = columns of image
  size_t pixels_per_row = inout->x;
  size_t grid_size;
  size_t block_size;
  size_t bytes_shared_mem;
  size_t image_bytes = sizeof(PPMPixel) * inout->x * inout->y;
  unsigned char *device_in = 0;
  unsigned char *device_out = 0;
  unsigned char *device_seq = 0;
  unsigned char *seqs[3];
  const unsigned char p0[] =
    { key->rp0, key->gp0, key->bp0 };
  cudaEvent_t start, stop;
  float time_ms = 0.0f;
  float time_ms_gpu;

  // UNDO CHEN SYSTEM TRANSFORMATION
  cudaMalloc((void**) &device_in, image_bytes);
  cudaMalloc((void**) &device_out, image_bytes);
  // Store only one Chen sequence on the GPU at a time
  // there are three unsigned chars per pixel, hence seq_bytes = image_bytes
  cudaMalloc((void**) &device_seq, image_bytes);

  if (!device_in || !device_out || !device_seq) {
    fprintf(stderr, "cudaMalloc failed for device arrays!");
    exit(EXIT_FAILURE);
  }

  cudaMemcpy(device_in, inout->data, image_bytes, cudaMemcpyHostToDevice);
  for (int i = 0; i < 3; i++) {
    seqs[i] = device_seq + i * seq_length;
  }
  cudaMemcpyToSymbol(const_seqs, seqs, sizeof(seqs));
  cudaMemcpyToSymbol(const_p0, p0, sizeof(p0));

  grid_size = rows;
  block_size = CEIL_TO_WARP_SIZE(inout->x);
  // we need shared memory for every byte if a pixel plus the previous pixel
  bytes_shared_mem = sizeof(unsigned char) * 3 * (pixels_per_row + 1);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = key->beta - 1; i >= 0; i--) {
    cudaMemcpy(device_seq, seq[i].sx, image_bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);

    undo_chen_gpu<<<grid_size, block_size, bytes_shared_mem>>>(device_in,
        device_out, pixels_per_row);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms_gpu, start, stop);
    time_ms += time_ms_gpu;

    cudaMemcpy(device_in, device_out, image_bytes, cudaMemcpyDeviceToDevice);
  }

  cudaMemcpy(inout->data, device_out, image_bytes, cudaMemcpyDeviceToHost);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(device_seq);
  cudaFree(device_in);
  cudaFree(device_out);

  return time_ms;
}

#if NUMBER_IMAGES
static float
apply_chen_images_gpu(size_t number_images, size_t width, size_t height,
    const sym_key *key, const ChenSequence seq[])
{
  // create zero images and apply Chen transformation
  // each pixel has three channels
  size_t bytes_images = number_images * width * height * sizeof(unsigned char)
      * 3;
  size_t bytes_seq = sizeof(unsigned char) * 3 * width * height;
  unsigned char *inouts;
  size_t block_size;
  // shared memory must fit a row of a sequence channel
  size_t bytes_shared_mem = sizeof(unsigned char) * width;
  unsigned char *device_images = 0;
  unsigned char *device_seq = 0;
  unsigned char *seqs[3];
  const unsigned char p0[] =
    { key->rp0, key->gp0, key->bp0 };
  float time_ms = 0.0f;
  float time_ms_gpu;
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  inouts = (unsigned char *) malloc(bytes_images);

  if (!inouts) {
    fprintf(stderr, "malloc failed for Chen transform images!");
    exit(EXIT_FAILURE);
  }

  memset(inouts, 0, bytes_images);

  // Store only one Chen sequence on the GPU at a time
  cudaMalloc((void**) &device_images, bytes_images);
  cudaMalloc((void**) &device_seq, bytes_seq);

  if (!device_images || !device_seq) {
    fprintf(stderr, "cudaMalloc failed for Chen transform device arrays!");
    exit(EXIT_FAILURE);
  }

  cudaMemcpy(device_images, inouts, bytes_images, cudaMemcpyHostToDevice);
  for (int i = 0; i < 3; i++) {
    seqs[i] = device_seq + i * width * height;
  }
  cudaMemcpyToSymbol(const_seqs, seqs, sizeof(seqs));
  cudaMemcpyToSymbol(const_p0, p0, sizeof(p0));

  block_size = CEIL_TO_WARP_SIZE(number_images);

  for (int i = 0; i < key->beta; i++) {
    cudaMemcpy(device_seq, seq[i].sx, bytes_seq, cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);

    apply_chen_gpu<<<1, block_size, bytes_shared_mem>>>(number_images,
        device_images, height, width);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms_gpu, start, stop);
    time_ms += time_ms_gpu;
  }

  cudaMemcpy(inouts, device_images, bytes_images, cudaMemcpyDeviceToHost);
  // could do something with inouts here
  free(inouts);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(device_seq);
  cudaFree(device_images);

  return time_ms;
}
#endif

inline static void
encrypt_gpu(const PPMImage *in, PPMImage *partial, PPMImage *out,
    const sym_key *key, const int *h, const int *v, const ChenSequence seq[],
    test_result *res)
{
  // PERMUTATION
  zeroImage(in, partial);
  res->t_enc_perm_gpu = permute_gpu_launch(in, partial, h, v, true);

  // CHEN TRANSFORMATION
  copyImage(partial, out);
  // Owing to dependencies of a pixel to the left-neighbor, Chen encryption
  // is not carried out on GPU (lack of parallelism)
  encrypt_chen_cpu(out, key, seq);
}

inline static void
decrypt_gpu(const PPMImage *in, PPMImage *partial, PPMImage *out, const int *h,
    const int *v, const sym_key *key, const ChenSequence seq[],
    test_result *res)
{
  // CHEN TRANSFORMATION
  copyImage(in, partial);
  res->t_dec_chen_gpu = undo_chen_gpu_launch(partial, key, seq);

  // PERMUTATION
  zeroImage(partial, out);
  res->t_dec_perm_gpu = permute_gpu_launch(partial, out, h, v, false);
}
#endif

static bool
pixels_equal(PPMPixel *a, PPMPixel *b)
{
  return (a->red == b->red) && (a->green == b->green) && (a->blue == b->blue);
}

static void
compare_images(const PPMImage *a, const PPMImage *b, const char *description)
{
  bool passed = true;

  if (!((a->x == b->x) && (a->y == b->y))) {
    passed = false;
  }

  for (int i = 0; (i < (a->x * a->y)) && passed; i++) {
    if (!pixels_equal(&a->data[i], &b->data[i])) {
      passed = false;
    }
  }
  std::cerr << "Test " << description;
  if (passed) {
    std::cerr << " PASSED" << std::endl;
  } else {
    std::cerr << " FAILED" << std::endl;
  }
}

static void
test_run(size_t image_width, test_result *res)
{
  PPMImage *image;
  PPMImage host_encimage;
  PPMImage host_partial_enc;
  PPMImage host_decimage;
  PPMImage host_partial_dec;
#if __ENABLE_GPU__
  PPMImage device_encimage;
  PPMImage device_partial_enc;
  PPMImage device_decimage;
  PPMImage device_partial_dec;
#endif
  sym_key key;
  int *h;
  int *v;
  ChenSequence *seq;
  char image_path[64];
  // Number images must not exceed maximum block thread size for GPU
  // 1024 OK for CC 2.0 and higher.
  const size_t number_images = NUMBER_IMAGES;

  snprintf(image_path, sizeof(image_path), "../img/lenna%ld.ppm", image_width);
  image = readPPM(image_path);
  init_key(&key);
  init_perm_arrays(image, &key, &h, &v);
  init_chen_arrays(image, &key, &seq);
  res->beta = key.beta;
  res->image_width = image->x;
  res->number_images = number_images;

  // Encrypt on CPU
  encrypt_cpu(image, &host_partial_enc, &host_encimage, &key, h, v, seq, res);
  writePPM(IMG_OUT "host-perm-enc.ppm", &host_partial_enc);
  writePPM(IMG_OUT "host-result-enc.ppm", &host_encimage);

  // Decrypt on CPU
  decrypt_cpu(&host_encimage, &host_partial_dec, &host_decimage, &key, h, v,
      seq, res);
  writePPM(IMG_OUT "host-result-dec.ppm", &host_decimage);

  compare_images(image, &host_decimage, "CPU");

#if __ENABLE_GPU__
  // Encrypt on GPU
  encrypt_gpu(image, &device_partial_enc, &device_encimage, &key, h, v, seq,
      res);
  writePPM(IMG_OUT "device-result-enc.ppm", &device_encimage);

  compare_images(&device_partial_enc, &host_partial_enc,
      "enc permute CPU == GPU");
  compare_images(&device_encimage, &host_encimage, "enc CPU == GPU");

  // Decrypt on GPU
  decrypt_gpu(&device_encimage, &device_partial_dec, &device_decimage, h, v,
      &key, seq, res);
  writePPM(IMG_OUT "device-result-dec.ppm", &device_decimage);

  compare_images(&device_partial_dec, &host_partial_dec, "dec Chen CPU == GPU");
  compare_images(&device_decimage, &host_decimage, "dec CPU == GPU");

  // Apply Chen transformation (encryption) for multiple images in parallel
  // on GPU.
#if NUMBER_IMAGES
  res->t_enc_chen_images_gpu = apply_chen_images_gpu(number_images, image_width,
      image_width, &key, seq);
#else
  res->t_enc_chen_images_gpu = 0.0f;
#endif
#endif

  free(h);
  free(v);
  free(image->data);
  free(image);
  free(host_encimage.data);
  free(host_partial_enc.data);
  free(host_decimage.data);
  free(host_partial_dec.data);
  for (int i = 0; i < key.beta; i++) {
    freeChenSequence(&seq[i]);
  }
  free(seq);
#if __ENABLE_GPU__
  free(device_encimage.data);
  free(device_partial_enc.data);
  free(device_decimage.data);
  free(device_partial_dec.data);
#endif
}

static void
print_test_result(test_result *res)
{
  std::cout << std::setprecision(PRECISION) << res->image_width << CSV_DELIMITER
      << res->beta << CSV_DELIMITER << res->t_enc_perm_cpu << CSV_DELIMITER
      << res->t_enc_chen_cpu << CSV_DELIMITER << res->t_dec_chen_cpu
      << CSV_DELIMITER << res->t_dec_perm_cpu;

#if __ENABLE_GPU__
  std::cout << std::setprecision(PRECISION) << CSV_DELIMITER
      << res->t_enc_perm_gpu << CSV_DELIMITER << res->t_dec_chen_gpu
      << CSV_DELIMITER << res->t_dec_perm_gpu << CSV_DELIMITER
      << res->number_images << CSV_DELIMITER << res->t_enc_chen_images_gpu
      << std::endl;
#else
  std::cout << std::endl;
#endif
}

static void
print_csv_header(void)
{
  // print CSV header
  std::cout << "img_width" << CSV_DELIMITER << "beta" << CSV_DELIMITER
      << "cpu_enc_perm" << CSV_DELIMITER << "cpu_enc_chen" << CSV_DELIMITER
      << "cpu_dec_chen" << CSV_DELIMITER << "cpu_dec_perm";

#if __ENABLE_GPU__
  std::cout << CSV_DELIMITER << "gpu_enc_perm" << CSV_DELIMITER
      << "gpu_dec_chen" << CSV_DELIMITER << "gpu_dec_perm" << CSV_DELIMITER
      << "no_imgs_gpu_enc_chen" << CSV_DELIMITER << "gpu_enc_chen" << std::endl;
#else
  std::cout << std::endl;
#endif
}

int
main(int argc, char *argv[])
{
  print_csv_header();

  for (size_t image_width = 32; image_width <= MAX_IMAGE_WIDTH;) {
    for (size_t i = 0; i < TEST_RUNS; i++) {
      test_result res;
      test_run(image_width, &res);
      print_test_result(&res);
    }
    if (image_width < 256) {
      image_width *= 2;
    } else {
      image_width += 128;
    }
  }

  return EXIT_SUCCESS;
}
