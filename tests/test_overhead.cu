#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <iomanip>

#include "libppm.h"
#include "enc_cpu.h"

#define MAX_IMAGE_WIDTH 1024
#define NUMBER_IMAGES 448
#define TEST_RUNS 20
#define PRECISION 10
#define CSV_DELIMITER ";"

typedef struct
{
  size_t image_width;
  size_t number_images;
  /* all times in ms */
  /*
   float malloc_h;
   float malloc_v;
   */
  float malloc_image;
  float malloc_imagemany;
  float h_init; /* malloc + value init */
  float v_init; /* malloc + value init */
  float chen_init;
  float chen_seq;
  float cudaMalloc_h;
  float cudaMalloc_v;
  float cudaMalloc_image;
  float cudaMalloc_imagemany;
  float cudaMemcpy_h2device;
  float cudaMemcpy_v2device;
  float cudaMemcpy_image2device;
  float cudaMemcpy_imagefromdevice;
  float cudaMemcpy_imagemany2device;
  float cudaMemcpy_imagemanyfromdevice;
  float cudaMemcpy_imagedevice2device;
  float cudaFree_h;
  float cudaFree_v;
  float cudaFree_image;
  float cudaFree_imagemany;
  float free_h;
  float free_v;
  float free_image;
  float free_imagemany;
} test_overhead_result;

static inline void
cudaTic(cudaEvent_t *start)
{
  cudaEventRecord(*start, 0);
}

static inline float
cudaToc(cudaEvent_t *start, cudaEvent_t *stop)
{
  float time_ms;

  cudaEventRecord(*stop, 0);
  cudaEventSynchronize(*stop);
  cudaEventElapsedTime(&time_ms, *start, *stop);

  return time_ms;
}

static void
test_run(size_t image_width, size_t number_images, const sym_key *key,
    test_overhead_result *res)
{
  size_t image_bytes = sizeof(PPMPixel) * image_width * image_width;
  size_t imagemany_bytes = image_bytes * number_images;
  size_t h_bytes = sizeof(int) * image_width;
  size_t v_bytes = sizeof(int) * 3 * 8 * image_width;
  size_t h_len = image_width;
  size_t v_len = 3 * 8 * image_width;
  size_t seq_len = image_width * image_width;
  int *h;
  int *v;
  ChenPixel pix;
  ChenSequence seq;
  unsigned char *image;
  unsigned char *imagemany;
  unsigned char *dev_h;
  unsigned char *dev_v;
  unsigned char *dev_image;
  unsigned char *dev_image2;
  unsigned char *dev_imagemany;
  cudaEvent_t start, stop;

  res->image_width = image_width;
  res->number_images = number_images;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // host malloc
  /*
   tic();
   h = (int *) malloc(h_bytes);
   res->malloc_h = toc();
   tic();
   v = (int *) malloc(v_bytes);
   res->malloc_v = toc();
   */
  tic();
  image = (unsigned char *) malloc(image_bytes);
  res->malloc_image = toc();
  tic();
  imagemany = (unsigned char *) malloc(imagemany_bytes);
  res->malloc_imagemany = toc();

  if (!image || !imagemany) {
    fprintf(stderr, "malloc failed on host!");
  }

  // host init values
  tic();
  h = perm_array(h_len, key->x0, key->px);
  res->h_init = toc();

  tic();
  v = perm_array(v_len, key->x0, key->px);
  res->v_init = toc();

  tic();
  chen_init(key, &pix);
  res->chen_init = toc();

  tic();
  chen_byte_seq_next(seq_len, key, &pix, &seq);
  res->chen_seq = toc();

  memset(image, 0, image_bytes);
  memset(imagemany, 0, imagemany_bytes);

  // cudaMalloc
  cudaTic(&start);
  cudaMalloc((void**) &dev_h, h_bytes);
  res->cudaMalloc_h = cudaToc(&start, &stop);

  cudaTic(&start);
  cudaMalloc((void**) &dev_v, v_bytes);
  res->cudaMalloc_v = cudaToc(&start, &stop);

  cudaTic(&start);
  cudaMalloc((void**) &dev_image, image_bytes);
  res->cudaMalloc_image = cudaToc(&start, &stop);

  cudaTic(&start);
  cudaMalloc((void**) &dev_imagemany, imagemany_bytes);
  res->cudaMalloc_imagemany = cudaToc(&start, &stop);

  // cudaMemcpy
  cudaTic(&start);
  cudaMemcpy(dev_h, h, h_bytes, cudaMemcpyHostToDevice);
  res->cudaMemcpy_h2device = cudaToc(&start, &stop);

  cudaTic(&start);
  cudaMemcpy(dev_v, v, v_bytes, cudaMemcpyHostToDevice);
  res->cudaMemcpy_v2device = cudaToc(&start, &stop);

  cudaTic(&start);
  cudaMemcpy(dev_image, image, image_bytes, cudaMemcpyHostToDevice);
  res->cudaMemcpy_image2device = cudaToc(&start, &stop);

  cudaTic(&start);
  cudaMemcpy(image, dev_image, image_bytes, cudaMemcpyDeviceToHost);
  res->cudaMemcpy_imagefromdevice = cudaToc(&start, &stop);

  cudaTic(&start);
  cudaMemcpy(dev_imagemany, imagemany, imagemany_bytes,
      cudaMemcpyHostToDevice);
  res->cudaMemcpy_imagemany2device = cudaToc(&start, &stop);

  cudaTic(&start);
  cudaMemcpy(imagemany, dev_imagemany, imagemany_bytes,
      cudaMemcpyDeviceToHost);
  res->cudaMemcpy_imagemanyfromdevice = cudaToc(&start, &stop);

  cudaTic(&start);
  cudaMemcpy(dev_image, image, image_bytes, cudaMemcpyHostToDevice);
  res->cudaMemcpy_image2device = cudaToc(&start, &stop);

  cudaMalloc((void**) &dev_image2, image_bytes);
  cudaTic(&start);
  cudaMemcpy(dev_image2, dev_image, image_bytes, cudaMemcpyDeviceToDevice);
  res->cudaMemcpy_imagedevice2device = cudaToc(&start, &stop);
  cudaFree(dev_image2);

  // cudaFree
  cudaTic(&start);
  cudaFree(dev_h);
  res->cudaFree_h = cudaToc(&start, &stop);

  cudaTic(&start);
  cudaFree(dev_v);
  res->cudaFree_v = cudaToc(&start, &stop);

  cudaTic(&start);
  cudaFree(dev_image);
  res->cudaFree_image = cudaToc(&start, &stop);

  cudaTic(&start);
  cudaFree(dev_imagemany);
  res->cudaFree_imagemany = cudaToc(&start, &stop);

  // host free
  tic();
  free(h);
  res->free_h = toc();

  tic();
  free(v);
  res->free_v = toc();

  // like free(image);
  freeChenSequence(&seq);

  tic();
  free(image);
  res->free_image = toc();

  tic();
  free(imagemany);
  res->free_imagemany = toc();

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

static void
print_test_result(test_overhead_result *res)
{
  const float res_floats[] =
    { res->malloc_image, res->malloc_imagemany,
        res->h_init, res->v_init,
        res->chen_init, res->chen_seq,
        res->cudaMalloc_h, res->cudaMalloc_v,
        res->cudaMalloc_image, res->cudaMalloc_imagemany,
        res->cudaMemcpy_h2device, res->cudaMemcpy_v2device,
        res->cudaMemcpy_image2device, res->cudaMemcpy_imagefromdevice,
        res->cudaMemcpy_imagemany2device, res->cudaMemcpy_imagemanyfromdevice,
        res->cudaMemcpy_imagedevice2device,
        res->cudaFree_h, res->cudaFree_v,
        res->cudaFree_image, res->cudaFree_imagemany,
        res->free_h, res->free_v,
        res->free_image, res->free_imagemany };
  size_t res_floats_len = sizeof(res_floats) / sizeof(res_floats[0]);

  std::cout << std::setprecision(PRECISION) << res->image_width
      << CSV_DELIMITER << res->number_images << CSV_DELIMITER;

  for (size_t i = 0; i < res_floats_len; i++) {
    std::cout << res_floats[i];
    if (i < (res_floats_len - 1)) {
      std::cout << CSV_DELIMITER;
    }
  }

  std::cout << std::endl;
}

static void
print_csv_header(void)
{
  const char *headers[] =
    { "img_width", "number_images", "malloc_image", "malloc_imagemany",
        "h_init", "v_init", "chen_init", "chen_seq", "cudaMalloc_h",
        "cudaMalloc_v", "cudaMalloc_image", "cudaMalloc_imagemany",
        "cudaMemcpy_h2device", "cudaMemcpy_v2device", "cudaMemcpy_image2device",
        "cudaMemcpy_imagefromdevice", "cudaMemcpy_imagemany2device",
        "cudaMemcpy_imagemanyfromdevice", "cudaMemcpy_imagedevice2device",
        "cudaFree_h", "cudaFree_v", "cudaFree_image", "cudaFree_imagemany",
        "free_h", "free_v", "free_image", "free_imagemany" };
  size_t headers_len = sizeof(headers) / sizeof(headers[0]);

  for (size_t i = 0; i < headers_len; i++) {
    std::cout << headers[i];
    if (i < (headers_len - 1)) {
      std::cout << CSV_DELIMITER;
    }
  }
  std::cout << std::endl;
}

int
main(int argc, char *argv[])
{
  int *device_ptr;
  sym_key key;
  size_t number_images = NUMBER_IMAGES;

  /* Make sure that device context gets initialized before tests.
   * First cudaMalloc may be slow.
   */
  cudaMalloc((void**) &device_ptr, sizeof(int) * (2 << 12));
  cudaFree(device_ptr);

  init_key(&key);
  print_csv_header();

  for (size_t image_width = 32; image_width <= MAX_IMAGE_WIDTH;) {
    for (size_t i = 0; i < TEST_RUNS; i++) {
      test_overhead_result res;
      test_run(image_width, number_images, &key, &res);
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
