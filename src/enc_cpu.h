/*
 * enc_cpu.h
 *
 *  Created on: Apr 29, 2014
 *      Author: Fabian Foerg
 */

#ifndef ENC_CPU_H_
#define ENC_CPU_H_

#include <sys/time.h>
#include <time.h>

#include "libppm.h"

// FIXME set to prevent NaNs. Different in original algorithm.
#define CHEN_SCALING (1.0e7f)

void
tic(void);

double
toc_us(void);

double
toc(void);

typedef struct
{
  float x0;
  float y0;
  float xc0;
  float yc0;
  float zc0;
  unsigned char rp0;
  unsigned char gp0;
  unsigned char bp0;
  float px;
  float py;
  float c;
  int n0;
  int beta;
} sym_key;

typedef struct
{
  size_t image_width;
  size_t beta;
  /* all times in ms */
  float t_enc_perm_cpu;
  float t_enc_chen_cpu;
  float t_dec_perm_cpu;
  float t_dec_chen_cpu;
  float t_enc_perm_gpu;
  float t_dec_perm_gpu;
  float t_dec_chen_gpu;
  size_t number_images;
  float t_enc_chen_images_gpu;
} test_result;

typedef struct
{
  float x;
  float y;
  float z;
} ChenPixel;

typedef struct
{
  unsigned char *sx;
  unsigned char *sy;
  unsigned char *sz;
  size_t s_length;
} ChenSequence;

void
freeChenSequence(ChenSequence *seq);

double
us_to_ms(double us);

void
init_key(sym_key *key);

float
scale_float(const float x);

void
chen_init(const sym_key *key, ChenPixel *out);

void
chen_byte_seq_next(size_t length, const sym_key *key, ChenPixel *pixelinout,
    ChenSequence *seqout);

int *
perm_array(int len, float x, float p);

float
permute(const PPMImage *in, PPMImage *out, const sym_key *key, const int *h,
    const int *v);

float
reverse_permute(const PPMImage *in, PPMImage *out, const sym_key *key,
    const int *h, const int *v);

void
init_perm_arrays(const PPMImage *model, const sym_key *key, int **h, int **v);

void
init_chen_arrays(const PPMImage *model, const sym_key *key,
    ChenSequence *seq_out[]);

float
encrypt_chen_cpu(PPMImage *inout, const sym_key *key, const ChenSequence seq[]);

void
encrypt_cpu(const PPMImage *in, PPMImage *partial, PPMImage *out,
    const sym_key *key, const int *h, const int *v, const ChenSequence seq[],
    test_result *res);

void
decrypt_cpu(const PPMImage *in, PPMImage *partial, PPMImage *out,
    const sym_key *key, const int *h, const int *v, const ChenSequence seq[],
    test_result *res);

#endif /* !ENC_CPU_H_ */
