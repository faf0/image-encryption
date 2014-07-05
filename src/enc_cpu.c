/*
 * enc_cpu.c
 *
 *      Author: Fabian Foerg
 */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

#include <sys/time.h>
#include <time.h>

#include "libppm.h"
#include "enc_cpu.h"

/**
 * Code to measure CPU/GPU time based on http://stackoverflow.com/a/16267334
 */
struct timespec init;
struct timespec after;

void
tic(void)
{
  clock_gettime(CLOCK_MONOTONIC, &init);
  //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &init);
}

double
toc_us(void)
{
  clock_gettime(CLOCK_MONOTONIC, &after);
  double us = (after.tv_sec - init.tv_sec) * 1000000.0f;
  return us + (after.tv_nsec - init.tv_nsec) / 1000.0f;
}

double
toc(void)
{
  double time_us = toc_us();
  return us_to_ms(time_us);
}

double
us_to_ms(double us)
{
  return us / 1000.0f;
}

void
init_key(sym_key *key)
{
  key->x0 = 0.978901234f;
  key->y0 = 0.145678901f;
  key->xc0 = -10.218293f;
  key->yc0 = 0.981829f;
  key->zc0 = 37.873617f;
  key->rp0 = 123;
  key->gp0 = 12;
  key->bp0 = 23;
  key->px = 0.434567890f;
  key->py = 0.245678901f;
  key->c = 28.0f;
  key->n0 = 189;
  key->beta = 2;
}

void
freeChenSequence(ChenSequence *seq)
{
  if (seq) {
    free(seq->sx);
  }
}

float
scale_float(const float x)
{
  float frac;
  int expt;

  frac = frexpf(x, &expt);
  return ldexpf(frac, expt & 0x0f);
}

static void
chen_iterate(const ChenPixel *in, ChenPixel *out, const float c)
{
  const float a = 35.0f;
  const float b = 3.0f;

  // FIXME x, y, z are not scaled in original algorithm. Here to
  // prevent NaNs.
  float x = scale_float(in->x);
  float y = scale_float(in->y);
  float z = scale_float(in->z);
  out->x = a * (y - x);
  out->y = (c - a) * x - x * z + c * y;
  out->z = x * y - b * z;
}

static unsigned char
chen_to_byte_cpu(float x)
{
  float absx = fabsf(x);
  return (unsigned char) ((int) ((absx - floorf(absx)) * CHEN_SCALING) % 256);
}

void
chen_init(const sym_key *key, ChenPixel *out)
{
  out->x = key->xc0;
  out->y = key->yc0;
  out->z = key->zc0;

  for (int i = 0; i < key->n0; i++) {
    chen_iterate(out, out, key->c);
  }
}

void
chen_byte_seq_next(size_t length, const sym_key *key, ChenPixel *pixelinout,
    ChenSequence *seqout)
{
  seqout->sx = (unsigned char *) malloc(3 * sizeof(unsigned char) * length);

  if (!seqout->sx) {
    fprintf(stderr, "malloc failed for next Chen sequence!");
    exit(EXIT_FAILURE);
  }

  seqout->sy = seqout->sx + length;
  seqout->sz = seqout->sy + length;
  seqout->s_length = length;

  for (size_t i = 0; i < length; i++) {
    chen_iterate(pixelinout, pixelinout, key->c);
    seqout->sx[i] = chen_to_byte_cpu(pixelinout->x);
    seqout->sy[i] = chen_to_byte_cpu(pixelinout->y);
    seqout->sz[i] = chen_to_byte_cpu(pixelinout->z);
  }
}

static void
chen_byte_seq_first(size_t length, const sym_key *key, ChenPixel *pixelout,
    ChenSequence *seqout)
{
  chen_init(key, pixelout);
  chen_byte_seq_next(length, key, pixelout, seqout);
}

static inline unsigned char
apply_chen_byte(const unsigned char c, const unsigned char cp,
    const unsigned char s)
{
  return ((unsigned char) ((int) c + (int) cp) % 256) ^ s;
}

static inline unsigned char
undo_chen_byte(const unsigned char c, const unsigned char cp,
    const unsigned char s)
{
  return (unsigned char) (((int) (c ^ s) - (int) cp) % 256);
}

static float
apply_chen_image(PPMImage *inout, const ChenSequence *seq, const sym_key *key,
    const bool encrypt)
{
  unsigned char pre_pixel_byte;
  unsigned char *s;
  unsigned char *seqs[] =
    { seq->sx, seq->sy, seq->sz };
  const unsigned char p0[] =
    { key->rp0, key->gp0, key->bp0 };

  tic();
  for (int ch = 0; ch < 3; ch++) {
    pre_pixel_byte = p0[ch];
    s = seqs[ch];

    for (int row = 0; row < inout->y; row++) {
      for (int col = 0; col < inout->x; col++) {
        unsigned char current_byte = getPixelByte(inout, row, col, ch);
        unsigned char new_byte;

        if (encrypt) {
          new_byte = apply_chen_byte(current_byte, pre_pixel_byte,
              s[row * inout->x + col]);
          setPixelByte(inout, row, col, ch, new_byte);
          pre_pixel_byte = new_byte;
        } else {
          new_byte = undo_chen_byte(current_byte, pre_pixel_byte,
              s[row * inout->x + col]);
          setPixelByte(inout, row, col, ch, new_byte);
          pre_pixel_byte = current_byte;
        }
      }
    }
  }

  return us_to_ms(toc_us());
}

static float
pwlcm(float x, float p)
{
  if ((0.0f <= x) && (x < p)) {
    return x / p;
  } else if ((p <= x) && (x < 0.5f)) {
    return (x - p) / (0.5f - p);
  } else {
    return pwlcm(1.0f - x, p);
  }
}

//static int
//comp(const void * a, const void * b)
//{
//  int va = *(int *) a;
//  int vb = *(int *) b;
//  if (va == vb)
//    return 0;
//  else if (va < vb)
//    return -1;
//  else
//    return 1;
//}

int *
perm_array(int len, float x, float p)
{
  int *array = (int *) malloc(sizeof(int) * len);
  float xi = x;

  if (!array) {
    fprintf(stderr, "malloc failed for permutation array!");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < len; i++) {
    array[i] = i;
  }

  for (int i = len - 1; i >= 0; i--) {
    int tmp, index;

    xi = pwlcm(xi, p);
    index = (int) floorf(xi * len) % (i + 1);
    tmp = array[i];
    array[i] = array[index];
    array[index] = tmp;
  }

  //  qsort(array, len, sizeof(array[0]), comp);
  //  for (int i = 0; i < len; i++) {
  //    if (i != array[i]) {
  //      printf("perm_array error at (%d, %d) ", i, array[i]);
  //    }
  //  }

  return array;
}

/**
 * Returns a byte which has bit bpos from b set at position outpos.
 * The other bits are zero.
 */
inline static unsigned char
set_bit(unsigned char b, int bpos, int outpos)
{
  return ((b >> bpos) & 1) << outpos;
}

/**
 * Updates an output pixel bit (for a specific color channel).
 * The bit comes from an input pixel bit (for a specific color channel).
 *
 * @param in input image
 * @param out output image
 * @param bits_per_row bits_per_row = number pixels * 8 (bits) * 3 (color channels)
 * @param in_row row number of input image
 * @param in_col bit column number of input image
 * @param out_row row number of output image
 * @param out_col bit column number of output image
 */
static void
set_bit_in_pixel(const PPMImage *in, PPMImage *out, size_t bits_per_row,
    int in_row, int in_col, int out_row, int out_col)
{
  size_t pixels_per_row = bits_per_row / (3 * 8);
  int in_pixel = in_col / (3 * 8);
  in_pixel += in_row * pixels_per_row;
  int in_channel = (in_col / 8) % 3;
  int in_bit = in_col % 8;
  int out_pixel = out_col / (3 * 8);
  out_pixel += out_row * pixels_per_row;
  int out_channel = (out_col / 8) % 3;
  int out_bit = out_col % 8;
  unsigned char *in_byte_ptr;
  unsigned char *out_byte_ptr;
  unsigned char updated_out_byte;

  in_byte_ptr = ((unsigned char *) &in->data[in_pixel]) + in_channel;
  out_byte_ptr = ((unsigned char *) &out->data[out_pixel]) + out_channel;

  updated_out_byte = set_bit(*in_byte_ptr, in_bit, out_bit);
  *out_byte_ptr |= updated_out_byte;
}

float
permute(const PPMImage *in, PPMImage *out, const sym_key *key, const int *h,
    const int *v)
{
  size_t bits_per_row = 3 * 8 * in->x;
  tic();

  for (int r = 0; r < in->y; r++) {
    for (size_t c = 0; c < bits_per_row; c++) {
      set_bit_in_pixel(in, out, bits_per_row, r, c, h[r], v[c]);
    }
  }

  return us_to_ms(toc_us());
}

float
reverse_permute(const PPMImage *in, PPMImage *out, const sym_key *key,
    const int *h, const int *v)
{
  size_t bits_per_row = 3 * 8 * in->x;
  tic();

  for (int r = 0; r < in->y; r++) {
    for (size_t c = 0; c < bits_per_row; c++) {
      set_bit_in_pixel(in, out, bits_per_row, h[r], v[c], r, c);
    }
  }

  return us_to_ms(toc_us());
}

void
init_perm_arrays(const PPMImage *model, const sym_key *key, int **h, int **v)
{
  size_t bits_per_row = 3 * 8 * model->x;

  *h = perm_array(model->y, key->y0, key->py);
  *v = perm_array(bits_per_row, key->x0, key->px);
}

void
init_chen_arrays(const PPMImage *model, const sym_key *key,
    ChenSequence *seq_out[])
{
  ChenPixel pixel;
  ChenSequence *seq;
  size_t seq_length = model->x * model->y;

  *seq_out = (ChenSequence*) malloc(sizeof(ChenSequence) * key->beta);
  seq = *seq_out;

  if (!seq) {
    fprintf(stderr, "malloc failed for ChenSequence");
  }

  chen_byte_seq_first(seq_length, key, &pixel, &seq[0]);

  for (int i = 1; i < key->beta; i++) {
    chen_byte_seq_next(seq_length, key, &pixel, &seq[i]);
  }
}

float
encrypt_chen_cpu(PPMImage *inout, const sym_key *key, const ChenSequence seq[])
{
  float time_ms = 0.0f;

  for (int i = 0; i < key->beta; i++) {
    time_ms += apply_chen_image(inout, &seq[i], key, true);
  }

  return time_ms;
}

static float
decrypt_chen_cpu(PPMImage *inout, const sym_key *key, const ChenSequence seq[])
{
  float time_ms = 0.0f;

  for (int i = key->beta - 1; i >= 0; i--) {
    time_ms += apply_chen_image(inout, &seq[i], key, false);
  }

  return time_ms;
}

void
encrypt_cpu(const PPMImage *in, PPMImage *partial, PPMImage *out,
    const sym_key *key, const int *h, const int *v, const ChenSequence seq[],
    test_result *res)
{
  // PERMUTATION
  zeroImage(in, partial);
  res->t_enc_perm_cpu = permute(in, partial, key, h, v);

  // CHEN SYSTEM TRANSFORMATION
  copyImage(partial, out);
  res->t_enc_chen_cpu = encrypt_chen_cpu(out, key, seq);
}

void
decrypt_cpu(const PPMImage *in, PPMImage *partial, PPMImage *out,
    const sym_key *key, const int *h, const int *v, const ChenSequence seq[],
    test_result *res)
{
  // UNDO CHEN SYSTEM TRANSFORMATION
  copyImage(in, partial);
  res->t_dec_chen_cpu = decrypt_chen_cpu(partial, key, seq);

  // UNDO PERMUTATION
  zeroImage(partial, out);
  res->t_dec_perm_cpu = reverse_permute(partial, out, key, h, v);
}
