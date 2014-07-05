/*
 * libppm.h
 *
 * From http://stackoverflow.com/a/2699908
 */

#ifndef _LIBPPM_H_
#define _LIBPPM_H_

typedef struct
{
  unsigned char red, green, blue;
} PPMPixel;

typedef struct
{
  int x, y;
  PPMPixel *data;
} PPMImage;

#define CREATOR "RPFELGUEIRAS"
#define RGB_COMPONENT_COLOR 255

PPMImage *
readPPM(const char *filename);
void
writePPM(const char *filename, PPMImage *img);
unsigned char
getPixelByte(const PPMImage *in, size_t row, size_t col, size_t channel);
void
setPixelByte(PPMImage *inout, size_t row, size_t col, size_t channel,
    unsigned char pixel_byte);
void
zeroImage(const PPMImage *model, PPMImage *zero);
void
copyImage(const PPMImage *in, PPMImage *copy);
void
changeColorPPM(PPMImage *img);

#endif // #ifndef _LIBPPM_H_

