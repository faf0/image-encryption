# About

Implementation of the color image encryption (and decryption) algorithm from the paper

    Hongjun Liu and Xingyuan Wang. Color Image Encryption using Spatial Bit-Level Permutation and High-Dimension Chaotic System. Optics Communications, 284(16-17):3895–3903, August 2011

for the GPU in CUDA and for the CPU in C and Julia.

This was my course project for [CS677](http://www.cs.stevens.edu/~mordohai/classes/cs677_s14.html) (Parallel Programming for Many-Core Processors) at Stevens Institute of Technology in the Spring of 2014.

# Caveats

The implementation deviates from the original algorithm proposed by Liu and Wang.
The Chen sequence generation algorithm was adapted in the implementation to prevent
overflowing floating point numbers.
The C code tackles this problem by trimming the exponent during each iteration, while
the Julia implementation reduces the numbers modulo a large number.

As the implementation depends on the floating point standard, as well as the
precision of the numbers, it is not portable.

The Julia code which runs on the CPU is complete, whereas the code which calls
GPU kernels is incomplete and may not work, as during the time of
writing, issues with the Julia CUDA library were encountered.

# Copyright

(Copyright) 2014 Fabian Foerg

