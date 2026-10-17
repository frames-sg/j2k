# Reversible early-stop fixtures

Reversible (5/3) JPEG 2000 fixtures whose code blocks' coding passes stop
before their lowest bit-planes.

T.800 E.1.2.2 (Equation E-8) reconstructs such coefficients at
q ± r·2^(Mb−Nb), with the reconstruction parameter r chosen by the decoder
(commonly 1/2). The fixtures below need r = 1/2: a decoder that uses r = 0
gets them wrong. The native, CUDA and Metal classic decoders share that rule.

## `emri_small_jpeg_2k_lossless_frame1.j2c`

A lossless codestream whose encoder stopped some code blocks early. It is
frame 1 (zero-based) of `emri_small_jpeg_2k_lossless.dcm` from
[pydicom-data](https://github.com/pydicom/pydicom-data) (commit
`abc42b90985fb6cf385aa4af766d2c9c94a257a4`; file SHA-256
`b2b4063359a08ed3b0afa9f4e4f72f84af79e5116515b446d9a30da9dc7f1888`). It is
that frame's encapsulated fragment, byte for byte, including the trailing
`00` padding after EOC. It is a 64 × 64 single-component codestream with SIZ
precision 16 (unsigned), 5 decomposition levels and 64 × 64 code blocks. Its
COM marker reads "Created by GDCM/OpenJPEG version 2.0".

The LL and coarsest LH code blocks stop before their last coding passes: for
example, the LL block codes 39 of its 43 possible passes. The encoder dropped
passes whose bits equal the r = 1/2 midpoint, so with r = 0 this frame decodes
off by up to 2 in 3,336 of its 4,096 samples.

`emri_small_frame1.u16le` holds the expected samples: frame 1 of
`emri_small.dcm` from the same pydicom-data commit (file SHA-256
`151233ec63f64ebb63b979df51aa827cd612a53422c073f6ef341770c7bc9a56`), the
uncompressed original of the same series, as row-major little-endian `u16`.
No JPEG 2000 decoder was involved in producing it; OpenJPEG 2.5.4 decodes the
codestream to the same bytes.

pydicom-data is MIT licensed; its license is retained in
`LICENSE.pydicom-data`.

## `openjpeg_53_rate3_gray12_64x64.j2k`

A rate-limited reversible codestream in which every code block stops early,
with four code blocks in each of the three finest sub-bands. OpenJPEG 2.5.4
encoded it from a synthetic 64 × 64 unsigned 12-bit image whose sample at
column `x` and row `y` is

    (1024 + 40·x + 23·y + (7·x·y mod 331)) mod 4096

using `opj_compress -i gray12.pgm -o openjpeg_53_rate3_gray12_64x64.j2k -r 3
-b 16,16 -n 3` (5/3, 2 decomposition levels, 16 × 16 code blocks, 3:1). Its
SHA-256 is `11ffbd4791283e57e052e8a2ff4683c44af89256a0fb35a091030fc443be459c`.

`openjpeg_53_rate3_gray12_64x64.u16le` holds the expected samples, as
row-major little-endian `u16`: the output of `opj_decompress` 2.5.4 for that
codestream (SHA-256
`501d47b082b782cf25bb8a3499112b63c09a5784da3eedfc3cd3f4ee8ff2859b`). With
r = 0 it decodes off by up to 70 in 3,977 of its 4,096 samples.

## `openjpeg_53_rate12_rgb8_128x128.j2k`

The same kind of codestream in the shape the Metal cooperative batch decoder
splits between CPU and GPU Tier-1: 128 × 128 8-bit RGB with the reversible
color transform. Every coded block stops early. OpenJPEG 2.5.4 encoded it
from a synthetic image whose red, green and blue samples at column `x` and row
`y` are

    (16 + x + y + (3·x·y mod 61)) mod 256
    (200 − x + (5·x·y mod 47)) mod 256
    (64 + 2·y − ⌊x/2⌋ + ((x² + y) mod 37)) mod 256

using `opj_compress -i rgb8.ppm -o openjpeg_53_rate12_rgb8_128x128.j2k -r 12
-b 16,16 -n 3`. Its SHA-256 is
`5d3316115e524acc571aec03f9e2c0ef090427e4131866f828e99ee01e91d260`.
`openjpeg_53_rate12_rgb8_128x128.rgb` contains the 49,152 interleaved RGB
samples from `opj_decompress` 2.5.4 (SHA-256
`d03c235ce98e68eeefdd5787c56eaa425605553a73efa94155a8deffbf76a24c`).
Tests compare those bytes directly. With r = 0, 43,474 samples differ.

## `openjpeg_53_rate3_roi2_gray8_19x13.j2k`

A rate-limited reversible codestream that combines early-stopped coding passes
with ROI maxshift. Its 19 x 13 unsigned 8-bit source is the repository's
`gray_u8_53.source.pgm` OpenJPH batch fixture. OpenJPEG 2.5.4 encoded it with

    opj_compress -i gray_u8_53.source.pgm \
      -o openjpeg_53_rate3_roi2_gray8_19x13.j2k \
      -r 3 -n 3 -ROI c=0,U=2

The codestream SHA-256 is
`2551f0ba982cbc6673fe3b6a3425d7b344361cacd78b1002011ff916253a46c5`.
`openjpeg_53_rate3_roi2_gray8_19x13.gray` contains the 247 samples from
`opj_decompress` 2.5.4 (SHA-256
`617924507e0f86d4b29ee9545ce136500426d59f9b9219a900d09292c18d7077`).
Applying ROI inverse before the reversible midpoint gets 203 samples wrong;
the largest absolute error is 36.
