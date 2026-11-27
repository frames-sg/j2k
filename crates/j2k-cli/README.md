# J2K CLI

> The `j2k` command-line tool for [J2K](https://github.com/frames-sg/j2k)

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-cli.svg)](https://crates.io/crates/j2k-cli)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

The `j2k` command inspects JPEG 2000, HTJ2K, and JPEG files, converts JPEG to
HTJ2K, and decodes JPEG 2000 with the options of OpenJPEG's `opj_decompress`.
It does not encode images; use the [`j2k`](https://crates.io/crates/j2k)
library for that.

```bash
cargo install j2k-cli
```

## Commands

```bash
j2k inspect <file>
j2k transcode <input.jpg> <output.j2k> --htj2k --lossless-53
j2k decode -i <input> -o <output> [opj_decompress options]
```

`inspect` prints one line describing the file: for JPEG 2000 and HTJ2K its
size, color space, bit depth, component count, resolution levels, and tiles;
for JPEG its size, frame type, color space, bit depth, sampling, MCU layout,
restart interval, and scan count. It reads at most the first 64 MiB.

`transcode` converts a JPEG to a reversible 5/3 HTJ2K codestream from the
JPEG's DCT coefficients, without decoding it to pixels. It needs both
`--htj2k` and `--lossless-53`. It prints the image size, component count,
output size, and stage timings.

`decode` writes one image file, as described below.

Exit status is 0 on success, 2 for a bad command line, and 1 for any other
failure. `j2k`, `j2k help`, `j2k -h`, and `j2k decode -h` print usage to
stderr and return 0.

## The `opj_decompress` interface

`j2k decode` accepts OpenJPEG 2.5 `opj_decompress` options with the same
meanings. Run under the name `opj_decompress` (through a symlink or a copy),
the same binary behaves as that tool, so programs that run `opj_decompress`,
such as Cantaloupe's `OpenJpegProcessor`, can run j2k unchanged:

```bash
ln -s "$(command -v j2k)" /usr/local/j2k/opj_decompress
opj_decompress -quiet -i image.jp2 -d 512,512,1536,1536 -r 1 -o tile.bmp
```

| Option | Meaning |
| --- | --- |
| `-i <file>` | JP2, JPH, or raw J2K/HTJ2K codestream, recognized by its signature, so names without an extension work |
| `-o <file>` | Output file. The extension picks the format: `.bmp`, `.png`, `.tif`/`.tiff`, `.pgm`, `.ppm`, or `.pnm` |
| `-r <n>` | Discard the `n` highest resolution levels, at most the smallest decomposition count of any tile and component |
| `-d <x0,y0,x1,y1>` | Decode area on the full-resolution reference grid, so the image's first sample is at its origin (`XOsiz`, `YOsiz`). At reduction `r` it keeps `[ceil(x0 / 2^r), ceil(x1 / 2^r))` of the reduced grid (ITU-T T.800 B.5). A start before the image area or an end past it is clamped with a warning; an area wholly outside the image, or empty, is an error. `0,0,0,0` is the whole image |
| `-threads <n\|ALL_CPUS>` | Decode threads. Without it, `OPJ_NUM_THREADS` (`ALL_CPUS`, or a count capped at twice the number of processors), else one |
| `-force-rgb` | Write gray images as RGB |
| `-l 0` | Decode every quality layer (the default) |
| `-quiet` | Print no notes or warnings |
| `-h` | Print the option summary |

These are rejected with an error that names the option: `-l` above 0, `-t`,
`-c`, `-p`, `-x`, `-upsample`, `-split-pnm`, `-allow-partial`, `-ImgDir`,
`-OutFor`, `.jpt` input, and `.pgx`, `.raw`, `.rawl`, `.yuv`, and `.tga`
output. j2k can't honor them exactly, and ignoring one would change what the
caller receives.

A decode prints nothing to stdout. Notes, warnings, and errors go to stderr,
so an `-o` path linked to `/dev/stdout` carries only the image, as Cantaloupe
expects. An input that is not JPEG 2000 fails with
`Unknown input file format`, the message Cantaloupe maps to a source-format
error. The help text names no `vX.Y.Z` version, because Cantaloupe reads one
from `opj_decompress -h` and stops passing `-quiet` below 2.2.

Run as `opj_decompress`, the binary uses that tool's exit status: every
failure returns 1, and `-h` prints to stdout and returns 1.

### Samples and color

Images whose components are Gray, RGB, or RGBA keep their precision and
signedness, and each writer converts them as OpenJPEG's does:

- BMP: 24-bit RGB from the first three components, or 8-bit gray with a gray
  palette. Deeper samples are rounded to 8 bits as `imagetobmp` rounds them.
  Samples shorter than 8 bits are refused, as OpenJPEG refuses them.
- PNG: 8- and 16-bit samples as they are, 9 to 15 bits scaled to 16, shorter
  color samples scaled to 8. Signed samples are offset to unsigned.
- PNM: `P5` gray (the first component for a `.pgm` name), `P6` RGB, `P7` with
  alpha, maximum value `2^precision - 1`, samples unscaled. Signed samples
  are offset by `2^(precision - 1)`, except in OpenJPEG's 8-bit
  multi-component path, which clamps them.
- TIFF: samples of up to 16 bits, signed ones offset to unsigned.

Palette, ICC, sYCC, channel-definition, and other color interpretations are
decoded to the library's 8-bit sRGB output
(`J2kDecoder::decode_region_srgb8_at_reduction`) and written as 8-bit gray,
RGB, or RGBA.

### Differences from `opj_decompress` 2.5.4

- The `image` crate writes the containers. Gray PNG shorter than 8 bits is
  written as 8-bit (scaled as OpenJPEG scales other short precisions), TIFF
  stores 1- to 16-bit samples in 8- or 16-bit fields instead of packing them,
  and RGB PNM deeper than 8 bits is written as `P7` instead of `P6`. Sample
  values are the same.
- ICC conversion uses j2k's color management rather than Little CMS, and can
  differ by one 8-bit step.
- OpenJPEG writes gray images with a gray ICC profile as RGB with three equal
  channels; j2k writes them as gray.
- Samples deeper than 16 bits are refused for every format; OpenJPEG writes
  them only to BMP.
- Images whose components differ in sampling, precision, or signedness are
  refused. OpenJPEG writes them to PNM as one file per component and to BMP
  as the first component, and refuses them for PNG and TIFF.
- Palette images fail at reductions where the reduced indices don't resolve
  to palette entries; OpenJPEG writes an image for them.
- For samples shorter than 8 bits, and signed 8-bit samples, a reduced decode
  of an area decodes the whole reduced image and then crops it.
- The input is read whole. JPT (JPIP) streams and partial codestreams are not
  decoded.

[Serving IIIF images](https://github.com/frames-sg/j2k/blob/main/book/iiif.md)
in the J2K Book shows how to run j2k behind Cantaloupe and how these
differences were measured.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
