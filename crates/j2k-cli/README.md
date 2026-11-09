# j2k-cli

Command-line entry point for J2K.

The CLI covers inspection, JPEG-to-HTJ2K transcoding and an
`opj_decompress`-compatible decoder. Full compress CLI parity is not claimed
by this package.

Supported commands:

```bash
j2k inspect <file>
j2k transcode <input.jpg> <output.j2k> --htj2k --lossless-53
j2k decode -i <input> -o <output> [opj_decompress options]
```

`transcode` writes an HTJ2K codestream from a supported JPEG input and reports
the source geometry, component count, output bytes, and timing/dispatch summary.

## `opj_decompress` interface

`j2k decode` accepts OpenJPEG 2.5 `opj_decompress` options with the same
meanings. The same binary invoked under the name `opj_decompress` (a symlink or
a copy) behaves as that tool, so programs that run `opj_decompress`, such as
Cantaloupe's `OpenJpegProcessor`, can run j2k unchanged:

```bash
ln -s "$(command -v j2k)" /usr/local/j2k/opj_decompress
opj_decompress -quiet -i image.jp2 -d 512,512,1536,1536 -r 1 -o tile.bmp
```

| Option | Meaning |
| --- | --- |
| `-i <file>` | JP2, JPH or raw J2K/HTJ2K codestream, recognised by its signature, so names without an extension work |
| `-o <file>` | output file; the extension picks the format: `.bmp`, `.png`, `.tif`/`.tiff`, `.pgm`, `.ppm`, `.pnm` |
| `-r <n>` | discard the `n` highest resolution levels; at most the smallest decomposition count of any tile and component |
| `-d <x0,y0,x1,y1>` | decode area on the full-resolution reference grid, so the image's first sample is at its origin (`XOsiz`, `YOsiz`); at reduction `r` it keeps `[ceil(x0 / 2^r), ceil(x1 / 2^r))` of the reduced grid (ITU-T T.800 B.5). Coordinates outside the image area are clamped with a warning; `0,0,0,0` is the whole image |
| `-threads <n\|ALL_CPUS>` | decode threads; without it, `OPJ_NUM_THREADS` (`ALL_CPUS`, or a count capped at twice the processors), else one |
| `-force-rgb` | write gray images as RGB |
| `-l 0` | decode every quality layer (the default) |
| `-quiet` | print no notes or warnings |
| `-h` | print the option summary |

Rejected with an error naming the option and exit status 1: `-l` above 0,
`-t`, `-c`, `-p`, `-x`, `-upsample`, `-split-pnm`, `-allow-partial`,
`-ImgDir`, `-OutFor`, `.jpt` input, and `.pgx`, `.raw`, `.rawl`, `.yuv` and
`.tga` output. j2k cannot honour them exactly, and an ignored option would
change what the caller receives.

Nothing is printed to stdout: notes, warnings and errors go to stderr, so an
`-o` path linked to `/dev/stdout` carries only the image, as Cantaloupe
expects. An input that is not JPEG 2000 fails with
`Unknown input file format`, the message Cantaloupe maps to a source-format
error. The help text names no `vX.Y.Z` version; Cantaloupe reads one from
`opj_decompress -h` and stops passing `-quiet` below 2.2.

Exit status: invoked as `opj_decompress`, every failure returns 1 and `-h`
returns 1 after printing to stdout, as OpenJPEG does. As `j2k decode`, argument
errors return 2, runtime failures 1, and help returns 0 on stderr.

### Samples and colour

Images whose components are Gray, RGB or RGBA keep their precision and
signedness, and each writer converts them as OpenJPEG's does:

- BMP: 24-bit RGB from the first three components or 8-bit gray with a gray
  palette; deeper samples are rounded to 8 bits as `imagetobmp` rounds them;
  samples shorter than 8 bits are refused, as OpenJPEG refuses them.
- PNG: 8- and 16-bit samples as they are, 9-15 bits scaled to 16, shorter
  colour samples scaled to 8; signed samples are offset to unsigned.
- PNM: `P5` gray (the first component for a `.pgm` name), `P6` RGB, `P7` with
  alpha, maximum value `2^precision - 1`, samples unscaled; signed samples are
  offset by `2^(precision - 1)` except in OpenJPEG's 8-bit multi-component
  path, which clamps them.
- TIFF: samples of up to 16 bits, signed ones offset to unsigned.

Palette, ICC, sYCC, channel-definition and other colour interpretations are
decoded to the facade's 8-bit sRGB output (`decode_region_srgb8_at_reduction`)
and written as 8-bit gray, RGB or RGBA.

Differences from `opj_decompress` 2.5.4:

- The `image` crate writes the containers: gray PNG shorter than 8 bits is
  written as 8-bit (scaled as OpenJPEG scales other short precisions), TIFF
  stores 1-16-bit samples in 8- or 16-bit fields instead of packing them, and
  RGB PNM deeper than 8 bits is written as `P7` instead of `P6`. Sample values
  are the same.
- ICC conversion uses j2k's colour management rather than Little CMS, and can
  differ by one 8-bit step.
- OpenJPEG writes gray images with a gray ICC profile as RGB with three equal
  channels; j2k writes them as gray.
- Samples deeper than 16 bits are refused for every format; OpenJPEG writes
  them only to BMP.
- Images whose components differ in sampling, precision or signedness are
  refused; OpenJPEG writes them to PNM as one file per component and to BMP as
  the first component, and refuses them for PNG and TIFF.
- Palette images fail at reductions where the reduced indices do not resolve
  to palette entries; OpenJPEG writes an image for them.
- Samples shorter than 8 bits and signed 8-bit samples at a reduction decode
  the whole reduced image and keep the requested area.
- The input is read whole; JPT (JPIP) streams and partial codestreams are not
  decoded.

[`docs/iiif.md`](../../docs/iiif.md) describes the parity sweep that measured
these differences and the Cantaloupe configuration.

## Links

- Package page: <https://docs.rs/j2k-cli>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
