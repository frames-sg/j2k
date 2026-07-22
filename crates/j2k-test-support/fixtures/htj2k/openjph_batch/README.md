# Independent OpenJPH batch fixtures

This directory contains small HTJ2K fixtures encoded independently of this
workspace with OpenJPH 0.27.0. The compressed fixtures use a 19 x 13 image,
11 x 7 tiles, 8 x 8 code blocks, and two wavelet decompositions. This gives
odd image, tile, edge-code-block, and multi-tile geometry without making the
checked-in corpus large.

The reversible 5/3 matrix covers signed and unsigned Gray and RGB at 8, 12,
and 16 bits. The irreversible 9/7 matrix adds unsigned RGB8 plus unsigned and
signed Gray12, Gray16, and RGB12 coverage. The >8-bit 9/7 fixtures use
precision-scaled quantization steps (`0.003 / 2^(precision - 8)`) so independent
integer reconstruction can be compared within one native LSB. Their source
patterns deliberately stay in an 8-bit amplitude range while retaining the
declared 12/16-bit and signed sample domains; this isolates 9/7 reconstruction,
native packing, and sign restoration. `gray_u12_53.jph` contains the
independently encoded OpenJPH codestream in a minimal JPH file wrapper.

The artifacts were generated on 2026-07-18 with the Homebrew arm64 OpenJPH
0.27.0 executables. The exact executable SHA-256 values were:

```text
b9846d39ca27506e0a93c66e42b287f4730ad071a26fc54bd4024aedaecf280f  ojph_compress
4b420506bd2a44439cf472d956bc1552c8be72ff6dffe2c10042e0e36b8de843  ojph_expand
```

OpenJPH is distributed under the BSD 2-Clause license reproduced in
`LICENSE.OpenJPH`.

## Reproduction

From this directory, use a temporary working directory and build the
deterministic source/oracle utility:

```sh
rustc --edition 2021 generate.rs -o "$work/generate"
"$work/generate" sources "$work/sources"
```

For each unsigned PGM/PPM source, run `ojph_compress` with the following exact
codec options. RGB commands also include `-colour_trans false`.

```sh
ojph_compress -i "$source" -o "$output" \
  -reversible true -num_decomps 2 -block_size '{8,8}' -tile_size '{11,7}'
```

For signed grayscale, run one command per precision (`8`, `12`, and `16`):

```sh
ojph_compress -i "$work/sources/gray_s${precision}_53.source.raw" \
  -o "$work/gray_s${precision}_53.j2c" \
  -dims '{19,13}' -num_comps 1 -signed true -bit_depth "$precision" \
  -downsamp '{1,1}' -reversible true -num_decomps 2 \
  -block_size '{8,8}' -tile_size '{11,7}'
```

OpenJPH 0.27.0's CLI YUV reader loads samples into unsigned containers even
when signed component metadata is requested. Signed RGB is therefore encoded
through the public OpenJPH library, supplying actual negative `i32` component
lines instead of accepting invalid CLI-YUV output:

```sh
c++ -std=c++17 -Wall -Wextra -Werror \
  -I/opt/homebrew/opt/openjph/include/openjph encode_signed_rgb.cpp \
  -L/opt/homebrew/opt/openjph/lib \
  -Wl,-rpath,/opt/homebrew/opt/openjph/lib -lopenjph \
  -o "$work/encode_signed_rgb"
"$work/encode_signed_rgb" "$work/rgb_s8_53.j2c" 8
"$work/encode_signed_rgb" "$work/rgb_s12_53.j2c" 12
"$work/encode_signed_rgb" "$work/rgb_s16_53.j2c" 16
"$work/encode_signed_rgb" "$work/rgb_s8_53_single.j2c" 8 single-tile
"$work/encode_signed_rgb" "$work/rgb_s12_53_single.j2c" 12 single-tile
"$work/encode_signed_rgb" "$work/rgb_s16_53_single.j2c" 16 single-tile
```

The single-tile variants retain the same deterministic samples and therefore
share the corresponding independently decoded `.oracle.raw` files. They
exercise adapter offset-plan reuse, while the original variants retain the
odd multi-tile coverage.

The irreversible fixture uses the checked-in RGB8 PPM source:

```sh
ojph_compress -i "$work/sources/rgb_u8_97.source.ppm" \
  -o "$work/rgb_u8_97.j2c" -reversible false -qstep 0.003 \
  -num_decomps 2 -block_size '{8,8}' -tile_size '{11,7}' \
  -colour_trans false
```

The checked-in `*_97.source.*` files reproduce the >8-bit single-tile 9/7
fixtures. Gray12/RGB12 use `-qstep 0.0001875`; Gray16 uses
`-qstep 0.00001171875`. Signed grayscale uses the same raw-input flags as the
reversible commands. Signed RGB12 uses the library helper with both
`irreversible` and `single-tile` arguments:

```sh
"$work/encode_signed_rgb" "$work/rgb_s12_97.j2c" 12 irreversible single-tile
```

Each `.oracle.raw` file is derived from a separate OpenJPH decode. PFM is used
as the intermediate because it preserves signed reconstructed sample codes;
the utility reverses PFM's bottom-up rows and removes its left bit alignment:

```sh
ojph_expand -i "$work/$name.j2c" -o "$work/$name.oracle.pfm"
"$work/generate" oracle "$work/$name.oracle.pfm" \
  "$work/$name.oracle.raw" "$precision" "$signed"
```

Oracle samples are top-down, interleaved NHWC. Precision up to 8 bits uses one
byte per sample; higher precision uses little-endian 16-bit containers. Signed
values use two's-complement `i8` or `i16` representation. The reversible
oracles were also checked against the deterministic source formula:

```sh
"$work/generate" verify "$work/$name.oracle.raw" \
  "$components" "$precision" "$signed"
```

OpenJPH 0.27.0 writes raw Part 15 codestreams but does not serialize the JPH
box container. The fixture utility independently constructs the standard JPH
signature, `ftyp`, `jp2h` (`ihdr` plus `colr`), and `jp2c` boxes around the
OpenJPH-generated Gray12 codestream:

```sh
"$work/generate" jph "$work/gray_u12_53.j2c" \
  "$work/gray_u12_53.jph" 1 12 false
```
