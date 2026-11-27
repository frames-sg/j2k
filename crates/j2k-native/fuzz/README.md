# j2k-native fuzz targets

- `native_decode_fuzz` decodes a codestream with `decode_native`. Depending on
  the low bits of the input length, it also decodes a region and the
  reversible 5/3 coefficients, in strict or lenient mode. Images over the
  whole-image pixel cutoff still exercise a bounded region at the
  right/bottom edge; smaller images exercise a corner chosen by the input
  length.
- `lossless_roundtrip_fuzz` encodes an image chosen by the input with the
  reversible transform, classic or HTJ2K, and checks that decoding returns
  the input samples exactly.

`cargo xtask fuzz-run` runs these and the other workspace fuzz targets with
nightly Rust and `cargo-fuzz`, for `J2K_FUZZ_RUNS` runs each (default 1000).
To run one target from this crate's directory:

```sh
cargo +nightly fuzz run native_decode_fuzz
```

## Seeds

The seeds in `corpus/native_decode_fuzz/` were encoded by `j2k-native` itself
from synthetic samples, so they carry no third-party content:

| Seed | Block coding | Image | Options |
| --- | --- | --- | --- |
| `classic_gray8_lrcp.j2c` | classic | 16×16, 1 × 8-bit | 2 levels |
| `htj2k_rgb8_rpcl_sop_eph.j2c` | HTJ2K | 16×16, 3 × 8-bit | 2 levels, MCT, RPCL, SOP, EPH |
| `classic_gray12_tiles_layers_plt_tlm.j2c` | classic | 24×20, 1 × 12-bit | 2 levels, 8×8 code-blocks, 16×16 tiles, 3 layers, PCRL, PLT, TLM |
| `htj2k_gray16_cprl.j2c` | HTJ2K | 17×9, 1 × 16-bit | 1 level, CPRL |
| `classic_rgb8_irreversible97_rlcp.j2c` | classic | 16×16, 3 × 8-bit | 9/7, 2 levels, MCT, RLCP |
| `classic_gray1.j2c` | classic | 9×7, 1 × 1-bit | 1 level |

Without them, random input rarely parses as a codestream at all.

`../fixtures/large_geometry/` has more long images, encoded independently.
Add that directory as a read-only seed corpus for large-coordinate runs: its
codestreams exercise region decoding even though their whole-image area is
over the target's cutoff.

The three synthetic inputs in `corpus/lossless_roundtrip_fuzz/` select classic
8×8 grayscale at 8 bits (1 level), HTJ2K 16×16 RGB at 16 bits (2 levels,
MCT, RPCL, SOP and EPH), and classic 33×25 RGB at 8 bits (3 levels, PCRL,
4×4 code-blocks and differing reference-grid precinct spans). Their trailing
ASCII bytes supply the samples. They are longer than the ten-byte header, so
even the 512-run CI smoke test exercises both encoders and multi-precinct
packet ordering; an empty corpus only reaches eight-byte inputs.
