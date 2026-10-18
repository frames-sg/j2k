# j2k-native fuzz targets

- `native_decode_fuzz` decodes a codestream with `decode_native`. Depending on
  the low bits of the input length, it also decodes a region and the
  reversible 5/3 coefficients, in strict or lenient mode.
- `lossless_roundtrip_fuzz` encodes an image chosen by the input with the
  reversible transform, classic or HTJ2K, and requires the decode to return
  the input samples exactly.

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

The three synthetic inputs in `corpus/lossless_roundtrip_fuzz/` select classic
8×8 grayscale at 8 bits (1 level), HTJ2K 16×16 RGB at 16 bits (2 levels,
MCT, RPCL, SOP and EPH), and classic 33×25 RGB at 8 bits (3 levels, PCRL,
4×4 code-blocks and differing reference-grid precinct spans). Their trailing
ASCII bytes supply the samples. These seeds exceed the ten-byte header so the
512-run CI smoke test exercises both encoder paths and multi-precinct packet
ordering immediately; an empty corpus only reaches eight-byte inputs.
