OpenJPEG position-major progression fixtures.

These codestreams use the PCRL and CPRL progression orders with several
precincts per resolution, and precinct sizes that differ between
resolutions. In these orders, packets of different resolutions interleave by
position on the reference grid, so they exercise packet ordering that the
LRCP, RLCP and RPCL orders do not. They come from OpenJPEG to provide an
independent interoperability reference for j2k's decoder.

`generate.sh` regenerates every file with OpenJPEG 2.5.3 (`opj_compress`,
openjp2 library v2.5.3, Debian package `libopenjp2-tools`
2.5.3-2.1~deb13u2). Each codestream has three decomposition levels, 8x8
code-blocks, 32x32 precincts at the two highest resolutions with smaller ones
below, and three quality layers, the last of them lossless.

| File | Contents | SHA-256 |
| --- | --- | --- |
| `source_97x61.rgb` | 97 x 61 interleaved RGB8 source samples, row-major. Every codestream decodes to these exactly. | `1595ad560ad5978b483d4df27685b95e41b89bdfa0ab0ef2f87cd99d2c02826a` |
| `pcrl_97x61.j2k` | PCRL, one tile, one tile-part, no PLT. | `79e0c8847d51ac84b7893cdc4c6c4c0188ea2accb488bb44bcc9b4a62e0ff278` |
| `cprl_97x61.j2k` | CPRL, one tile, one tile-part, no PLT. | `41cffc1f753d8658c3221492746a8693d7417a8a7ee32f2093bcfe4987cb2b53` |
| `pcrl_97x61_tiles_plt.j2k` | PCRL, two partial 64x64 tiles, split into tile-parts at every resolution change (`-TP R`, 48 per tile), each with PLT. | `7f1f00eb09c0730177b0c575337462eeb9a52f83ae8ba3876e03a47c78987ccf` |
| `cprl_97x61_tiles_plt.j2k` | CPRL, two partial 64x64 tiles, split into tile-parts at every resolution change (`-TP R`, 48 per tile), each with PLT. | `a7cc77b53e5e889efa4091da970a26583362f51ff522b16c95863e519969de17` |

Kakadu 8.6.2 (`kdu_expand`) and OpenJPEG 2.5.3 (`opj_decompress`) both decode
each codestream to `source_97x61.rgb` exactly.

With `-TP R` and 48x32 tiles, OpenJPEG 2.5.3 writes codestreams that both
Kakadu and OpenJPEG reject, which is why the tiled fixtures use 64x64 tiles.
