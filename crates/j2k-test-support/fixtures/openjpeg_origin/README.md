OpenJPEG and OpenJPH odd-origin fixtures.

`rgb_23x17_origin_3_5.j2k` (classic block coding, from OpenJPEG) and
`rgb_23x17_origin_3_5_ht.j2c` (HTJ2K block coding, from OpenJPH) place one
23 x 17 RGB8 image at reference-grid origin (3, 5), which is odd at every
resolution. A reduced image spans `[ceil(origin / s), ceil(extent / s))` on
each axis (ITU-T T.800 B.5), so at every reduction it is narrower than
`ceil(size / s)`: 11 x 8 rather than 12 x 9 at one level, 6 x 4 rather than
6 x 5 at two, and 3 x 2 rather than 3 x 3 at three. The `reduced_r*.rgb`
files are OpenJPEG's own decodes at those reductions, an independent
reference for j2k's reduced-resolution and region decodes of images with
such origins.

`generate.sh` regenerates every file with OpenJPEG 2.5.4 (`opj_compress` and
`opj_decompress`, openjp2 library v2.5.4, Homebrew `openjpeg`) and OpenJPH
0.27.0 (`ojph_compress`, Homebrew `openjph`). Both codestreams have three
decomposition levels, one tile, the reversible colour transform and
reversible 5/3 lossless coding. The script checks that OpenJPEG decodes both
to the source at full resolution and to the same samples at every
reduction.

| File | Contents | SHA-256 |
| --- | --- | --- |
| `source_23x17.rgb` | 23 x 17 interleaved RGB8 source samples, row-major. Both codestreams decode to these exactly. | `e8eaca5743093c518fba4da973fe1fd9b7e37ac5960d9abf968ec12480b0d5c8` |
| `rgb_23x17_origin_3_5.j2k` | The source at origin (3, 5), one tile, classic block coding. | `3aedce96f2592824a6660127a1e5e1e20335565b64f91cd8f52199cb99d84e0d` |
| `rgb_23x17_origin_3_5_ht.j2c` | The source at origin (3, 5), one tile, HTJ2K block coding. | `e596b270e18a4a1b44a879c52db957bac6db21506400c7c9a1eeef2f5110a0ee` |
| `reduced_r1.rgb` | OpenJPEG's decode with one resolution discarded (`-r 1`): 11 x 8 RGB8. | `ba5d5d2a739a66b63b1435a40a39b6d2e36635522fa099c94021a1575c46996a` |
| `reduced_r2.rgb` | OpenJPEG's decode with two resolutions discarded (`-r 2`): 6 x 4 RGB8. | `c295e2b5ee2eee1444a0c31192502a4950b3cb6aab184b841e4d2d6f7edfee20` |
| `reduced_r3.rgb` | OpenJPEG's decode with three resolutions discarded (`-r 3`): 3 x 2 RGB8. | `da7a762f572dcb699ecce7e000054c6f887eaefb5addb3521ce1157b7db3a7e6` |
