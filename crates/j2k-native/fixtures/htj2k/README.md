HTJ2K fixtures for native decoder coverage.

The OpenHTJ2K fixtures are copied from OpenHTJ2K commit
`ffe5acf9f1eedb87c36c3fd2134fdc1ddea5e75f`. They are tiny HTONLY codestreams
derived from the JPEG 2000 Part 4 / ITU-T T.803 HTJ2K conformance set.

`openhtj2k_ds0_ht_12_b11.j2k` is copied from `ds0_ht_12_b11.j2k`, blob
`cf3fb0bc7e55898b4e6977f38ba0d38d91c359bf`. The native decoder sees 8 HT code
blocks, 2 non-empty refinement jobs, and up to 3 HT coding passes.

`openhtj2k_ds0_ht_09_b11.j2k` is copied from `ds0_ht_09_b11.j2k`, blob
`d4f2031359c32eb24825d00dde05a92cf3ae451e`. The native decoder sees 14 HT
code blocks, all with non-empty refinement jobs, and up to 3 HT coding passes.

It is intentionally checked-in test data: tests must not invoke an external
encoder or decoder at runtime.

The paired `.gray` file contains the expected 8-bit grayscale samples in
row-major order from decoding the checked-in codestream with OpenJPH.

The OpenHTJ2K source license is retained in `LICENSE.OpenHTJ2K`.

The `gray_u12_53` and `rgb_u12_53` pairs are package-local copies of the
independent OpenJPH batch fixtures used by the native multi-tile integration
tests. OpenJPH 0.27.0 encoded them on 2026-07-18 as reversible 5/3 raw Part 15
codestreams with a 19 x 13 image, 11 x 7 tiles, 8 x 8 code blocks, and two
wavelet decompositions. A separate OpenJPH decode produced each top-down raw
oracle; 12-bit samples use little-endian `u16` containers and RGB samples are
interleaved. The generating `ojph_compress` and `ojph_expand` executable
SHA-256 values were, respectively,
`b9846d39ca27506e0a93c66e42b287f4730ad071a26fc54bd4024aedaecf280f` and
`4b420506bd2a44439cf472d956bc1552c8be72ff6dffe2c10042e0e36b8de843`.

The copied artifact SHA-256 values are:

```text
f2735a7b4911f82ce53f4e0da9c155900f0974be43a9f813612ee5215c7492aa  gray_u12_53.j2c
8962a79ec54f5bb1c244bfce724b77c6e3d16367b8a5cec10a4b7809b7bc2b85  gray_u12_53.oracle.raw
ae47e7be0f46df81c1da8a59e06ef76ec310992254344ffc27c62fc58d800b16  rgb_u12_53.j2c
88653085de44d28bd02f19e8a245d9d7c000be33ce33d6cce5087818933ac15b  rgb_u12_53.oracle.raw
```

The canonical source/oracle generator and full reproduction commands remain in
`crates/j2k-test-support/fixtures/htj2k/openjph_batch/` in the repository. The
OpenJPH BSD 2-Clause license is retained here in `LICENSE.OpenJPH` so packaged
native tests keep their fixture provenance and license material together.
