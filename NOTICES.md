# Notices

Portions of the native JPEG 2000 engine were originally imported from
dicom-toolkit-jpeg2000 0.5.0, a Hayro-derived implementation, and have since
been modified and extended for j2k. Applicable upstream copyright and license
notices are retained.

The following standards and implementations were also used for derived code,
design, verification, generated tables, or small test/data assets as noted. This file
records source and asset provenance; it does not make a legal-compliance
determination.

- **ITU-T Recommendation T.81 (09/92)** — *Information technology — Digital
  compression and coding of continuous-tone still images — Requirements and
  guidelines.* Primary reference for all SOF variants (baseline, extended
  sequential, progressive, lossless Annex H), marker semantics, Huffman coding,
  and color transform behavior.

- **libjpeg-turbo** <https://libjpeg-turbo.org/> and the Independent JPEG
  Group's software — The JPEG codec reproduces libjpeg-turbo's output bit for
  bit, and parts of it are derived from libjpeg-turbo source, rewritten in
  Rust and Metal Shading Language:
  - the integer ISLOW inverse DCTs, full size and reduced, from `jidctint.c`,
    `jidctred.c`, and `jidctint-avx2.asm`:
    `crates/j2k-codec-math/src/jpeg/idct.rs`, `crates/j2k-jpeg/src/idct/`, and
    the JPEG Metal shaders;
  - the "fancy" chroma upsampling filters, from `jdsample.c`:
    `crates/j2k-codec-math/src/jpeg/upsample.rs`,
    `crates/j2k-jpeg/src/color/upsample.rs`, and the JPEG Metal shaders;
  - the YCbCr to RGB conversion constants, from `jdcolor.c` and
    `jdcolor-neon.c`: `crates/j2k-codec-math/src/jpeg/ycbcr.rs`,
    `crates/j2k-jpeg/src/color/ycbcr.rs`, and
    `crates/j2k-jpeg/src/backend/neon.rs`;
  - optimal Huffman table construction, from `jpeg_gen_optimal_table` in
    `jchuff.c`: `crates/j2k-jpeg/src/optimal_huffman.rs`.

  Marker handling, color-space selection, YCCK conversion, lossless
  prediction, and the bit reader's fill policy were written to match
  libjpeg-turbo's behavior. These parts of libjpeg-turbo are covered by the
  IJG License; its SIMD sources also bear the zlib License. The license text
  is in `LICENSE.libjpeg-turbo` and `README.ijg` in `crates/j2k-codec-math/`,
  `crates/j2k-jpeg/`, `crates/j2k-jpeg-metal/`, and `crates/j2k-python/`, and
  each derived Rust file names its upstream file and copyright holders. This
  software is based in part on the work of the Independent JPEG Group.

  libjpeg-turbo is also the oracle for parity verification. Expected outputs
  in `corpus/conformance/` and in the `conformance`, `lossless_matrix`, and
  `scaled_matrix` directories of `crates/j2k-test-support/fixtures/` are
  generated using libjpeg-turbo's `cjpeg` and `djpeg` command-line tools and
  are committed to the repository; the pinned version is recorded in
  `corpus/conformance/manifest.json`.

- **OpenHTJ2K conformance fixtures** — Tiny HTONLY codestream fixtures in
  `crates/j2k-native/fixtures/htj2k/`, shared-test copies in
  `crates/j2k-test-support/fixtures/htj2k/`, and CUDA test copies in
  `crates/j2k-cuda/tests/fixtures/htj2k/` are copied from OpenHTJ2K commit
  `ffe5acf9f1eedb87c36c3fd2134fdc1ddea5e75f`, except
  `openhtj2k_sigprop_refinement_overlap.j2k`, which is from commit
  `3a1f96f63492ffff167ed8a764c3e362d2491bd9`. The retained upstream BSD
  3-Clause license is adjacent to all three fixture sets:
  `crates/j2k-native/fixtures/htj2k/LICENSE.OpenHTJ2K` and
  `crates/j2k-test-support/fixtures/htj2k/LICENSE.OpenHTJ2K`, and
  `crates/j2k-cuda/tests/fixtures/htj2k/LICENSE.OpenHTJ2K`.

- **pydicom-data** <https://github.com/pydicom/pydicom-data> — The lossless
  JPEG 2000 fixture `emri_small_jpeg_2k_lossless_frame1.j2c` in
  `crates/j2k-native/fixtures/reversible_early_stop/` is one frame of
  `emri_small_jpeg_2k_lossless.dcm`, with expected samples from the same
  series' uncompressed `emri_small.dcm`; provenance and hashes are in that
  directory's `README.md`. The retained upstream MIT license is
  `crates/j2k-native/fixtures/reversible_early_stop/LICENSE.pydicom-data`.

- **OpenJPEG** <https://www.openjpeg.org/> — Reference implementation used as
  an encoder and oracle for small test assets. The synthetic codestreams in
  `crates/j2k-native/fixtures/reversible_early_stop/` (such as
  `openjpeg_53_rate3_gray12_64x64.j2k`) and their expected samples were
  produced with OpenJPEG 2.5.4's `opj_compress` and `opj_decompress`; the
  commands are in that directory's `README.md`. OpenJPEG also encoded the
  synthetic fixtures in the `image_origin`, `large_geometry`, `multi_segment`,
  `packet_directory`, and `quality_layers` directories of
  `crates/j2k-native/fixtures/` and the `empty_coarse_levels`,
  `openjpeg_origin`, and `openjpeg_progression` directories of
  `crates/j2k-test-support/fixtures/`. No OpenJPEG source code is incorporated
  into `j2k`.

- **DCMTK** <https://dicom.offis.de/dcmtk> — The JPEG codestreams in
  `crates/j2k-test-support/fixtures/dicom_matrix/` were written by DCMTK 3.7's
  `dcmcjpeg`, with its `dcmdjpeg` decode as the reference. No DCMTK source code
  is incorporated into `j2k`.

- **Compact ICC Profiles** <https://github.com/saucecontrol/Compact-ICC-Profiles>
  — `ProPhoto-v2-micro.icc` and `CGATS001Compat-v2-micro.icc` in
  `crates/j2k-native/assets/` are from this project. The decoder embeds the
  ProPhoto profile for ROMM-RGB images, and `LAB.icc`, created for this project
  with Little CMS, for CIE Lab images. The grayscale profiles `sGrey-v2-micro.icc` and
  `sGrey-v4.icc` in `crates/j2k/fuzz/corpus/srgb8_fuzz/` (from commit
  `bdd84663061bc4ae95ca70decff54f581e27f702`) seed `srgb8_fuzz`. The profiles
  are available under CC0 1.0 Universal; the retained license text is in
  `crates/j2k-native/assets/LICENSE.txt`.

- **OpenJPH** <https://github.com/aous72/OpenJPH> — HTJ2K lookup tables in
  `crates/j2k-native/src/j2c/ht_tables.rs` and
  `crates/j2k-native/src/j2c/ht_encode_tables.rs` are generated from
  OpenJPH source tables. They are table data, not linked OpenJPH source files.
  The fixtures in `crates/j2k-test-support/fixtures/htj2k/openjph_batch/`, and
  their copies in `crates/j2k-native/fixtures/htj2k/`, were encoded with
  OpenJPH 0.27.0; the retained BSD 2-Clause license is `LICENSE.OpenJPH` in
  each directory.
