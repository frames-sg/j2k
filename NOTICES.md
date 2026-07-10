# Notices

`j2k` includes an in-repository pure-Rust JPEG 2000 codec engine imported from
[`dicom-toolkit-jpeg2000` 0.5.0](https://docs.rs/dicom-toolkit-jpeg2000/0.5.0/dicom_toolkit_jpeg2000/)
and adapted for this workspace. The source project is
[`dicom-toolkit-rs`](https://github.com/knopkem/dicom-toolkit-rs), and that
codec is a maintained fork of the original `hayro-jpeg2000` project from the
upstream [`Hayro`](https://github.com/LaurenzV/hayro) repository. The retained
MIT license at `crates/j2k-native/LICENSE-MIT` identifies "The Hayro Authors"
as the copyright holders.

The following standards and implementations were also used for design,
verification, generated tables, or small test/data assets as noted. This file
records source and asset provenance; it does not make a legal-compliance
determination.

- **ITU-T Recommendation T.81 (09/92)** — *Information technology — Digital
  compression and coding of continuous-tone still images — Requirements and
  guidelines.* Primary reference for all SOF variants (baseline, extended
  sequential, progressive, lossless Annex H), marker semantics, Huffman coding,
  and color transform behavior.

- **libjpeg-turbo** <https://libjpeg-turbo.org/> — Reference implementation
  used as an oracle for parity verification. Expected outputs in
  `corpus/conformance/` are generated using libjpeg-turbo's `cjpeg` and `djpeg`
  command-line tools and are committed to the repository; the pinned version
  is recorded in `corpus/conformance/manifest.json`. No libjpeg-turbo source
  code is incorporated into `j2k`.

- **OpenHTJ2K conformance fixtures** — Tiny HTONLY codestream fixtures in
  `crates/j2k-native/fixtures/htj2k/`, shared-test copies in
  `crates/j2k-test-support/fixtures/htj2k/`, and CUDA test copies in
  `crates/j2k-cuda/tests/fixtures/htj2k/` are copied from OpenHTJ2K commit
  `ffe5acf9f1eedb87c36c3fd2134fdc1ddea5e75f`. The retained upstream BSD
  3-Clause license is adjacent to all three fixture sets:
  `crates/j2k-native/fixtures/htj2k/LICENSE.OpenHTJ2K` and
  `crates/j2k-test-support/fixtures/htj2k/LICENSE.OpenHTJ2K`, and
  `crates/j2k-cuda/tests/fixtures/htj2k/LICENSE.OpenHTJ2K`.

- **Compact ICC Profiles** <https://github.com/saucecontrol/Compact-ICC-Profiles>
  — ICC profile assets in `crates/j2k-native/assets/` are used for
  color-management tests. The profiles are available under CC0 1.0 Universal;
  the retained license text is in
  `crates/j2k-native/assets/LICENSE.txt`.

- **OpenJPH** <https://github.com/aous72/OpenJPH> — HTJ2K lookup tables in
  `crates/j2k-native/src/j2c/ht_tables.rs` and
  `crates/j2k-native/src/j2c/ht_encode_tables.rs` are generated from
  OpenJPH source tables. They are table data, not linked OpenJPH source files.
