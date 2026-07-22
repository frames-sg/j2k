HTJ2K fixtures for shared test support.

These fixtures mirror the tiny OpenHTJ2K-derived HTONLY codestream fixtures in
`crates/j2k-native/fixtures/htj2k/`. They are copied from OpenHTJ2K commit
`ffe5acf9f1eedb87c36c3fd2134fdc1ddea5e75f`.

`openhtj2k_ds0_ht_12_b11.j2k` is copied from `ds0_ht_12_b11.j2k`, blob
`cf3fb0bc7e55898b4e6977f38ba0d38d91c359bf`.

`openhtj2k_ds0_ht_09_b11.j2k` is copied from `ds0_ht_09_b11.j2k`, blob
`d4f2031359c32eb24825d00dde05a92cf3ae451e`.

`openhtj2k_hifi_ht1_02.j2k` is copied from `hifi_ht1_02.j2k`, blob
`61ced26eac84e240e28bc14e38928581b0601c01`. It is a 128 x 128 unsigned
RGB12 conformance codestream whose code blocks include exactly two coding
passes (`Cleanup` plus `SigProp`) as well as three-pass refinement. Its SHA-256
is `8ecb1ddcd469e4b3fda6df01c919e831e16f7c3bc19a0b3c41d72038f2b44e53`.

The paired `.gray` files contain expected 8-bit grayscale samples in row-major
order from decoding the checked-in codestreams with OpenJPH.

`openhtj2k_hifi_ht1_02.oracle.raw` contains top-down interleaved little-endian
RGB12 samples decoded with OpenJPH 0.27.0. Its SHA-256 is
`5ae8c1c25d8bffc4701c2b72d8000e34cf8b4e96e665d4177dc33dd0f6244d8d`;
the `ojph_expand` executable SHA-256 was
`4b420506bd2a44439cf472d956bc1552c8be72ff6dffe2c10042e0e36b8de843`.

`openhtj2k_sigprop_refinement_overlap.j2k` is copied from
`tests/data/sigprop_refinement_overlap.j2k` at OpenHTJ2K commit
`3a1f96f63492ffff167ed8a764c3e362d2491bd9`, blob
`f31e936d6e1903593a33d6b611345db49efe2b4f`. It is a 512 x 64 unsigned
RGB8 stream with cleanup-only, two-pass SigProp, and three-pass MagRef blocks,
including a refinement byte whose set stuffed position overlaps the reverse
MagRef stream. Its SHA-256 is
`dec18535d0d6b9e113c0ea23a319e0cc77cd4e5ec9e37fe1e827b9489dca615d`.
The paired `.openht.oracle.ppm` is top-down interleaved RGB8 decoded with
`open_htj2k_dec` built from that same OpenHTJ2K commit. Its SHA-256 is
`d592eea6dc7d5d28693d0c5eb92cfcb7d51372e4acd8ffb1664f69b7dce7f3da`;
the decoder executable SHA-256 was
`483c92fa604823f98d899642367f68a1ae62a91276600aeb68cdfbaddbcd0fa4`.

OpenJPH 0.27.0 produces a materially different result for this deliberately
overlapping refinement stream: 305 of its output bytes differ from OpenHTJ2K
by more than one LSB. That output is retained as
`.openjph.oracle.raw` (SHA-256
`dddeda5af064c87abfce577ea1e6c2d714c203cef87ebb5bd3bf2507efe052b0`)
as cross-decoder evidence, but is not used as the correctness oracle for this
OpenHTJ2K-specific edge case. The independent `hifi_ht1_02` fixture above and
the `openjph_batch/` corpus continue to provide OpenJPH parity coverage.

The OpenHTJ2K BSD 3-Clause license is retained in `LICENSE.OpenHTJ2K`.

The `openjph_batch/` directory is a separate OpenJPH 0.27.0 corpus for the
owned batch API. It covers native signed and unsigned Gray/RGB sample types,
reversible and irreversible transforms, raw Part 15 and JPH input, and odd
multi-tile geometry. Its reproduction commands, decoded raw-oracle format, and
OpenJPH license are documented in `openjph_batch/README.md`.
