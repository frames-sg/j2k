HTJ2K fixture for CUDA refinement-plan and runtime-boundary coverage.

The local `openhtj2k_ds0_ht_09_b11.j2k` fixture is from the OpenHTJ2K
conformance data, copied from `ds0_ht_09_b11.j2k` at OpenHTJ2K commit
`ffe5acf9f1eedb87c36c3fd2134fdc1ddea5e75f`, blob
`d4f2031359c32eb24825d00dde05a92cf3ae451e`. It is a tiny HTONLY codestream
derived from the JPEG 2000 Part 4 / ITU-T T.803 HTJ2K conformance set.

The local `.j2k` preserves the licensed fixture copy beside this README and its
OpenHTJ2K license. Current CUDA host-surface, plan, and kernel tests source the
codestream bytes through `j2k_test_support::openhtj2k_refinement_odd_fixture`,
backed by
`crates/j2k-test-support/fixtures/htj2k/openhtj2k_ds0_ht_09_b11.j2k`. The
host-surface test uses the paired local `openhtj2k_ds0_ht_09_b11.gray` oracle.

The fixture is intentionally checked-in test data: tests must not invoke an
external encoder or decoder at runtime.

The local `.gray` file contains expected 8-bit grayscale samples in row-major
order from decoding the checked-in codestream with OpenJPH.

The OpenHTJ2K source license is retained in `LICENSE.OpenHTJ2K`.
