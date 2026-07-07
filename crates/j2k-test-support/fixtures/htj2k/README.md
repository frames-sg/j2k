HTJ2K fixtures for shared test support.

These fixtures mirror the tiny OpenHTJ2K-derived HTONLY codestream fixtures in
`crates/j2k-native/fixtures/htj2k/`. They are copied from OpenHTJ2K commit
`ffe5acf9f1eedb87c36c3fd2134fdc1ddea5e75f`.

`openhtj2k_ds0_ht_12_b11.j2k` is copied from `ds0_ht_12_b11.j2k`, blob
`cf3fb0bc7e55898b4e6977f38ba0d38d91c359bf`.

`openhtj2k_ds0_ht_09_b11.j2k` is copied from `ds0_ht_09_b11.j2k`, blob
`d4f2031359c32eb24825d00dde05a92cf3ae451e`.

The paired `.gray` files contain expected 8-bit grayscale samples in row-major
order from decoding the checked-in codestreams with OpenJPH.

The OpenHTJ2K BSD 3-Clause license is retained in `LICENSE.OpenHTJ2K`.
