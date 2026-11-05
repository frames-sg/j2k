// SPDX-License-Identifier: MIT OR Apache-2.0

//! Synthetic images shared by the integration tests and `make_fixture`.

mod encode;

pub(crate) use encode::encode;

/// A 32x24 8-bit gray codestream and its source pixels.
pub(crate) fn gray_fixture(ht: bool) -> (Vec<u8>, Vec<u8>) {
    let pixels: Vec<u8> = (0..32 * 24)
        .map(|i| u8::try_from((i * 13 + 7) % 256).unwrap())
        .collect();
    (encode(&pixels, 32, 24, 1, 8, false, ht), pixels)
}
