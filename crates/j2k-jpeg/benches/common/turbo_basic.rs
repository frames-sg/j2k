// SPDX-License-Identifier: MIT OR Apache-2.0

use super::libjpeg_turbo::{self, TurboJpegDecoder};

pub(crate) fn libjpeg_turbo_available() -> bool {
    libjpeg_turbo::is_available()
}

pub(crate) fn libjpeg_turbo_decode_batch(
    decoder: &mut TurboJpegDecoder,
    bytes: &[u8],
    batch_size: usize,
) {
    for _ in 0..batch_size {
        let out = decoder.decode_rgb(bytes).expect("libjpeg-turbo decode");
        std::hint::black_box(out);
    }
}
