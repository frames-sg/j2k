// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{error::Error, sync::Arc};

use j2k::{encode_j2k_lossless, J2kBlockCodingMode, J2kLosslessEncodeOptions, J2kLosslessSamples};

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;

pub(super) fn generated_rgb8(seed: u8) -> Result<Arc<[u8]>, Box<dyn Error>> {
    let pixel_count = u8::try_from(WIDTH * HEIGHT)?;
    let pixels = (0..pixel_count)
        .flat_map(|position| {
            [
                position.wrapping_add(seed),
                position.wrapping_mul(3).wrapping_add(seed),
                position.wrapping_mul(7).wrapping_add(seed),
            ]
        })
        .collect::<Vec<_>>();
    let samples = J2kLosslessSamples::new(&pixels, WIDTH, HEIGHT, 3, 8, false)?;
    let encoded = encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions::default()
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput),
    )?;
    Ok(Arc::from(encoded.codestream))
}
