// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_jpeg::ColorSpace;

pub(crate) const FULL_FRAME_MAX_OUTPUT_BYTES: usize = 512 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum DecodeMode {
    Gray,
    Rgb,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum CorpusInputClass {
    BoundedFullFrame,
    VeryLarge,
}

pub(crate) fn color_space_mode(color_space: ColorSpace) -> Option<DecodeMode> {
    match color_space {
        ColorSpace::Grayscale => Some(DecodeMode::Gray),
        ColorSpace::YCbCr | ColorSpace::Rgb => Some(DecodeMode::Rgb),
        ColorSpace::Cmyk | ColorSpace::Ycck => None,
    }
}

pub(crate) fn classify_corpus_input(dimensions: (u32, u32), mode: DecodeMode) -> CorpusInputClass {
    match full_frame_output_len(dimensions, mode) {
        Some(bytes) if bytes <= FULL_FRAME_MAX_OUTPUT_BYTES => CorpusInputClass::BoundedFullFrame,
        _ => CorpusInputClass::VeryLarge,
    }
}

fn full_frame_output_len(dimensions: (u32, u32), mode: DecodeMode) -> Option<usize> {
    let bpp = match mode {
        DecodeMode::Gray => 1usize,
        DecodeMode::Rgb => 3usize,
    };
    usize::try_from(dimensions.0)
        .ok()
        .zip(usize::try_from(dimensions.1).ok())
        .and_then(|(width, height)| width.checked_mul(height))
        .and_then(|pixels| pixels.checked_mul(bpp))
}
