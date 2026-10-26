// SPDX-License-Identifier: MIT OR Apache-2.0

use super::JpegBatchInput;
use crate::{Decoder, DecoderContext, Downscale, JpegError, JpegView, PixelFormat, SofKind};

#[derive(Clone, Copy)]
pub(super) struct Admission {
    pub(super) dimensions: (u32, u32),
    pub(super) image_len: usize,
    pub(super) worker_bytes: usize,
}

pub(super) fn admit(
    input: JpegBatchInput<'_>,
    ctx: &mut DecoderContext,
) -> Result<Admission, JpegError> {
    let view = JpegView::parse_with_options(input.bytes(), input.options())?;
    let info = view.info();
    if info.bit_depth != 8 {
        return Err(JpegError::UnsupportedBitDepth {
            depth: info.bit_depth,
        });
    }
    let dimensions = info.dimensions;
    let image_len = (dimensions.0 as usize)
        .saturating_mul(dimensions.1 as usize)
        .saturating_mul(3);
    let cap = crate::decoder::DEFAULT_MAX_DECODE_BYTES;
    if image_len > cap {
        return Err(JpegError::MemoryCapExceeded {
            requested: image_len,
            cap,
        });
    }
    // Every component sample in lossless coding, or block's initial DC value
    // in DCT coding, needs at least one Huffman bit. This deliberately weak
    // lower bound rejects impossible headers without guessing a compression
    // ratio or allocating storage proportional to the claimed geometry.
    let unit = if info.sof_kind == SofKind::Lossless {
        1
    } else {
        8
    };
    let bits: usize = info
        .sampling
        .components()
        .iter()
        .map(|&(h, v)| {
            let width =
                (dimensions.0 as usize * usize::from(h)).div_ceil(usize::from(info.sampling.max_h));
            let height =
                (dimensions.1 as usize * usize::from(v)).div_ceil(usize::from(info.sampling.max_v));
            width.div_ceil(unit) * height.div_ceil(unit)
        })
        .sum();
    let header = view.parsed_header();
    let entropy_bytes = if header.progressive_scans.is_empty() {
        input
            .bytes()
            .len()
            .saturating_sub(header.sos_offset.unwrap_or(input.bytes().len()))
    } else {
        header
            .progressive_scans
            .iter()
            .map(|scan| scan.terminal_offset.saturating_sub(scan.entropy_offset))
            .sum()
    };
    let minimum = bits.div_ceil(8);
    if entropy_bytes < minimum {
        return Err(JpegError::Truncated {
            offset: input.bytes().len(),
            expected: minimum - entropy_bytes,
        });
    }
    let plan = Decoder::with_view_in_context_with_external_live(view, ctx, 0, |decoder| {
        crate::decoder::planned_jpeg_tile_decode_for_decoder(
            decoder,
            PixelFormat::Rgb8,
            None,
            Downscale::None,
        )
    })?;
    Ok(Admission {
        dimensions,
        image_len,
        worker_bytes: plan.worker_live_bytes,
    })
}
