// SPDX-License-Identifier: MIT OR Apache-2.0

//! Baseline JPEG encode with Huffman tables built from the image's own
//! statistics, as `cjpeg -optimize` writes it.
//!
//! The image is first encoded exactly as [`super::encode_jpeg_baseline`]
//! encodes it; its quantized coefficients are then re-coded with optimal
//! tables. Decoded pixels are therefore identical to the standard-table
//! encode; only the entropy-coded data and the DHT segments change. MCU
//! padding blocks outside the image are coded as libjpeg codes them, so the
//! entropy-coded data matches `jpegtran -optimize` byte for byte.

use alloc::vec::Vec;

use super::dct_entropy::{encode_dct_scan_optimized, DctComponent, DctScan};
use super::{
    encode_jpeg_baseline_cpu, EncodedJpeg, JpegBackend, JpegEncodeError, JpegEncodeOptions,
    JpegSamples,
};
use crate::adapter::baseline_encode_tables;
use crate::encoded_frame::{DctFrame, FrameKind, FrameWriter};
use crate::transcode::{extract_dct_blocks, DctExtractOptions};

/// Largest DC difference category of 8-bit DCT JPEG.
const MAX_DC_CATEGORY_8BIT: u8 = 11;

/// Encode grayscale or RGB samples as a baseline JPEG with optimized
/// Huffman tables.
///
/// Takes the same samples and options as [`super::encode_jpeg_baseline`]
/// and decodes to the same pixels, at the cost of a second entropy pass.
///
/// # Errors
///
/// Returns the errors [`super::encode_jpeg_baseline`] returns, and an
/// allocation error when the image's coefficients exceed the host
/// allocation cap. GPU backends are not available for this mode.
pub fn encode_jpeg_baseline_optimized(
    samples: JpegSamples<'_>,
    options: JpegEncodeOptions,
) -> Result<EncodedJpeg, JpegEncodeError> {
    if !matches!(options.backend, JpegBackend::Auto | JpegBackend::Cpu) {
        return Err(JpegEncodeError::UnsupportedBackend {
            backend: options.backend,
        });
    }
    let standard = encode_jpeg_baseline_cpu(samples, options)?;
    let image = extract_dct_blocks(&standard.data, DctExtractOptions::default()).map_err(|_| {
        JpegEncodeError::InternalInvariant {
            reason: "baseline encoder output failed coefficient extraction",
        }
    })?;
    drop(standard);
    let tables = baseline_encode_tables(options)?;
    let components: Vec<DctComponent<'_>> = image
        .components
        .iter()
        .zip(0_usize..)
        .map(|(component, index)| DctComponent {
            h: component.h_samp,
            v: component.v_samp,
            block_cols: component.block_cols,
            real_cols: component.width.div_ceil(8),
            real_rows: component.height.div_ceil(8),
            blocks: &component.quantized_blocks,
            table: usize::from(index != 0),
        })
        .collect();
    let entropy = encode_dct_scan_optimized(&DctScan {
        width: image.width,
        height: image.height,
        components: &components,
        restart_interval: options.restart_interval,
        max_dc_category: MAX_DC_CATEGORY_8BIT,
    })?;
    let quant = [tables.q_luma.map(u16::from), tables.q_chroma.map(u16::from)];
    FrameWriter::new(entropy.bytes.len())?.assemble_dct(
        DctFrame {
            kind: FrameKind::Baseline,
            precision: 8,
            dimensions: (image.width, image.height),
            sampling: tables.sampling,
            quant: &quant,
            huffman: core::array::from_fn(|index| {
                entropy.tables.dc[index]
                    .as_ref()
                    .zip(entropy.tables.ac[index].as_ref())
                    .map(|(dc, ac)| [dc.into(), ac.into()])
            }),
            restart_interval: options.restart_interval,
        },
        &entropy.bytes,
        JpegBackend::Cpu,
    )
}
