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

use super::dct_entropy::{encode_dct_scan_optimized, DctComponent, DctEntropy, DctScan};
use super::{
    encode_jpeg_baseline_cpu, EncodedJpeg, JpegBackend, JpegEncodeError, JpegEncodeOptions,
    JpegSamples,
};
use crate::adapter::{baseline_encode_tables, JpegBaselineSampling};
use crate::encoded_frame::{FrameComponent, FrameKind, FrameWriter, ScanComponent, ScanParameters};
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
    assemble_dct_frame(
        FrameKind::Baseline,
        8,
        (image.width, image.height),
        tables.sampling,
        &quant,
        options.restart_interval,
        &entropy,
    )
}

/// Assemble a sequential DCT frame in the baseline encoder's segment order:
/// DQT, DRI, SOF, then one DHT per table, SOS, and the entropy-coded data.
pub(super) fn assemble_dct_frame(
    kind: FrameKind,
    precision: u8,
    dimensions: (u32, u32),
    sampling: JpegBaselineSampling,
    quant: &[[u16; 64]; 2],
    restart_interval: Option<u16>,
    entropy: &DctEntropy,
) -> Result<EncodedJpeg, JpegEncodeError> {
    let count = usize::from(sampling.components);
    let mut frame = FrameWriter::new(entropy.bytes.len())?;
    frame.dqt(0, &quant[0])?;
    if count == 3 {
        frame.dqt(1, &quant[1])?;
    }
    if let Some(interval) = restart_interval {
        frame.dri(interval)?;
    }
    let mut frame_components = [FrameComponent {
        id: 0,
        h: 1,
        v: 1,
        quant_table: 0,
    }; 3];
    let mut scan_components = [ScanComponent {
        id: 0,
        dc_table: 0,
        ac_table: 0,
    }; 3];
    for (index, (frame_component, scan_component)) in frame_components
        .iter_mut()
        .zip(&mut scan_components)
        .enumerate()
        .take(count)
    {
        let id = u8::try_from(index + 1).unwrap_or(u8::MAX);
        let table = u8::from(index != 0);
        *frame_component = FrameComponent {
            id,
            h: sampling.h[index],
            v: sampling.v[index],
            quant_table: table,
        };
        *scan_component = ScanComponent {
            id,
            dc_table: table,
            ac_table: table,
        };
    }
    frame.sof(kind, precision, dimensions, &frame_components[..count])?;
    for (table_id, (dc, ac)) in (0_u8..).zip(entropy.tables.dc.iter().zip(&entropy.tables.ac)) {
        if let (Some(dc), Some(ac)) = (dc, ac) {
            frame.dht(0, table_id, dc)?;
            frame.dht(1, table_id, ac)?;
        }
    }
    frame.sos(
        &scan_components[..count],
        ScanParameters {
            ss: 0,
            se: 63,
            ah: 0,
            al: 0,
        },
    )?;
    frame.finish(&entropy.bytes, JpegBackend::Cpu)
}
