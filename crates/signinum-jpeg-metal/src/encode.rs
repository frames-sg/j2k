// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::similar_names)]

#[cfg(target_os = "macos")]
use metal::Buffer;
#[cfg(target_os = "macos")]
use signinum_core::PixelFormat;
#[cfg(target_os = "macos")]
use signinum_jpeg::adapter::{
    assemble_jpeg_baseline_frame, baseline_encode_tables, jpeg_baseline_entropy_capacity_bytes,
    validate_jpeg_baseline_dimensions, JpegBaselineHuffmanTable, JpegBaselineSampling,
};
use signinum_jpeg::{EncodedJpeg, JpegEncodeOptions};
#[cfg(target_os = "macos")]
use signinum_jpeg::{JpegBackend, JpegEncodeError, JpegSubsampling};

#[cfg(target_os = "macos")]
use crate::compute;

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy)]
pub struct JpegBaselineMetalEncodeTile<'a> {
    pub buffer: &'a Buffer,
    pub byte_offset: usize,
    pub width: u32,
    pub height: u32,
    pub pitch_bytes: usize,
    pub output_width: u32,
    pub output_height: u32,
    pub format: PixelFormat,
}

#[cfg(not(target_os = "macos"))]
#[derive(Debug, Clone, Copy)]
pub struct JpegBaselineMetalEncodeTile<'a> {
    _private: core::marker::PhantomData<&'a ()>,
}

#[cfg(target_os = "macos")]
pub fn encode_jpeg_baseline_from_metal_buffer(
    tile: JpegBaselineMetalEncodeTile<'_>,
    options: JpegEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<EncodedJpeg, crate::Error> {
    validate_tile(tile, options)?;
    let tables = baseline_encode_tables(options)?;
    let sampling = tables.sampling;

    let entropy_capacity = entropy_capacity_bytes(
        tile.output_width,
        tile.output_height,
        sampling,
        options.restart_interval,
    )?;
    let params = encode_params(tile, options, sampling, entropy_capacity)?;
    let job = compute::JpegBaselineEntropyEncodeJob {
        input: tile.buffer,
        input_offset: tile.byte_offset,
        params,
        q_luma: tables.q_luma,
        q_chroma: tables.q_chroma,
        huff_dc_luma: compute_huffman_table(&tables.huff_dc_luma),
        huff_ac_luma: compute_huffman_table(&tables.huff_ac_luma),
        huff_dc_chroma: compute_huffman_table(&tables.huff_dc_chroma),
        huff_ac_chroma: compute_huffman_table(&tables.huff_ac_chroma),
        entropy_capacity,
    };
    let entropy = compute::encode_jpeg_baseline_entropy_with_session(session, &job)?;

    assemble_jpeg_baseline_frame(
        &entropy,
        tile.output_width,
        tile.output_height,
        &tables,
        options,
        JpegBackend::Metal,
    )
    .map_err(Into::into)
}

#[cfg(target_os = "macos")]
pub fn encode_jpeg_baseline_batch_from_metal_buffers(
    tiles: &[JpegBaselineMetalEncodeTile<'_>],
    options: JpegEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<EncodedJpeg>, crate::Error> {
    if tiles.is_empty() {
        return Ok(Vec::new());
    }
    if tiles.len() == 1 {
        return encode_jpeg_baseline_from_metal_buffer(tiles[0], options, session)
            .map(|encoded| vec![encoded]);
    }

    let tables = baseline_encode_tables(options)?;
    let sampling = tables.sampling;

    let mut encoded = Vec::with_capacity(tiles.len());
    let mut start = 0usize;
    while start < tiles.len() {
        validate_tile(tiles[start], options)?;
        let buffer_address = tiles[start].buffer.gpu_address();
        let mut end = start + 1;
        while end < tiles.len() && tiles[end].buffer.gpu_address() == buffer_address {
            validate_tile(tiles[end], options)?;
            end += 1;
        }

        if end - start == 1 {
            encoded.push(encode_jpeg_baseline_from_metal_buffer(
                tiles[start],
                options,
                session,
            )?);
            start = end;
            continue;
        }

        let mut params = Vec::with_capacity(end - start);
        let mut total_entropy_capacity = 0usize;
        for tile in &tiles[start..end] {
            let entropy_capacity = entropy_capacity_bytes(
                tile.output_width,
                tile.output_height,
                sampling,
                options.restart_interval,
            )?;
            let mut param = encode_params(*tile, options, sampling, entropy_capacity)?;
            param.input_offset_bytes =
                u32::try_from(tile.byte_offset).map_err(|_| crate::Error::MetalKernel {
                    message: "JPEG Baseline Metal batch input offset exceeds u32".to_string(),
                })?;
            param.entropy_offset_bytes =
                u32::try_from(total_entropy_capacity).map_err(|_| crate::Error::MetalKernel {
                    message: "JPEG Baseline Metal batch entropy offset exceeds u32".to_string(),
                })?;
            total_entropy_capacity = total_entropy_capacity
                .checked_add(entropy_capacity)
                .ok_or_else(|| {
                    metal_kernel_error("JPEG Baseline Metal batch entropy capacity overflow")
                })?;
            params.push(param);
        }
        let entropy_chunks = compute::encode_jpeg_baseline_entropy_batch_with_session(
            session,
            &compute::JpegBaselineEntropyEncodeBatchJob {
                input: tiles[start].buffer,
                params,
                q_luma: tables.q_luma,
                q_chroma: tables.q_chroma,
                huff_dc_luma: compute_huffman_table(&tables.huff_dc_luma),
                huff_ac_luma: compute_huffman_table(&tables.huff_ac_luma),
                huff_dc_chroma: compute_huffman_table(&tables.huff_dc_chroma),
                huff_ac_chroma: compute_huffman_table(&tables.huff_ac_chroma),
                entropy_capacity: total_entropy_capacity,
            },
        )?;
        for (tile, entropy) in tiles[start..end].iter().zip(entropy_chunks.iter()) {
            encoded.push(assemble_jpeg_baseline_frame(
                entropy,
                tile.output_width,
                tile.output_height,
                &tables,
                options,
                JpegBackend::Metal,
            )?);
        }
        start = end;
    }
    Ok(encoded)
}

#[cfg(not(target_os = "macos"))]
pub fn encode_jpeg_baseline_batch_from_metal_buffers(
    tiles: &[JpegBaselineMetalEncodeTile<'_>],
    options: JpegEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<EncodedJpeg>, crate::Error> {
    let _ = (tiles, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
pub fn encode_jpeg_baseline_from_metal_buffer(
    tile: JpegBaselineMetalEncodeTile<'_>,
    options: JpegEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<EncodedJpeg, crate::Error> {
    let _ = (tile, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(target_os = "macos")]
fn validate_tile(
    tile: JpegBaselineMetalEncodeTile<'_>,
    options: JpegEncodeOptions,
) -> Result<(), crate::Error> {
    if options.backend == JpegBackend::Cpu {
        return Err(crate::Error::UnsupportedMetalRequest {
            reason: "JPEG Baseline Metal encode does not accept Cpu backend",
        });
    }
    if options.restart_interval == Some(0) {
        return Err(JpegEncodeError::InvalidRestartInterval.into());
    }
    validate_jpeg_baseline_dimensions(tile.output_width, tile.output_height)?;
    if tile.width == 0 || tile.height == 0 {
        return Err(JpegEncodeError::EmptyDimensions.into());
    }
    if tile.width > tile.output_width || tile.height > tile.output_height {
        return Err(crate::Error::UnsupportedMetalRequest {
            reason: "JPEG Baseline Metal encode input cannot exceed output dimensions",
        });
    }

    let bytes_per_pixel = match (tile.format, options.subsampling) {
        (PixelFormat::Gray8, JpegSubsampling::Gray) => 1usize,
        (
            PixelFormat::Rgb8,
            JpegSubsampling::Ybr444 | JpegSubsampling::Ybr422 | JpegSubsampling::Ybr420,
        ) => 3usize,
        (PixelFormat::Gray8 | PixelFormat::Rgb8, _) => {
            return Err(JpegEncodeError::IncompatibleSubsampling {
                subsampling: options.subsampling,
                samples: if tile.format == PixelFormat::Gray8 {
                    "Gray8"
                } else {
                    "Rgb8"
                },
            }
            .into());
        }
        _ => {
            return Err(crate::Error::UnsupportedMetalRequest {
                reason: "JPEG Baseline Metal encode supports only Gray8 and Rgb8 input buffers",
            });
        }
    };

    let row_bytes = (tile.width as usize)
        .checked_mul(bytes_per_pixel)
        .ok_or_else(|| metal_kernel_error("JPEG Baseline Metal encode row byte count overflow"))?;
    if tile.pitch_bytes < row_bytes {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "JPEG Baseline Metal encode pitch is shorter than one row: need {row_bytes}, got {}",
                tile.pitch_bytes
            ),
        });
    }
    let last_row = (tile.height as usize)
        .checked_sub(1)
        .and_then(|row| row.checked_mul(tile.pitch_bytes))
        .ok_or_else(|| metal_kernel_error("JPEG Baseline Metal encode input range overflow"))?;
    let required_end = tile
        .byte_offset
        .checked_add(last_row)
        .and_then(|offset| offset.checked_add(row_bytes))
        .ok_or_else(|| metal_kernel_error("JPEG Baseline Metal encode input range overflow"))?;
    let buffer_len =
        usize::try_from(tile.buffer.length()).map_err(|_| crate::Error::MetalKernel {
            message: "JPEG Baseline Metal encode buffer length exceeds usize".to_string(),
        })?;
    if required_end > buffer_len {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "JPEG Baseline Metal encode input range exceeds buffer length: need {required_end}, buffer has {buffer_len}"
            ),
        });
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn encode_params(
    tile: JpegBaselineMetalEncodeTile<'_>,
    options: JpegEncodeOptions,
    sampling: JpegBaselineSampling,
    entropy_capacity: usize,
) -> Result<compute::JpegBaselineEncodeParams, crate::Error> {
    let mcu_width = u32::from(sampling.max_h) * 8;
    let mcu_height = u32::from(sampling.max_v) * 8;
    let mcus_per_row = tile.output_width.div_ceil(mcu_width);
    let mcu_rows = tile.output_height.div_ceil(mcu_height);
    let pitch_bytes = u32::try_from(tile.pitch_bytes).map_err(|_| crate::Error::MetalKernel {
        message: "JPEG Baseline Metal encode pitch exceeds u32".to_string(),
    })?;
    let format = match tile.format {
        PixelFormat::Gray8 => compute::JPEG_BASELINE_ENCODE_FORMAT_GRAY8,
        PixelFormat::Rgb8 => compute::JPEG_BASELINE_ENCODE_FORMAT_RGB8,
        _ => {
            return Err(crate::Error::UnsupportedMetalRequest {
                reason: "JPEG Baseline Metal encode supports only Gray8 and Rgb8 input buffers",
            });
        }
    };
    Ok(compute::JpegBaselineEncodeParams {
        input_offset_bytes: 0,
        input_width: tile.width,
        input_height: tile.height,
        output_width: tile.output_width,
        output_height: tile.output_height,
        pitch_bytes,
        mcus_per_row,
        mcu_rows,
        restart_interval_mcus: u32::from(options.restart_interval.unwrap_or(0)),
        format,
        components: u32::from(sampling.components),
        max_h: u32::from(sampling.max_h),
        max_v: u32::from(sampling.max_v),
        h0: u32::from(sampling.h[0]),
        v0: u32::from(sampling.v[0]),
        h1: u32::from(sampling.h[1]),
        v1: u32::from(sampling.v[1]),
        h2: u32::from(sampling.h[2]),
        v2: u32::from(sampling.v[2]),
        entropy_offset_bytes: 0,
        entropy_capacity: u32::try_from(entropy_capacity).map_err(|_| {
            crate::Error::UnsupportedMetalRequest {
                reason: "JPEG Baseline Metal encode entropy capacity exceeds Metal kernel limits",
            }
        })?,
    })
}

#[cfg(target_os = "macos")]
fn entropy_capacity_bytes(
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    restart_interval: Option<u16>,
) -> Result<usize, crate::Error> {
    let capacity = jpeg_baseline_entropy_capacity_bytes(width, height, sampling, restart_interval)?;
    if capacity > u32::MAX as usize {
        return Err(crate::Error::UnsupportedMetalRequest {
            reason: "JPEG Baseline Metal encode entropy capacity exceeds Metal kernel limits",
        });
    }
    Ok(capacity)
}

#[cfg(target_os = "macos")]
fn compute_huffman_table(
    source: &JpegBaselineHuffmanTable,
) -> compute::JpegBaselineEncodeHuffmanTable {
    compute::JpegBaselineEncodeHuffmanTable {
        codes: source.codes,
        lens: source.lens,
    }
}

#[cfg(target_os = "macos")]
fn metal_kernel_error(message: &'static str) -> crate::Error {
    crate::Error::MetalKernel {
        message: message.to_string(),
    }
}
