// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::similar_names)]

#[cfg(feature = "cuda-runtime")]
use j2k_core::PixelFormat;
#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::{
    CudaDeviceBuffer, CudaJpegBaselineEncodeFormat, CudaJpegBaselineEncodeHuffmanTable,
    CudaJpegBaselineEncodeParams, CudaJpegBaselineEntropyEncodeBatchJob,
    CudaJpegBaselineEntropyEncodeJob,
};
#[cfg(feature = "cuda-runtime")]
use j2k_jpeg::adapter::{
    assemble_jpeg_baseline_frame, baseline_encode_tables, jpeg_baseline_entropy_capacity_bytes,
    validate_jpeg_baseline_dimensions, JpegBaselineHuffmanTable, JpegBaselineSampling,
};
use j2k_jpeg::{EncodedJpeg, JpegEncodeOptions};
#[cfg(feature = "cuda-runtime")]
use j2k_jpeg::{JpegBackend, JpegEncodeError, JpegSubsampling};

#[cfg(feature = "cuda-runtime")]
use crate::runtime::cuda_error;

#[cfg(feature = "cuda-runtime")]
#[derive(Debug, Clone, Copy)]
/// CUDA buffer and layout metadata for one baseline JPEG encode tile.
pub struct JpegBaselineCudaEncodeTile<'a> {
    /// Source CUDA buffer containing RGB8 or Gray8 pixels.
    pub buffer: &'a CudaDeviceBuffer,
    /// Byte offset of the first source pixel in `buffer`.
    pub byte_offset: usize,
    /// Width of the valid input region in pixels.
    pub width: u32,
    /// Height of the valid input region in pixels.
    pub height: u32,
    /// Number of bytes between consecutive input rows.
    pub pitch_bytes: usize,
    /// Encoded frame width in pixels.
    pub output_width: u32,
    /// Encoded frame height in pixels.
    pub output_height: u32,
    /// Pixel format of the source buffer.
    pub format: PixelFormat,
}

#[cfg(not(feature = "cuda-runtime"))]
#[derive(Debug, Clone, Copy)]
/// Placeholder encode tile type for builds without `cuda-runtime`.
pub struct JpegBaselineCudaEncodeTile<'a> {
    _private: core::marker::PhantomData<&'a ()>,
}

#[cfg(feature = "cuda-runtime")]
/// Encode one CUDA-resident tile as a baseline JPEG frame.
pub fn encode_jpeg_baseline_from_cuda_buffer(
    tile: JpegBaselineCudaEncodeTile<'_>,
    options: JpegEncodeOptions,
    session: &mut crate::CudaSession,
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
    let params = encode_params(tile, options, sampling, entropy_capacity, 0)?;
    let _ = session;
    let context = tile.buffer.context();
    let entropy = context
        .encode_jpeg_baseline_entropy(&CudaJpegBaselineEntropyEncodeJob {
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
        })
        .map_err(cuda_error)?;
    assemble_jpeg_baseline_frame(
        &entropy,
        tile.output_width,
        tile.output_height,
        &tables,
        options,
        JpegBackend::Cuda,
    )
    .map_err(Into::into)
}

#[cfg(feature = "cuda-runtime")]
/// Encode multiple CUDA-resident tiles as baseline JPEG frames.
///
/// Consecutive tiles that share a source CUDA buffer are submitted through a
/// single entropy-kernel batch. The returned frames preserve input order.
pub fn encode_jpeg_baseline_batch_from_cuda_buffers(
    tiles: &[JpegBaselineCudaEncodeTile<'_>],
    options: JpegEncodeOptions,
    session: &mut crate::CudaSession,
) -> Result<Vec<EncodedJpeg>, crate::Error> {
    if tiles.is_empty() {
        return Ok(Vec::new());
    }
    if tiles.len() == 1 {
        return encode_jpeg_baseline_from_cuda_buffer(tiles[0], options, session)
            .map(|encoded| vec![encoded]);
    }

    let tables = baseline_encode_tables(options)?;
    let sampling = tables.sampling;
    let mut encoded = Vec::with_capacity(tiles.len());
    let mut start = 0usize;
    while start < tiles.len() {
        validate_tile(tiles[start], options)?;
        let buffer_ptr = tiles[start].buffer.device_ptr();
        let mut end = start + 1;
        while end < tiles.len() && tiles[end].buffer.device_ptr() == buffer_ptr {
            validate_tile(tiles[end], options)?;
            end += 1;
        }

        if end - start == 1 {
            encoded.push(encode_jpeg_baseline_from_cuda_buffer(
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
            let param = encode_params(
                *tile,
                options,
                sampling,
                entropy_capacity,
                total_entropy_capacity,
            )?;
            total_entropy_capacity = total_entropy_capacity
                .checked_add(entropy_capacity)
                .ok_or_else(|| {
                    cuda_request_error("JPEG Baseline CUDA batch entropy capacity overflow")
                })?;
            params.push(param);
        }
        let entropy_chunks = tiles[start]
            .buffer
            .context()
            .encode_jpeg_baseline_entropy_batch(&CudaJpegBaselineEntropyEncodeBatchJob {
                input: tiles[start].buffer,
                params,
                q_luma: tables.q_luma,
                q_chroma: tables.q_chroma,
                huff_dc_luma: compute_huffman_table(&tables.huff_dc_luma),
                huff_ac_luma: compute_huffman_table(&tables.huff_ac_luma),
                huff_dc_chroma: compute_huffman_table(&tables.huff_dc_chroma),
                huff_ac_chroma: compute_huffman_table(&tables.huff_ac_chroma),
                entropy_capacity: total_entropy_capacity,
            })
            .map_err(cuda_error)?;
        for (tile, entropy) in tiles[start..end].iter().zip(entropy_chunks.iter()) {
            encoded.push(assemble_jpeg_baseline_frame(
                entropy,
                tile.output_width,
                tile.output_height,
                &tables,
                options,
                JpegBackend::Cuda,
            )?);
        }
        start = end;
    }
    Ok(encoded)
}

#[cfg(not(feature = "cuda-runtime"))]
/// Return `Error::CudaUnavailable` for batch CUDA encode requests without `cuda-runtime`.
pub fn encode_jpeg_baseline_batch_from_cuda_buffers(
    tiles: &[JpegBaselineCudaEncodeTile<'_>],
    options: JpegEncodeOptions,
    session: &mut crate::CudaSession,
) -> Result<Vec<EncodedJpeg>, crate::Error> {
    let _ = (tiles, options, session);
    Err(crate::Error::CudaUnavailable)
}

#[cfg(not(feature = "cuda-runtime"))]
/// Return `Error::CudaUnavailable` for CUDA encode requests without `cuda-runtime`.
pub fn encode_jpeg_baseline_from_cuda_buffer(
    tile: JpegBaselineCudaEncodeTile<'_>,
    options: JpegEncodeOptions,
    session: &mut crate::CudaSession,
) -> Result<EncodedJpeg, crate::Error> {
    let _ = (tile, options, session);
    Err(crate::Error::CudaUnavailable)
}

#[cfg(feature = "cuda-runtime")]
fn validate_tile(
    tile: JpegBaselineCudaEncodeTile<'_>,
    options: JpegEncodeOptions,
) -> Result<(), crate::Error> {
    match options.backend {
        JpegBackend::Auto | JpegBackend::Cuda => {}
        JpegBackend::Cpu => {
            return Err(crate::Error::UnsupportedCudaRequest {
                reason: "JPEG Baseline CUDA encode does not accept Cpu backend",
            });
        }
        JpegBackend::Metal => {
            return Err(crate::Error::UnsupportedCudaRequest {
                reason: "JPEG Baseline CUDA encode does not accept Metal backend",
            });
        }
    }
    if options.restart_interval == Some(0) {
        return Err(JpegEncodeError::InvalidRestartInterval.into());
    }
    validate_jpeg_baseline_dimensions(tile.output_width, tile.output_height)?;
    if tile.width == 0 || tile.height == 0 {
        return Err(JpegEncodeError::EmptyDimensions.into());
    }
    if tile.width > tile.output_width || tile.height > tile.output_height {
        return Err(crate::Error::UnsupportedCudaRequest {
            reason: "JPEG Baseline CUDA encode input cannot exceed output dimensions",
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
            return Err(crate::Error::UnsupportedCudaRequest {
                reason: "JPEG Baseline CUDA encode supports only Gray8 and Rgb8 input buffers",
            });
        }
    };

    let row_bytes = (tile.width as usize)
        .checked_mul(bytes_per_pixel)
        .ok_or_else(|| cuda_request_error("JPEG Baseline CUDA encode row byte count overflow"))?;
    if tile.pitch_bytes < row_bytes {
        return Err(crate::Error::UnsupportedCudaRequest {
            reason: "JPEG Baseline CUDA encode pitch is shorter than one row",
        });
    }
    let last_row = (tile.height as usize)
        .checked_sub(1)
        .and_then(|row| row.checked_mul(tile.pitch_bytes))
        .ok_or_else(|| cuda_request_error("JPEG Baseline CUDA encode input range overflow"))?;
    let required_end = tile
        .byte_offset
        .checked_add(last_row)
        .and_then(|offset| offset.checked_add(row_bytes))
        .ok_or_else(|| cuda_request_error("JPEG Baseline CUDA encode input range overflow"))?;
    if required_end > tile.buffer.byte_len() {
        return Err(crate::Error::UnsupportedCudaRequest {
            reason: "JPEG Baseline CUDA encode input range exceeds buffer length",
        });
    }

    Ok(())
}

#[cfg(feature = "cuda-runtime")]
fn encode_params(
    tile: JpegBaselineCudaEncodeTile<'_>,
    options: JpegEncodeOptions,
    sampling: JpegBaselineSampling,
    entropy_capacity: usize,
    entropy_offset: usize,
) -> Result<CudaJpegBaselineEncodeParams, crate::Error> {
    let mcu_width = u32::from(sampling.max_h) * 8;
    let mcu_height = u32::from(sampling.max_v) * 8;
    let mcus_per_row = tile.output_width.div_ceil(mcu_width);
    let mcu_rows = tile.output_height.div_ceil(mcu_height);
    let pitch_bytes =
        u32::try_from(tile.pitch_bytes).map_err(|_| crate::Error::UnsupportedCudaRequest {
            reason: "JPEG Baseline CUDA encode pitch exceeds CUDA kernel limits",
        })?;
    let input_offset_bytes =
        u32::try_from(tile.byte_offset).map_err(|_| crate::Error::UnsupportedCudaRequest {
            reason: "JPEG Baseline CUDA encode input offset exceeds CUDA kernel limits",
        })?;
    let entropy_offset_bytes =
        u32::try_from(entropy_offset).map_err(|_| crate::Error::UnsupportedCudaRequest {
            reason: "JPEG Baseline CUDA encode entropy offset exceeds CUDA kernel limits",
        })?;
    let format = match tile.format {
        PixelFormat::Gray8 => CudaJpegBaselineEncodeFormat::Gray8.abi(),
        PixelFormat::Rgb8 => CudaJpegBaselineEncodeFormat::Rgb8.abi(),
        _ => {
            return Err(crate::Error::UnsupportedCudaRequest {
                reason: "JPEG Baseline CUDA encode supports only Gray8 and Rgb8 input buffers",
            });
        }
    };
    Ok(CudaJpegBaselineEncodeParams {
        input_offset_bytes,
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
        entropy_offset_bytes,
        entropy_capacity: u32::try_from(entropy_capacity).map_err(|_| {
            crate::Error::UnsupportedCudaRequest {
                reason: "JPEG Baseline CUDA encode entropy capacity exceeds CUDA kernel limits",
            }
        })?,
    })
}

#[cfg(feature = "cuda-runtime")]
fn entropy_capacity_bytes(
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    restart_interval: Option<u16>,
) -> Result<usize, crate::Error> {
    let capacity = jpeg_baseline_entropy_capacity_bytes(width, height, sampling, restart_interval)?;
    if capacity > u32::MAX as usize {
        return Err(crate::Error::UnsupportedCudaRequest {
            reason: "JPEG Baseline CUDA encode entropy capacity exceeds CUDA kernel limits",
        });
    }
    Ok(capacity)
}

#[cfg(feature = "cuda-runtime")]
fn compute_huffman_table(source: &JpegBaselineHuffmanTable) -> CudaJpegBaselineEncodeHuffmanTable {
    CudaJpegBaselineEncodeHuffmanTable {
        codes: source.codes,
        lens: source.lens,
    }
}

#[cfg(feature = "cuda-runtime")]
fn cuda_request_error(reason: &'static str) -> crate::Error {
    crate::Error::UnsupportedCudaRequest { reason }
}
