// SPDX-License-Identifier: MIT OR Apache-2.0

//! The streamed full-image route for images of several tiles: workers
//! decode one tile each at a time, as a region decode of that tile into a
//! tile-sized buffer, and copy it into the output.
//!
//! A region decode of one tile decodes that tile alone, and tiles do not
//! share samples, so each output sample comes from exactly one tile decode
//! whichever worker runs it and whenever it finishes. A worker holds one
//! tile's decode buffers and its output rows, so the decode's memory grows
//! with the workers and the tile size, never with the image.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use j2k_core::{PixelFormat, Rect};
use j2k_native::CpuDecodeParallelism;

use crate::backend::Image;
use crate::decode::decode_image_region_into_with_native_context;
use crate::J2kError;

/// Decode the tiles at `tiles`, output rectangles that cover `image` once,
/// on `workers` threads into `out`: the calling thread and `workers - 1`
/// scoped ones. Each tile decodes serially on its worker's thread.
///
/// The error is the one of the first tile, in raster order, that failed:
/// tiles are claimed in that order and every claimed tile finishes, so the
/// first failing tile is always among them.
pub(crate) fn decode_tiles_into(
    image: &Image<'_>,
    tiles: &[Rect],
    workers: usize,
    (out, stride, fmt): (&mut [u8], usize, PixelFormat),
) -> Result<(), J2kError> {
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let first_error: Mutex<Option<(usize, J2kError)>> = Mutex::new(None);
    let out = Mutex::new(out);
    let work = || {
        let mut native_context = j2k_native::DecoderContext::default();
        native_context.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
        let mut buffer = Vec::new();
        while !failed.load(Ordering::Relaxed) {
            let index = next.fetch_add(1, Ordering::Relaxed);
            let Some(&tile) = tiles.get(index) else {
                break;
            };
            let decoded = decode_tile(image, &mut native_context, &mut buffer, tile, fmt).and_then(
                |tile_stride| {
                    let mut out = out
                        .lock()
                        .map_err(|_| J2kError::internal_backend("tile output lock poisoned"))?;
                    copy_tile(
                        &buffer[..tile_stride * tile.h as usize],
                        tile_stride,
                        &mut out,
                        stride,
                        tile,
                        fmt,
                    );
                    Ok(())
                },
            );
            if let Err(error) = decoded {
                failed.store(true, Ordering::Relaxed);
                if let Ok(mut first) = first_error.lock() {
                    if first.as_ref().is_none_or(|(at, _)| index < *at) {
                        *first = Some((index, error));
                    }
                }
            }
        }
    };
    std::thread::scope(|scope| {
        for _ in 1..workers {
            scope.spawn(work);
        }
        work();
    });
    match first_error.into_inner() {
        Ok(None) => Ok(()),
        Ok(Some((_, error))) => Err(error),
        Err(_) => Err(J2kError::internal_backend("tile error lock poisoned")),
    }
}

/// Region-decode `tile` into the start of `buffer`, grown to fit it, and
/// return its stride. The decode writes every byte it covers, so what an
/// earlier tile left there is overwritten.
fn decode_tile<'a>(
    image: &Image<'a>,
    native_context: &mut j2k_native::DecoderContext<'a>,
    buffer: &mut Vec<u8>,
    tile: Rect,
    fmt: PixelFormat,
) -> Result<usize, J2kError> {
    let tile_stride = tile.w as usize * fmt.bytes_per_pixel();
    let len = tile_stride * tile.h as usize;
    if buffer.len() < len {
        buffer
            .try_reserve_exact(len - buffer.len())
            .map_err(|_| J2kError::internal_backend("tile output buffer allocation failed"))?;
        buffer.resize(len, 0);
    }
    decode_image_region_into_with_native_context(
        image,
        native_context,
        &mut buffer[..len],
        tile_stride,
        fmt,
        tile,
    )?;
    Ok(tile_stride)
}

/// Copy a decoded tile's rows from `buffer` into `out` at the tile.
fn copy_tile(
    buffer: &[u8],
    tile_stride: usize,
    out: &mut [u8],
    stride: usize,
    tile: Rect,
    fmt: PixelFormat,
) {
    let x = tile.x as usize * fmt.bytes_per_pixel();
    for (row, source) in buffer.chunks_exact(tile_stride).enumerate() {
        let start = (tile.y as usize + row) * stride + x;
        out[start..start + tile_stride].copy_from_slice(source);
    }
}
