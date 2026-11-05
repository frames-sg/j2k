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
//!
//! The workers share one native decode budget: each tile decode runs beside
//! the shares of the others and its own tile buffer, so the decodes in
//! flight together stay within it. A tile its share cannot hold decodes
//! again once the others finish, alone with the whole budget, as a serial
//! decode of it would.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use j2k_core::{PixelFormat, Rect};
use j2k_native::CpuDecodeParallelism;

use crate::backend::Image;
use crate::decode::{
    decode_image_region_into_with_native_context, decode_image_region_rows_into_beside,
    region_rows_write,
};
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
    let workers = workers.max(1);
    let shares = TileShares::new(image, workers)?;
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let first_error: Mutex<Option<(usize, J2kError)>> = Mutex::new(None);
    let refused: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    let out = Mutex::new(out);
    let record = |index: usize, error: J2kError| {
        failed.store(true, Ordering::Relaxed);
        if let Ok(mut first) = first_error.lock() {
            if first.as_ref().is_none_or(|(at, _)| index < *at) {
                *first = Some((index, error));
            }
        }
    };
    let work = || {
        let mut native_context = j2k_native::DecoderContext::default();
        native_context.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
        let mut buffer = Vec::new();
        while !failed.load(Ordering::Relaxed) {
            let index = next.fetch_add(1, Ordering::Relaxed);
            let Some(&tile) = tiles.get(index) else {
                break;
            };
            let baseline = shares.beside_share(tile, fmt);
            match decode_and_copy(
                image,
                &mut native_context,
                &mut buffer,
                tile,
                baseline,
                (&out, stride, fmt),
            ) {
                Ok(()) => {}
                // Its share could not hold the tile: decode it alone later.
                Err(J2kError::ResourceLimit { .. }) if workers > 1 => {
                    native_context = j2k_native::DecoderContext::default();
                    native_context.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
                    #[cfg(test)]
                    REFUSED_TILES.fetch_add(1, Ordering::Relaxed);
                    match refused.lock() {
                        Ok(mut refused) => refused.push(index),
                        Err(_) => record(
                            index,
                            J2kError::internal_backend("refused tile list poisoned"),
                        ),
                    }
                }
                Err(error) => record(index, error),
            }
        }
    };
    std::thread::scope(|scope| {
        for _ in 1..workers {
            // A thread the system refuses only leaves fewer workers: the
            // calling thread claims every tile the others do not.
            if std::thread::Builder::new()
                .spawn_scoped(scope, work)
                .is_err()
            {
                break;
            }
        }
        work();
    });
    let mut refused = refused
        .into_inner()
        .map_err(|_| J2kError::internal_backend("refused tile list poisoned"))?;
    if !failed.load(Ordering::Relaxed) && !refused.is_empty() {
        refused.sort_unstable();
        let mut native_context = j2k_native::DecoderContext::default();
        native_context.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
        let mut buffer = Vec::new();
        for index in refused {
            let tile = tiles[index];
            let alone = shares.alone(tile, fmt);
            if let Err(error) = decode_and_copy(
                image,
                &mut native_context,
                &mut buffer,
                tile,
                alone,
                (&out, stride, fmt),
            ) {
                record(index, error);
                break;
            }
        }
    }
    match first_error.into_inner() {
        Ok(None) => Ok(()),
        Ok(Some((_, error))) => Err(error),
        Err(_) => Err(J2kError::internal_backend("tile error lock poisoned")),
    }
}

/// How the workers of one decode share the native decode budget: each tile
/// decode is charged the other workers' shares and its own tile buffer
/// beside its decode.
struct TileShares {
    /// The budget the workers other than one hold together.
    others: usize,
}

/// Test-only count of tiles refused by their share, in every test.
#[cfg(test)]
pub(crate) static REFUSED_TILES: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
std::thread_local! {
    /// Test-only share of each worker on this thread's decodes, in bytes.
    pub(crate) static TILE_SHARE_OVERRIDE: core::cell::Cell<Option<usize>> =
        const { core::cell::Cell::new(None) };
}

impl TileShares {
    fn new(image: &Image<'_>, workers: usize) -> Result<Self, J2kError> {
        let retained = image
            .retained_allocation_bytes()
            .map_err(J2kError::from_native_decode_error)?;
        let available = j2k_native::DEFAULT_MAX_DECODE_BYTES.saturating_sub(retained);
        #[cfg(test)]
        let share = TILE_SHARE_OVERRIDE
            .with(core::cell::Cell::get)
            .map_or(available / workers, |share| share.min(available));
        #[cfg(not(test))]
        let share = available / workers;
        Ok(Self {
            others: available - share,
        })
    }

    /// The baseline of a tile decode beside the other workers.
    fn beside_share(&self, tile: Rect, fmt: PixelFormat) -> usize {
        self.others.saturating_add(tile_buffer_bytes(tile, fmt))
    }

    /// The baseline of a tile decode running alone.
    fn alone(&self, tile: Rect, fmt: PixelFormat) -> usize {
        let _ = self;
        tile_buffer_bytes(tile, fmt)
    }
}

fn tile_buffer_bytes(tile: Rect, fmt: PixelFormat) -> usize {
    (tile.w as usize)
        .saturating_mul(tile.h as usize)
        .saturating_mul(fmt.bytes_per_pixel())
}

/// Decode `tile` beside `baseline` bytes into `buffer` and copy it into the
/// output.
fn decode_and_copy<'a>(
    image: &Image<'a>,
    native_context: &mut j2k_native::DecoderContext<'a>,
    buffer: &mut Vec<u8>,
    tile: Rect,
    baseline: usize,
    (out, stride, fmt): (&Mutex<&mut [u8]>, usize, PixelFormat),
) -> Result<(), J2kError> {
    let tile_stride = decode_tile(image, native_context, buffer, tile, fmt, baseline)?;
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
    baseline: usize,
) -> Result<usize, J2kError> {
    let tile_stride = tile.w as usize * fmt.bytes_per_pixel();
    let len = tile_stride * tile.h as usize;
    if buffer.len() < len {
        buffer
            .try_reserve_exact(len - buffer.len())
            .map_err(|_| J2kError::internal_backend("tile output buffer allocation failed"))?;
        buffer.resize(len, 0);
    }
    if region_rows_write(image, fmt, tile) {
        decode_image_region_rows_into_beside(
            image,
            native_context,
            (&mut buffer[..len], tile_stride, fmt),
            tile,
            baseline,
        )?;
    } else {
        decode_image_region_into_with_native_context(
            image,
            native_context,
            &mut buffer[..len],
            tile_stride,
            fmt,
            tile,
        )?;
    }
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
