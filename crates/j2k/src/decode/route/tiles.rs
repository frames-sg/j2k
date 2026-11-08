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
use crate::decode::decode_image_region_rows_into_beside;
use crate::J2kError;

/// Decode the tiles at `tiles`, output rectangles that cover `image` once,
/// on `workers` threads into `out`: the calling thread and `workers - 1`
/// scoped ones. Each tile decodes serially on its worker's thread.
///
/// The error is the one of the first tile, in raster order, that failed:
/// tiles are claimed in that order and every claimed tile finishes, so the
/// first failing tile is always among them.
#[cfg(test)]
pub(crate) fn decode_tiles_into(
    image: &Image<'_>,
    tiles: &[Rect],
    workers: usize,
    (out, stride, fmt): (&mut [u8], usize, PixelFormat),
) -> Result<(), J2kError> {
    decode_tiles_into_with_control(
        image,
        tiles,
        workers,
        (out, stride, fmt),
        &j2k_native::DecodeOperationControl::default(),
    )
}

pub(crate) fn decode_tiles_into_with_control(
    image: &Image<'_>,
    tiles: &[Rect],
    workers: usize,
    (out, stride, fmt): (&mut [u8], usize, PixelFormat),
    control: &j2k_native::DecodeOperationControl,
) -> Result<(), J2kError> {
    let workers = workers.max(1);
    let shares = TileShares::new(image, workers, control)?;
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
            let baseline = shares.others;
            match native_context.with_operation_control(control.clone(), |context| {
                decode_and_copy(
                    image,
                    context,
                    &mut buffer,
                    tile,
                    baseline,
                    (&out, stride, fmt),
                )
            }) {
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
            if let Err(error) = native_context.with_operation_control(control.clone(), |context| {
                decode_and_copy(image, context, &mut buffer, tile, 0, (&out, stride, fmt))
            }) {
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
    fn new(
        image: &Image<'_>,
        workers: usize,
        control: &j2k_native::DecodeOperationControl,
    ) -> Result<Self, J2kError> {
        let retained = image
            .retained_allocation_bytes()
            .map_err(J2kError::from_native_decode_error)?;
        let retained = control
            .baseline(retained)
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
    let tile_stride = (tile.w as usize)
        .checked_mul(fmt.bytes_per_pixel())
        .ok_or_else(|| {
            J2kError::from_native_decode_error(j2k_native::ValidationError::ImageTooLarge.into())
        })?;
    let len = tile_stride.checked_mul(tile.h as usize).ok_or_else(|| {
        J2kError::from_native_decode_error(j2k_native::ValidationError::ImageTooLarge.into())
    })?;
    let retained = image
        .retained_allocation_bytes()
        .map_err(J2kError::from_native_decode_error)?;
    let control = native_context.operation_control().clone();
    let check_capacity = |capacity: usize| {
        let requested = control
            .baseline(baseline.saturating_add(retained).saturating_add(capacity))
            .map_err(J2kError::from_native_decode_error)?;
        let cap = j2k_native::DEFAULT_MAX_DECODE_BYTES;
        if requested > cap {
            Err(J2kError::from_native_decode_error(
                j2k_native::DecodeError::AllocationTooLarge {
                    what: "streamed tile output buffer",
                    requested,
                    cap,
                },
            ))
        } else {
            Ok(())
        }
    };
    if buffer.capacity() < len {
        // A reallocation may hold both buffers. Release the previous decode's
        // scratch before growing, so only its output remains beside the growth.
        check_capacity(buffer.capacity().saturating_add(len))?;
        *native_context = native_context.fresh_for_operation();
        buffer
            .try_reserve_exact(len - buffer.len())
            .map_err(|_| J2kError::internal_backend("tile output buffer allocation failed"))?;
    }
    check_capacity(buffer.capacity())?;
    buffer.resize(len, 0);
    // StreamedTiles already established row-output eligibility. A small gray
    // tile's standalone preference for native packing must not discard its
    // shared budget; this API also charges any native component-plane fallback.
    let baseline = baseline.saturating_add(buffer.capacity());
    decode_image_region_rows_into_beside(
        image,
        native_context,
        (&mut buffer[..len], tile_stride, fmt),
        tile,
        baseline,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::image_at_scale;
    use crate::decode::route::tests::codestream;
    use crate::DecodeSettings;
    use j2k_core::Downscale;
    use j2k_native::EncodeOptions;

    #[test]
    fn retained_tile_output_and_growth_must_fit_before_decode() {
        let bytes = codestream(
            1,
            &EncodeOptions {
                num_decomposition_levels: 2,
                tile_size: Some((64, 64)),
                ..EncodeOptions::default()
            },
        );
        let image = image_at_scale(&bytes, DecodeSettings::default(), Downscale::None).unwrap();
        let full = Rect {
            x: 0,
            y: 0,
            w: 64,
            h: 64,
        };
        let edge = Rect {
            x: 64,
            y: 64,
            w: 32,
            h: 16,
        };
        let fmt = PixelFormat::Gray16;
        let mut context = j2k_native::DecoderContext::default();
        context.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
        let mut buffer = Vec::new();
        let retained = image.retained_allocation_bytes().unwrap();
        let tight_baseline = j2k_native::DEFAULT_MAX_DECODE_BYTES - retained - 1024;
        assert!(matches!(
            decode_tile(&image, &mut context, &mut buffer, full, fmt, tight_baseline),
            Err(J2kError::ResourceLimit { .. })
        ));
        assert_eq!(
            buffer.capacity(),
            0,
            "refuse growth before allocating the tile buffer"
        );
        decode_tile(&image, &mut context, &mut buffer, full, fmt, 0).unwrap();
        let capacity = buffer.capacity();
        let edge_baseline = j2k_native::DEFAULT_MAX_DECODE_BYTES - retained - capacity + 1;
        assert!(matches!(
            decode_tile(&image, &mut context, &mut buffer, edge, fmt, edge_baseline),
            Err(J2kError::ResourceLimit { .. })
        ));
        assert_eq!(buffer.capacity(), capacity);
        decode_tile(&image, &mut context, &mut buffer, edge, fmt, 0).unwrap();
        let mut fresh = Vec::new();
        decode_tile(&image, &mut context, &mut fresh, edge, fmt, 0).unwrap();
        assert_eq!(&buffer[..fresh.len()], fresh);
    }
}
