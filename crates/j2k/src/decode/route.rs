// SPDX-License-Identifier: MIT OR Apache-2.0

//! Plan-time choice of how a full-image 8- or 16-bit decode runs.
//!
//! The planes route decodes every component into an image-sized plane of
//! `f32` samples and packs the output from them. Its working memory grows
//! with the image, and the native decoder refuses a decode whose planes and
//! largest tile's workspace pass its budget. The streamed route runs the
//! decode as a region decode of the whole image, which hands finished rows
//! to the output and keeps no planes (see
//! `Image::decode_region_rows_for_integer_output_with_context`); an image of
//! several tiles runs it on several tiles at once (see [`tiles`]). The
//! native decoder plans the planes route's charge and whether the region
//! decode streams from the codestream before anything is decoded (see
//! `Image::full_decode_planes_floor`), so the choice is made once, before
//! the decode. Only a stream chosen by preference whose own working memory
//! the budget refuses runs again, on the planes, and only when they fit (see
//! `planes_fit`): streaming by preference never fails a decode the planes
//! finish.

use j2k_core::{Downscale, PixelFormat, Rect};
use j2k_native::CpuDecodeParallelism;

use super::output::u16_rows_pack_floor;
use crate::backend::Image;

pub(crate) mod tiles;

/// How a full-image decode runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FullImageRoute {
    /// Component planes for the whole image.
    Planes,
    /// A region decode of the whole image that streams its rows.
    StreamedRegion,
    /// Streamed region decodes of one tile each, `workers` at once, each on
    /// its worker's thread alone.
    StreamedTiles { tiles: Vec<Rect>, workers: usize },
}

impl FullImageRoute {
    /// Whether the decode streams, holding no component planes.
    pub(crate) fn streams(&self) -> bool {
        !matches!(self, Self::Planes)
    }
}

/// What a full-image decode's component planes may charge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlanesBudget {
    /// The native decode budget; an image within it still streams where
    /// streaming is the better route (see [`streams_by_default`]).
    Default,
    /// These bytes and no preference: the planes route unless the planes
    /// would pass them. A test hook that forces either route.
    Fixed(usize),
}

/// The facts of a decode request the route depends on besides the image.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RouteRequest {
    pub(crate) fmt: PixelFormat,
    pub(crate) budget: PlanesBudget,
    pub(crate) parallelism: CpuDecodeParallelism,
    /// The image's components.
    pub(crate) components: u16,
    /// The bytes of the coded image, its codestream and container.
    pub(crate) coded_bytes: usize,
    /// The resolution the image is decoded at.
    pub(crate) scale: Downscale,
}

/// How a full-image decode of `image` runs for `request`: streamed when the
/// output has 8 bits per sample, or 16 bits whose rows carry the native
/// samples the planes route packs (see `u16_rows_pack_floor`), the region
/// decode streams, and either the planes (or their 16-bit pack) would pass
/// the budget or the streamed route is the better one for this image.
///
/// A planning error keeps the planes route, which meets the same error and
/// reports it.
pub(crate) fn full_image_route(image: &Image<'_>, request: RouteRequest) -> FullImageRoute {
    full_image_route_beside(
        image,
        request,
        0,
        &j2k_native::DecodeOperationControl::default(),
    )
}

/// Select the same route while admitting its metadata scan beside live owners.
pub(crate) fn full_image_route_beside(
    image: &Image<'_>,
    request: RouteRequest,
    retained_capacity: usize,
    control: &j2k_native::DecodeOperationControl,
) -> FullImageRoute {
    let Some(pack_floor) = pack_floor(image, request.fmt) else {
        return FullImageRoute::Planes;
    };
    let planes_budget = match request.budget {
        PlanesBudget::Default => j2k_native::DEFAULT_MAX_DECODE_BYTES,
        PlanesBudget::Fixed(bytes) => bytes,
    };
    // A budget of zero skips the planner's header-only shortcut, so the
    // floor is known whenever the region decode would stream; the shortcut
    // saves the tile parse when only the budget can choose streaming.
    let streams = if request.budget == PlanesBudget::Default && streams_by_default(image, request) {
        matches!(
            image.full_decode_planes_floor_with_operation_control(0, retained_capacity, control),
            Ok(Some(_))
        )
    } else {
        matches!(
            image.full_decode_planes_floor_with_operation_control(planes_budget, retained_capacity, control),
            Ok(Some(floor)) if floor.max(pack_floor) > planes_budget
        )
    };
    if streams {
        streamed_route(image, request)
    } else {
        FullImageRoute::Planes
    }
}

/// Whether streamed rows can write `image` in `fmt` (see [`pack_floor`]).
pub(crate) fn output_streams_rows(image: &Image<'_>, fmt: PixelFormat) -> bool {
    pack_floor(image, fmt).is_some()
}

/// What the 16-bit pack of `image` into `fmt` holds besides the planes (see
/// `u16_rows_pack_floor`), nothing for an 8-bit output, or `None` when the
/// streamed rows cannot write `fmt`.
fn pack_floor(image: &Image<'_>, fmt: PixelFormat) -> Option<usize> {
    match fmt {
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Gray8 => Some(0),
        PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16 => {
            u16_rows_pack_floor(image, fmt)
        }
        _ => None,
    }
}

/// Whether a full-image decode of `image` for `request` that streams by
/// preference may run on the planes instead when the budget refuses its
/// stream: at the default budget, when the planes and their 16-bit pack fit
/// it. The planes grow with an image's area and the streamed rows' line
/// buffers with its width, so a wide and shallow image can fit the one and
/// not the other: an RGB image 8,388,608 samples wide and a row or two deep
/// fits its planes, but not its reconstruction rows.
pub(super) fn planes_fit(
    image: &Image<'_>,
    request: RouteRequest,
    context: &j2k_native::DecoderContext<'_>,
) -> bool {
    let budget = j2k_native::DEFAULT_MAX_DECODE_BYTES;
    request.budget == PlanesBudget::Default
        && pack_floor(image, request.fmt).is_some_and(|pack_floor| {
            !matches!(
                image.full_decode_planes_floor_with_operation_control(
                    budget, context.retained_allocation_bytes().unwrap_or(usize::MAX), context.operation_control(),
                ),
                Ok(Some(floor)) if floor.max(pack_floor) > budget
            )
        })
}

/// Pool threads from which a classic tile of several components
/// reconstructs each component on its own pool reader (`parallel_synthesis`
/// in the native decoder's pool bands). From there, a full-resolution
/// untiled image decodes 3-6% faster on the planes.
const CLASSIC_SYNTHESIS_THREADS: usize = 12;

/// Samples from which a full-resolution classic image streams on 1 to 11
/// pool threads (a 512 x 512 image at 6.6 coded bits per pixel and 768 x 512
/// ones at 9.6 decode 5-10% faster streamed on 6 threads, 1% slower on one,
/// in a third of the memory); smaller images spend more of their time
/// setting up the pool bands, and on one thread the line IDWT's rows.
const MIN_STREAMED_SAMPLES: u64 = 1 << 18;

/// Samples below which a full-resolution classic image also streams from
/// [`CLASSIC_SYNTHESIS_THREADS`] pool threads: the 512 x 512 and 768 x 512
/// images decode 3-14% faster streamed on 18 threads, a 4096 x 1024 one
/// 6-7% slower.
const MAX_STREAMED_SAMPLES_ON_A_LARGE_POOL: u64 = 1 << 19;

/// Samples a tile needs to stream one after another with the pool behind
/// it (measured on 1024 x 1024 tiles only).
const MIN_STREAMED_TILE_SAMPLES: u64 = 1 << 20;

/// Samples a classic tile needs to stream one after another on one thread:
/// 4096 x 4096 and 2048 x 2048 RGB images in 256 x 256 tiles decode 1%
/// slower streamed, in a fiftieth to a two-hundredth of the memory. An HTJ2K
/// tile needs [`MIN_STREAMED_SAMPLES`] (512 x 512 tiles 0.93-0.99 of the
/// planes), the streamed IDWT's rows costing more on narrower tiles.
const MIN_STREAMED_TILE_SAMPLES_ON_ONE_THREAD: u64 = 1 << 16;

/// Output samples from which a reduced-resolution classic decode streams on
/// any pool (a 1024 x 512 output and larger: 6-53% faster on 18 threads),
/// and an HTJ2K one on two threads or more or on one at 1/4 and below. On 12
/// pool threads or more a smaller classic output gains nothing from
/// streaming (a 384 x 256 output of a 768 x 512 image 2-3% slower on 12
/// threads, 2% faster on 18), and smaller HTJ2K outputs are up to 59% slower
/// streamed on any pool.
const MIN_REDUCED_STREAMED_SAMPLES: u64 = 1 << 19;

/// Output samples from which a smaller reduced classic decode streams on 2
/// to 11 pool threads: 256 x 256 and 384 x 256 outputs decode 3-12% faster
/// streamed on 6 threads, a 192 x 128 one 1-3% slower.
const MIN_SMALL_REDUCED_STREAMED_SAMPLES: u64 = 1 << 16;

/// Output samples from which a smaller reduced classic decode streams on one
/// thread: 512 x 256 to 758 x 532 outputs of 4096 x 2048 to 6061 x 4252
/// images decode 3-11% faster streamed, in a thirtieth to a sixtieth of the
/// memory (the planes route holds the full-resolution coefficients), while
/// 256 x 256 and 384 x 256 outputs of small images are 0-6% slower.
const MIN_SMALL_REDUCED_STREAMED_SAMPLES_ON_ONE_THREAD: u64 = 1 << 17;

/// Samples from which an untiled one-component classic image streams at
/// full resolution on 2 to 11 pool threads: the smallest gray image
/// measured. With the pool bands decoding a band in parts when the rings
/// hold too few bands, 2896 x 2896 to 5792 x 5792 gray images decode 12-15%
/// faster streamed on 6 threads, in a fifth to a seventh of the memory
/// (5792 x 5792: 459 vs 68 MB).
const MIN_STREAMED_GRAY_SAMPLES: u64 = 2896 * 2896;

/// Samples from which a one-component classic image streams at full
/// resolution on one thread: 1024 x 1024 to 4096 x 1024 gray images at 1.5
/// bits per pixel to lossless decode 0-3% slower streamed, in a fifteenth to
/// a thirtieth of the memory.
const MIN_STREAMED_GRAY_SAMPLES_ON_ONE_THREAD: u64 = 1 << 20;

/// Samples from which a gray image streams on larger pools too: 5792 x 5792
/// decodes as fast streamed on 18 threads (0.96-1.02), 2896 x 2896 and
/// 4096 x 4096 are 3-14% slower.
const MIN_STREAMED_GRAY_SAMPLES_ON_A_LARGE_POOL: u64 = 5792 * 5792;

/// Coded bits per sample from which a full-resolution untiled classic image
/// streams as [`full_resolution_streams`] says: below it Tier-1 is light, the
/// decode is bound by the inverse transforms and the pack, and the planes'
/// fused last level does those faster on small images.
const MIN_STREAMED_CODED_BITS_PER_SAMPLE: u64 = 4;

/// Samples from which a full-resolution classic image below
/// [`MIN_STREAMED_CODED_BITS_PER_SAMPLE`] streams on a pool of up to
/// [`MAX_LOW_RATE_SHARED_READER_THREADS`] threads or of
/// [`CLASSIC_SYNTHESIS_THREADS`] or more: with its rows packed through the
/// fused RGB8 pack, a 4096 x 2048 image at 2 bits per pixel decodes 4-7%
/// faster streamed on 4 to 6 threads and 5% faster on 18, in less than half
/// the memory; on one thread 4096 x 2048 to 5428 x 4460 images at 1.5 to 2
/// bits per pixel decode 2-3% slower streamed, in a tenth to a twentieth of
/// the memory (5428 x 4460: 309 vs 16 MiB), and 2048 x 2048 to 2896 x 2896
/// ones 4% slower; 512 x 512 ones at 2 bits per pixel are 22% slower on 6
/// threads.
const MIN_STREAMED_LOW_RATE_SAMPLES: u64 = 1 << 23;

/// Pool threads up to which a large image below
/// [`MIN_STREAMED_CODED_BITS_PER_SAMPLE`] streams while one pool reader
/// reconstructs all of its components (below [`CLASSIC_SYNTHESIS_THREADS`]):
/// on more, Tier-1 of so few coded bits outruns that reader, and the
/// 4096 x 2048 image decodes 16-19% slower streamed on 10 threads.
const MAX_LOW_RATE_SHARED_READER_THREADS: usize = 6;

/// Whether an image the planes route can decode streams anyway, because
/// streaming was measured at least as fast for its kind, or on one thread
/// nearly as fast in much less memory; for three components:
/// - a full-resolution tiled image whose tiles decode at once (see
///   [`tile_workers`]);
/// - an untiled reduced image of [`MIN_REDUCED_STREAMED_SAMPLES`] output
///   samples or more: classic at any thread count, HTJ2K on two threads or
///   more and on one from 1/4 (1/4 outputs of 2896 x 2896 to 4096 x 4096
///   images at 1 to 12 bits per pixel 6-13% faster streamed, in a fifteenth
///   of the memory; 1/2 outputs 1-9% slower); a smaller classic one of
///   [`MIN_SMALL_REDUCED_STREAMED_SAMPLES`] or more on 2 to 11 pool threads
///   and of [`MIN_SMALL_REDUCED_STREAMED_SAMPLES_ON_ONE_THREAD`] or more on
///   one; on one thread a tiled one too (4096 x 4096 RGB in 1024 x 1024
///   tiles at 1/2 and 1/4 1-4% faster, in a fiftieth of the memory);
/// - a full-resolution classic image of at least 4 coded bits per pixel, or a
///   large one at fewer on most pools and on one thread (see
///   [`full_resolution_streams`]); a tiled one whose tiles do not decode at
///   once streams them one after another, so each of its tiles must meet
///   that as an image of its own, of [`MIN_STREAMED_TILE_SAMPLES`] or more
///   (4096 x 4096 RGB in 16 tiles of 1024 x 1024: 116.9 vs 124.9 ms on the
///   planes with 6 threads); on one thread any classic tile of
///   [`MIN_STREAMED_TILE_SAMPLES_ON_ONE_THREAD`] or more streams, and any
///   HTJ2K tile of [`MIN_STREAMED_SAMPLES`] or more.
///
/// A one-component classic image streams as [`gray_streams`] says, or on
/// one thread at full resolution in tiles of
/// [`MIN_STREAMED_TILE_SAMPLES_ON_ONE_THREAD`] or more, and a 16-bit output
/// on one thread as [`sixteen_bit_streams_on_one_thread`] says. Otherwise
/// untiled HTJ2K images at full resolution (whose streamed IDWT and pack
/// cost more per sample than the planes' fused last level: 5-28% slower on
/// one thread), HTJ2K gray images, small images and, on pools, reduced tiled
/// images (one tile after another was 1-13% slower on 6 threads, 1.5-10
/// times on 18) keep the planes.
fn streams_by_default(image: &Image<'_>, request: RouteRequest) -> bool {
    let threads = pool_threads(request.parallelism);
    let htj2k = image.uses_high_throughput_block_coding();
    let tiled = image.tile_count() > 1;
    let mut samples = u64::from(image.width()) * u64::from(image.height());
    if threads == 1
        && matches!(request.components, 1 | 3)
        && matches!(
            request.fmt,
            PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16
        )
        && sixteen_bit_streams_on_one_thread(image, samples)
    {
        return true;
    }
    if request.components == 1 {
        let reduced = request.scale != Downscale::None;
        if htj2k {
            return false;
        }
        if tiled && !reduced && threads == 1 {
            return average_tile_samples(image, samples)
                .is_some_and(|tile| tile >= MIN_STREAMED_TILE_SAMPLES_ON_ONE_THREAD);
        }
        return gray_streams(samples, reduced, tiled, threads);
    }
    if request.components != 3 {
        return false;
    }
    if request.scale != Downscale::None {
        return (!tiled || threads == 1) && reduced_streams(samples, request.scale, htj2k, threads);
    }
    let mut coded_bytes = request.coded_bytes;
    if tiled {
        let Ok(rects) = image.output_tile_rects() else {
            return false;
        };
        if tile_workers(&rects, threads).is_some() {
            return true;
        }
        let tile_count = rects.len().max(1);
        samples /= u64::try_from(tile_count).unwrap_or(u64::MAX);
        coded_bytes /= tile_count;
        if threads == 1 {
            return samples
                >= if htj2k {
                    MIN_STREAMED_SAMPLES
                } else {
                    MIN_STREAMED_TILE_SAMPLES_ON_ONE_THREAD
                };
        }
        if samples < MIN_STREAMED_TILE_SAMPLES {
            return false;
        }
    }
    !htj2k && full_resolution_streams(samples, coded_bytes, threads)
}

/// Whether a 16-bit decode of `image`, of `samples` output samples, streams
/// on one thread whatever its coding: from [`MIN_STREAMED_SAMPLES`], untiled
/// or in tiles of [`MIN_STREAMED_TILE_SAMPLES_ON_ONE_THREAD`] or more. The
/// planes route packs such outputs from planes of native samples it fills
/// first, which the streamed rows skip: 4096 x 2048 HTJ2K images decode
/// 21-25% faster streamed, 2896 x 2896 to 5428 x 4460 classic images at 1.5
/// bits per pixel 6-9% faster and 512 x 512 to 768 x 768 ones 1% faster,
/// while smaller images and tiles are 4-5% slower.
fn sixteen_bit_streams_on_one_thread(image: &Image<'_>, samples: u64) -> bool {
    samples >= MIN_STREAMED_SAMPLES
        && (image.tile_count() <= 1
            || average_tile_samples(image, samples)
                .is_some_and(|tile| tile >= MIN_STREAMED_TILE_SAMPLES_ON_ONE_THREAD))
}

/// The output samples of an average tile of `image`, of `samples` output
/// samples in all, or `None` when its tiles cannot be listed.
fn average_tile_samples(image: &Image<'_>, samples: u64) -> Option<u64> {
    let tiles = image.output_tile_rects().ok()?.len().max(1);
    Some(samples / u64::try_from(tiles).unwrap_or(u64::MAX))
}

/// Whether a one-component classic image of `samples` output samples,
/// `tiled` or not, streams on `threads` pool threads: on one thread reduced
/// to [`MIN_SMALL_REDUCED_STREAMED_SAMPLES_ON_ONE_THREAD`] or more, tiled or
/// not (5100 x 6600 and 5792 x 5792 images at 1/2 to 1/8 0-3% faster, in a
/// fiftieth to a hundredth of the memory), or untiled at full resolution
/// from [`MIN_STREAMED_GRAY_SAMPLES_ON_ONE_THREAD`]; on more threads untiled
/// at full resolution from [`MIN_STREAMED_GRAY_SAMPLES`], and from 12 threads
/// only from [`MIN_STREAMED_GRAY_SAMPLES_ON_A_LARGE_POOL`]. A tiled gray image
/// at full resolution streams on one thread in tiles of
/// [`MIN_STREAMED_TILE_SAMPLES_ON_ONE_THREAD`] or more (2048 x 2048 in
/// 512 x 512 tiles to 5100 x 6600 in 1024 x 1024 ones 1-3% slower, in a
/// fortieth to a hundredth of the memory).
fn gray_streams(samples: u64, reduced: bool, tiled: bool, threads: usize) -> bool {
    if threads == 1 {
        return if reduced {
            samples >= MIN_SMALL_REDUCED_STREAMED_SAMPLES_ON_ONE_THREAD
        } else {
            !tiled && samples >= MIN_STREAMED_GRAY_SAMPLES_ON_ONE_THREAD
        };
    }
    !reduced
        && !tiled
        && samples >= MIN_STREAMED_GRAY_SAMPLES
        && (threads < CLASSIC_SYNTHESIS_THREADS
            || samples >= MIN_STREAMED_GRAY_SAMPLES_ON_A_LARGE_POOL)
}

/// Whether a reduced-resolution decode of a three-component image at
/// `scale` to `samples` output samples streams on `threads` pool threads.
fn reduced_streams(samples: u64, scale: Downscale, htj2k: bool, threads: usize) -> bool {
    if samples >= MIN_REDUCED_STREAMED_SAMPLES {
        !htj2k || threads >= 2 || (threads == 1 && scale.denominator() >= 4)
    } else if threads == 1 {
        !htj2k && samples >= MIN_SMALL_REDUCED_STREAMED_SAMPLES_ON_ONE_THREAD
    } else {
        !htj2k
            && samples >= MIN_SMALL_REDUCED_STREAMED_SAMPLES
            && (2..CLASSIC_SYNTHESIS_THREADS).contains(&threads)
    }
}

/// Whether a full-resolution three-component classic image (or tile) of
/// `samples` pixels coded in `coded_bytes` streams on `threads` pool threads:
/// from [`MIN_STREAMED_SAMPLES`] and [`MIN_STREAMED_CODED_BITS_PER_SAMPLE`]
/// below [`CLASSIC_SYNTHESIS_THREADS`] pool threads, and on more when it is
/// below [`MAX_STREAMED_SAMPLES_ON_A_LARGE_POOL`]; at fewer coded bits, from
/// [`MIN_STREAMED_LOW_RATE_SAMPLES`] on up to
/// [`MAX_LOW_RATE_SHARED_READER_THREADS`] pool threads and from
/// [`CLASSIC_SYNTHESIS_THREADS`].
fn full_resolution_streams(samples: u64, coded_bytes: usize, threads: usize) -> bool {
    let coded_bits = u64::try_from(coded_bytes)
        .unwrap_or(u64::MAX)
        .saturating_mul(8);
    if coded_bits < samples.saturating_mul(MIN_STREAMED_CODED_BITS_PER_SAMPLE) {
        return samples >= MIN_STREAMED_LOW_RATE_SAMPLES
            && (matches!(threads, 1..=MAX_LOW_RATE_SHARED_READER_THREADS)
                || threads >= CLASSIC_SYNTHESIS_THREADS);
    }
    samples >= MIN_STREAMED_SAMPLES
        && match threads {
            0 => samples >= MIN_STREAMED_TILE_SAMPLES,
            1..CLASSIC_SYNTHESIS_THREADS => true,
            _ => samples < MAX_STREAMED_SAMPLES_ON_A_LARGE_POOL,
        }
}

/// The threads a decode with `parallelism` runs on.
fn pool_threads(parallelism: CpuDecodeParallelism) -> usize {
    match parallelism {
        CpuDecodeParallelism::Auto => j2k_native::cpu_decode_pool_threads(),
        CpuDecodeParallelism::Serial => 1,
    }
}

/// Tiles above which an image decodes one tile after another rather than at
/// once: each tile's region decode parses the tile-part headers of the whole
/// image again, a cost that grows with the square of the tiles.
const MAX_TILES_AT_ONCE: usize = 1024;

/// Output samples a tile needs on average for its own region decode to pay
/// for setting it up.
const MIN_SAMPLES_PER_TILE: usize = 256 * 256;

/// Workers that decode the tiles at `rects` at once on `threads` pool
/// threads: the calling thread and one per pool thread, the threads a
/// streamed region decode runs on. Each worker decodes its tile alone, so
/// the image needs a tile for every worker, and its tiles must keep at least
/// 80% of the workers busy over the rounds they take: 16 tiles on 7 workers
/// take three rounds with about a quarter of them idle, and decode no faster
/// than the planes. Other images decode one tile after another with the pool
/// behind each.
fn tile_workers(rects: &[(u32, u32, u32, u32)], threads: usize) -> Option<usize> {
    let tiles = rects.len();
    let samples: usize = rects
        .iter()
        .map(|&(_, _, w, h)| w as usize * h as usize)
        .sum();
    let workers = threads.checked_add(1)?;
    let rounds = tiles.div_ceil(workers);
    (threads >= 2
        && (workers..=MAX_TILES_AT_ONCE).contains(&tiles)
        && samples >= tiles.saturating_mul(MIN_SAMPLES_PER_TILE)
        && tiles.saturating_mul(5) >= rounds.saturating_mul(workers).saturating_mul(4))
    .then_some(workers)
}

/// Bytes a tile worker needs per sample of its tile beside the tile's
/// output rows: a streamed decode of one tile on one thread.
const EST_STREAMED_BYTES_PER_TILE_SAMPLE: usize = 16;

/// The streamed route for `image`: its tiles at once when it has a tile for
/// every worker and the workers' tiles fit in the native decode budget, else
/// one region decode of the whole image.
fn streamed_route(image: &Image<'_>, request: RouteRequest) -> FullImageRoute {
    let Ok(rects) = image.output_tile_rects() else {
        return FullImageRoute::StreamedRegion;
    };
    let Some(workers) = tile_workers(&rects, pool_threads(request.parallelism)) else {
        return FullImageRoute::StreamedRegion;
    };
    let largest = rects
        .iter()
        .map(|&(_, _, w, h)| w as usize * h as usize)
        .max()
        .unwrap_or(0);
    let per_worker = largest
        .saturating_mul(EST_STREAMED_BYTES_PER_TILE_SAMPLE + request.fmt.bytes_per_pixel())
        .max(1);
    let workers = workers.min(j2k_native::DEFAULT_MAX_DECODE_BYTES / per_worker);
    if workers < 2 {
        return FullImageRoute::StreamedRegion;
    }
    FullImageRoute::StreamedTiles {
        tiles: rects
            .into_iter()
            .map(|(x, y, w, h)| Rect { x, y, w, h })
            .collect(),
        workers,
    }
}

#[cfg(test)]
mod tests;
