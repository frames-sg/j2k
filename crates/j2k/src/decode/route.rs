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
//! the decode, and a decode never reruns after a failure.

use j2k_core::{PixelFormat, Rect};
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
    /// streaming is the faster route (see [`streams_by_default`]).
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
    /// Whether the image is decoded below full resolution.
    pub(crate) reduced: bool,
}

/// How a full-image decode of `image` runs for `request`: streamed when the
/// output has 8 bits per sample, or 16 bits whose rows carry the native
/// samples the planes route packs (see `u16_rows_pack_floor`), the region
/// decode streams, and either the planes (or their 16-bit pack) would pass
/// the budget or the streamed route is the faster one for this image.
///
/// A planning error keeps the planes route, which meets the same error and
/// reports it.
pub(crate) fn full_image_route(image: &Image<'_>, request: RouteRequest) -> FullImageRoute {
    let pack_floor = match request.fmt {
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Gray8 => 0,
        PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16 => {
            match u16_rows_pack_floor(image, request.fmt) {
                Some(floor) => floor,
                None => return FullImageRoute::Planes,
            }
        }
        _ => return FullImageRoute::Planes,
    };
    let planes_budget = match request.budget {
        PlanesBudget::Default => j2k_native::DEFAULT_MAX_DECODE_BYTES,
        PlanesBudget::Fixed(bytes) => bytes,
    };
    // A budget of zero skips the planner's header-only shortcut, so the
    // floor is known whenever the region decode would stream; the shortcut
    // saves the tile parse when only the budget can choose streaming.
    let streams = if request.budget == PlanesBudget::Default && streams_by_default(image, request) {
        matches!(image.full_decode_planes_floor(0), Ok(Some(_)))
    } else {
        matches!(
            image.full_decode_planes_floor(planes_budget),
            Ok(Some(floor)) if floor.max(pack_floor) > planes_budget
        )
    };
    if streams {
        streamed_route(image, request)
    } else {
        FullImageRoute::Planes
    }
}

/// Pool threads from which a classic tile of several components
/// reconstructs each component on its own pool reader (`parallel_synthesis`
/// in the native decoder's pool bands). From there, a full-resolution
/// untiled image decodes 3-6% faster on the planes.
const CLASSIC_SYNTHESIS_THREADS: usize = 12;

/// Samples from which a full-resolution classic image streams on 2 to 11
/// pool threads (a 512 x 512 image at 6.6 coded bits per pixel and 768 x 512
/// ones at 9.6 decode 5-10% faster streamed on 6 threads); smaller images
/// spend more of their time setting up the pool bands.
const MIN_STREAMED_SAMPLES: u64 = 1 << 18;

/// Samples below which a full-resolution classic image also streams from
/// [`CLASSIC_SYNTHESIS_THREADS`] pool threads: the 512 x 512 and 768 x 512
/// images decode 3-14% faster streamed on 18 threads, a 4096 x 1024 one
/// 6-7% slower.
const MAX_STREAMED_SAMPLES_ON_A_LARGE_POOL: u64 = 1 << 19;

/// Samples a tile needs to stream one after another with the pool behind
/// it (measured on 1024 x 1024 tiles only), and an untiled classic image to
/// stream on one thread: 2048 x 2048 to 4096 x 4096 RGB images at 5-11
/// coded bits per pixel decode as fast streamed on one thread (0.99-1.01),
/// in a third of the memory.
const MIN_STREAMED_TILE_SAMPLES: u64 = 1 << 20;

/// Output samples from which a reduced-resolution decode streams on any
/// pool (a 1024 x 512 output and larger: 6-53% faster on 18 threads). On 12
/// pool threads or more a smaller classic output decodes on the pool's
/// bands as slowly as on one thread (a 384 x 256 output of a 768 x 512
/// image 4.9 times as long as on the planes), and smaller HTJ2K outputs are
/// up to 59% slower streamed on any pool.
const MIN_REDUCED_STREAMED_SAMPLES: u64 = 1 << 19;

/// Output samples from which a smaller reduced classic decode streams on 2
/// to 11 pool threads: 256 x 256 and 384 x 256 outputs decode 3-12% faster
/// streamed on 6 threads, a 192 x 128 one 1-3% slower.
const MIN_SMALL_REDUCED_STREAMED_SAMPLES: u64 = 1 << 16;

/// Samples from which an untiled one-component classic image streams at
/// full resolution below [`CLASSIC_SYNTHESIS_THREADS`] pool threads: the
/// smallest gray image measured. With the pool bands decoding a band in
/// parts when the rings hold too few bands, 2896 x 2896 to 5792 x 5792 gray
/// images decode 12-15% faster streamed on 6 threads and as fast on one, in
/// a fifth to a seventh of the memory (5792 x 5792: 459 vs 68 MB).
const MIN_STREAMED_GRAY_SAMPLES: u64 = 2896 * 2896;

/// Samples from which a gray image streams on larger pools too: 5792 x 5792
/// decodes as fast streamed on 18 threads (0.96-1.02), 2896 x 2896 and
/// 4096 x 4096 are 3-14% slower.
const MIN_STREAMED_GRAY_SAMPLES_ON_A_LARGE_POOL: u64 = 5792 * 5792;

/// Coded bits per sample from which a full-resolution untiled classic image
/// streams: below it Tier-1 is light, the decode is bound by the inverse
/// transforms and the pack, and the planes' fused last level does those
/// faster (a 2 bits-per-pixel image is 0-3% slower streamed, 5-11 bits per
/// pixel images are 10% faster).
const MIN_STREAMED_CODED_BITS_PER_SAMPLE: u64 = 4;

/// Whether an image the planes route can decode streams anyway, because
/// streaming was measured at least as fast for its kind; for three
/// components:
/// - a full-resolution tiled image whose tiles decode at once (see
///   [`tile_workers`]);
/// - an untiled reduced image of [`MIN_REDUCED_STREAMED_SAMPLES`] output
///   samples or more: classic at any thread count, HTJ2K on two threads or
///   more (one thread decodes a half resolution HTJ2K image 2-9% faster on
///   the planes); a smaller classic one of
///   [`MIN_SMALL_REDUCED_STREAMED_SAMPLES`] or more on 2 to 11 pool threads;
/// - a full-resolution classic image of at least 4 coded bits per pixel (see
///   [`full_resolution_streams`]); a tiled one whose tiles do not decode at
///   once streams them one after another, so each of its tiles must meet
///   that as an image of its own, of [`MIN_STREAMED_TILE_SAMPLES`] or more
///   (4096 x 4096 RGB in 16 tiles of 1024 x 1024: 116.9 vs 124.9 ms on the
///   planes with 6 threads).
///
/// An untiled one-component classic image streams at full resolution as
/// [`gray_streams`] says; a tiled three-component one on one thread streams
/// its tiles one after another from 512 x 512 tiles (as fast as the planes:
/// 0.93-1.00, in a quarter of the memory). Reduced and tiled gray images,
/// full-resolution HTJ2K images (whose streamed IDWT and pack cost more per
/// sample than the planes' fused last level), small images and reduced
/// tiled images (one tile after another was 1-13% slower on 6 threads,
/// 1.5-10 times on 18) keep the planes.
fn streams_by_default(image: &Image<'_>, request: RouteRequest) -> bool {
    let threads = pool_threads(request.parallelism);
    let htj2k = image.uses_high_throughput_block_coding();
    let tiled = image.tile_count() > 1;
    let mut samples = u64::from(image.width()) * u64::from(image.height());
    if request.components == 1 {
        return !request.reduced && !tiled && !htj2k && gray_streams(samples, threads);
    }
    if request.components != 3 {
        return false;
    }
    if request.reduced {
        return !tiled && reduced_streams(samples, htj2k, threads);
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
            return samples >= MIN_STREAMED_SAMPLES;
        }
        if samples < MIN_STREAMED_TILE_SAMPLES {
            return false;
        }
    }
    !htj2k && full_resolution_streams(samples, coded_bytes, threads)
}

/// Whether an untiled one-component classic image of `samples` streams at
/// full resolution on `threads` pool threads.
fn gray_streams(samples: u64, threads: usize) -> bool {
    samples >= MIN_STREAMED_GRAY_SAMPLES
        && (threads < CLASSIC_SYNTHESIS_THREADS
            || samples >= MIN_STREAMED_GRAY_SAMPLES_ON_A_LARGE_POOL)
}

/// Whether a reduced-resolution decode of an untiled three-component image
/// to `samples` output samples streams on `threads` pool threads.
fn reduced_streams(samples: u64, htj2k: bool, threads: usize) -> bool {
    if samples >= MIN_REDUCED_STREAMED_SAMPLES {
        !htj2k || threads >= 2
    } else {
        !htj2k
            && samples >= MIN_SMALL_REDUCED_STREAMED_SAMPLES
            && (2..CLASSIC_SYNTHESIS_THREADS).contains(&threads)
    }
}

/// Whether a full-resolution three-component classic image (or tile) of
/// `samples` pixels coded in `coded_bytes` streams on `threads` pool threads:
/// from [`MIN_STREAMED_SAMPLES`] and [`MIN_STREAMED_CODED_BITS_PER_SAMPLE`],
/// on 2 to 11 pool threads, on more when it is below
/// [`MAX_STREAMED_SAMPLES_ON_A_LARGE_POOL`], and on one from
/// [`MIN_STREAMED_TILE_SAMPLES`].
fn full_resolution_streams(samples: u64, coded_bytes: usize, threads: usize) -> bool {
    let coded_bits = u64::try_from(coded_bytes)
        .unwrap_or(u64::MAX)
        .saturating_mul(8);
    samples >= MIN_STREAMED_SAMPLES
        && coded_bits >= samples.saturating_mul(MIN_STREAMED_CODED_BITS_PER_SAMPLE)
        && match threads {
            0 | 1 => samples >= MIN_STREAMED_TILE_SAMPLES,
            2..CLASSIC_SYNTHESIS_THREADS => true,
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

/// Tiles from which an image decodes one tile after another rather than at
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
/// take three rounds with a third of them idle, and decode no faster than
/// the planes. Other images decode one tile after another with the pool
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
