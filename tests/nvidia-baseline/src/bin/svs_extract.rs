// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Extract benchmark JPEG tiles from a whole-slide image.
//
// Many GDC SVS files store their tiles as JPEG 2000 (Aperio compression 33003 /
// 33005), not JPEG, so they cannot feed a JPEG -> HTJ2K transcode benchmark
// directly. This tool reads tiles through wsi-rs, converts them to RGB, and
// re-encodes a deterministic, tissue-only subset as baseline JPEG into an
// output directory for `transcode_compare`.
//
// Re-encoding adds one lossy step, so the tiles are realistic WSI *content* at
// realistic tile sizes rather than byte-identical originals - fine for a
// throughput benchmark (and the PSNR reference is self-consistent across codecs).
//
// Usage:
//   svs_extract <slide.svs> <out-dir> [--limit N] [--stride S] [--quality Q]
//
// Defaults: --limit 256 --stride 7 --quality 85. Near-white background tiles are
// skipped (mean luma > 235).

use std::path::Path;

use j2k_jpeg::encoder::{encode_jpeg_baseline, JpegEncodeOptions, JpegSamples};
use wsi_rs::{Slide, TileLayout, TileOutputPreference, TilePixels, TileRequest};

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(svs_path) = args.next() else {
        eprintln!(
            "usage: svs_extract <slide.svs> <out-dir> [--limit N] [--stride S] [--quality Q]"
        );
        std::process::exit(2);
    };
    let Some(out_dir) = args.next() else {
        eprintln!(
            "usage: svs_extract <slide.svs> <out-dir> [--limit N] [--stride S] [--quality Q]"
        );
        std::process::exit(2);
    };
    let mut limit = 256usize;
    let mut stride = 7usize;
    let mut quality = 85u8;
    let mut min_tissue = 0.5f64;
    let mut scene = 0usize;
    let mut series = 0usize;
    let mut level = 0u32;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--limit" => limit = args.next().and_then(|v| v.parse().ok()).unwrap_or(limit),
            "--stride" => {
                stride = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(stride)
                    .max(1);
            }
            "--quality" => quality = args.next().and_then(|v| v.parse().ok()).unwrap_or(quality),
            "--min-tissue" => {
                min_tissue = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(min_tissue);
            }
            "--scene" => scene = args.next().and_then(|v| v.parse().ok()).unwrap_or(scene),
            "--series" => series = args.next().and_then(|v| v.parse().ok()).unwrap_or(series),
            "--level" => level = args.next().and_then(|v| v.parse().ok()).unwrap_or(level),
            other => {
                eprintln!("unknown flag: {other}");
                std::process::exit(2);
            }
        }
    }

    let options = ExtractOptions {
        limit,
        stride,
        quality,
        min_tissue,
        scene,
        series,
        level,
    };
    if let Err(error) = run(&svs_path, &out_dir, &options) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

struct ExtractOptions {
    limit: usize,
    stride: usize,
    quality: u8,
    min_tissue: f64,
    scene: usize,
    series: usize,
    level: u32,
}

struct SlideGrid {
    dimensions: (u64, u64),
    tile_width: u32,
    tile_height: u32,
    tiles_across: u64,
    tile_count: u64,
}

#[derive(Default)]
struct ExtractionStats {
    written: usize,
    attempted: usize,
    skipped_blank: usize,
    decode_failures: usize,
    min_seen_fraction: f64,
    sum_fraction: f64,
}

impl ExtractionStats {
    fn record_written(&mut self, tissue_fraction: f64) {
        self.written += 1;
        if self.written == 1 {
            self.min_seen_fraction = tissue_fraction;
        } else {
            self.min_seen_fraction = self.min_seen_fraction.min(tissue_fraction);
        }
        self.sum_fraction += tissue_fraction;
    }

    fn mean_tissue_fraction(&self) -> f64 {
        if self.written == 0 {
            0.0
        } else {
            self.sum_fraction / self.written as f64
        }
    }
}

fn run(svs_path: &str, out_dir: &str, options: &ExtractOptions) -> Result<(), String> {
    validate_options(options)?;
    let slide = Slide::open(svs_path).map_err(|e| format!("open slide {svs_path}: {e}"))?;
    let grid = slide_grid(&slide, options)?;
    println!(
        "slide: {}x{} px, {}x{} tiles, {} tiles total, scene {}, series {}, level {}",
        grid.dimensions.0,
        grid.dimensions.1,
        grid.tile_width,
        grid.tile_height,
        grid.tile_count,
        options.scene,
        options.series,
        options.level
    );

    std::fs::create_dir_all(out_dir).map_err(|e| format!("create {out_dir}: {e}"))?;
    let stats = extract_tiles(&slide, Path::new(out_dir), options, &grid)?;
    print_stats(out_dir, options, &stats);
    if stats.written == 0 {
        return Err(
            "no tiles written - decode may be unsupported, or all tiles below the tissue threshold"
                .to_string(),
        );
    }
    Ok(())
}

fn validate_options(options: &ExtractOptions) -> Result<(), String> {
    if options.limit == 0 {
        return Err("--limit must be greater than zero".to_string());
    }
    if !(0.0..=1.0).contains(&options.min_tissue) {
        return Err("--min-tissue must be between 0 and 1".to_string());
    }
    Ok(())
}

fn slide_grid(slide: &Slide, options: &ExtractOptions) -> Result<SlideGrid, String> {
    let level = selected_level(slide, options.scene, options.series, options.level)?;
    let TileLayout::Regular {
        tile_width,
        tile_height,
        tiles_across,
        tiles_down,
    } = level.tile_layout
    else {
        return Err(format!(
            "scene {} series {} level {} is not a regular tiled level",
            options.scene, options.series, options.level
        ));
    };
    let tile_count = tiles_across
        .checked_mul(tiles_down)
        .ok_or_else(|| "tile grid is too large".to_string())?;
    Ok(SlideGrid {
        dimensions: level.dimensions,
        tile_width,
        tile_height,
        tiles_across,
        tile_count,
    })
}

fn extract_tiles(
    slide: &Slide,
    out_dir: &Path,
    options: &ExtractOptions,
    grid: &SlideGrid,
) -> Result<ExtractionStats, String> {
    let jpeg_options = JpegEncodeOptions {
        quality: options.quality,
        ..JpegEncodeOptions::default()
    };

    let mut stats = ExtractionStats::default();
    let mut index = 0u64;
    let stride = u64::try_from(options.stride.max(1)).map_err(|_| "--stride is too large")?;
    while index < grid.tile_count && stats.written < options.limit {
        let row = index / grid.tiles_across;
        let col = index % grid.tiles_across;
        index = index
            .checked_add(stride)
            .ok_or_else(|| "tile stride overflow".to_string())?;

        stats.attempted += 1;

        let request = TileRequest::new(
            options.scene,
            options.series,
            options.level,
            i64::try_from(col).map_err(|_| "tile column is too large")?,
            i64::try_from(row).map_err(|_| "tile row is too large")?,
        );
        let Ok((rgb, w, h)) = read_tile_rgb(slide, &request) else {
            stats.decode_failures += 1;
            continue;
        };
        let fraction = tissue_fraction(&rgb);
        // Require both real tissue coverage and visible structure (texture), so
        // flat homogeneous stroma/background is rejected in favour of cellular
        // tissue with nuclei and edges.
        if fraction < options.min_tissue || luma_stddev(&rgb) < 12.0 {
            stats.skipped_blank += 1;
            continue;
        }

        let encoded = encode_jpeg_baseline(
            JpegSamples::Rgb8 {
                data: &rgb,
                width: w,
                height: h,
            },
            jpeg_options,
        )
        .map_err(|e| format!("encode tile row {row} col {col}: {e}"))?;

        let path = out_dir.join(format!("tile_{:05}.jpg", stats.written));
        std::fs::write(&path, &encoded.data)
            .map_err(|e| format!("write {}: {e}", path.display()))?;
        stats.record_written(fraction);
    }
    Ok(stats)
}

fn print_stats(out_dir: &str, options: &ExtractOptions, stats: &ExtractionStats) {
    println!(
        "wrote {} JPEG tiles to {out_dir} (attempted {}, skipped {} below {:.0}% tissue, {} decode failures)",
        stats.written,
        stats.attempted,
        stats.skipped_blank,
        options.min_tissue * 100.0,
        stats.decode_failures
    );
    println!(
        "tissue coverage of written tiles: min {:.0}%, mean {:.0}%",
        stats.min_seen_fraction * 100.0,
        stats.mean_tissue_fraction() * 100.0
    );
}

fn selected_level(
    slide: &Slide,
    scene: usize,
    series: usize,
    level: u32,
) -> Result<&wsi_rs::Level, String> {
    let dataset = slide.dataset();
    let scene_ref = dataset
        .scenes
        .get(scene)
        .ok_or_else(|| format!("scene {scene} is out of range"))?;
    let series_ref = scene_ref
        .series
        .get(series)
        .ok_or_else(|| format!("series {series} is out of range"))?;
    series_ref
        .levels
        .get(level as usize)
        .ok_or_else(|| format!("level {level} is out of range"))
}

fn read_tile_rgb(slide: &Slide, request: &TileRequest) -> Result<(Vec<u8>, u32, u32), String> {
    match slide
        .read_tile(request, TileOutputPreference::cpu())
        .map_err(|e| format!("read tile: {e}"))?
    {
        TilePixels::Cpu(tile) => {
            let rgb = tile
                .to_rgb()
                .map_err(|e| format!("convert tile to RGB: {e}"))?;
            let width = rgb.width();
            let height = rgb.height();
            Ok((rgb.into_raw(), width, height))
        }
        TilePixels::Device(_) => Err("wsi-rs returned a device tile for CPU output".to_string()),
        _ => Err("wsi-rs returned an unsupported tile output for CPU output".to_string()),
    }
}

/// Fraction of pixels that look like stained tissue rather than bright glass /
/// background. A pixel is tissue when it has meaningful absorption (its darkest
/// channel is well below white) or visible stain saturation (channel spread).
fn tissue_fraction(rgb: &[u8]) -> f64 {
    if rgb.is_empty() {
        return 0.0;
    }
    let mut tissue = 0usize;
    let total = rgb.len() / 3;
    for px in rgb.chunks_exact(3) {
        let min_c = px[0].min(px[1]).min(px[2]);
        let max_c = px[0].max(px[1]).max(px[2]);
        let absorbs = min_c < 210; // not near-white on every channel
        let saturated = u16::from(max_c) - u16::from(min_c) > 25; // visible stain color
        if absorbs || saturated {
            tissue += 1;
        }
    }
    tissue as f64 / total as f64
}

/// Standard deviation of per-pixel luma, a cheap proxy for spatial structure.
/// Flat regions (uniform stain, glass) score near zero; cellular tissue is high.
fn luma_stddev(rgb: &[u8]) -> f64 {
    let total = rgb.len() / 3;
    if total == 0 {
        return 0.0;
    }
    let luma: Vec<f64> = rgb
        .chunks_exact(3)
        .map(|px| 0.299 * f64::from(px[0]) + 0.587 * f64::from(px[1]) + 0.114 * f64::from(px[2]))
        .collect();
    let mean = luma.iter().sum::<f64>() / total as f64;
    let variance = luma.iter().map(|&l| (l - mean) * (l - mean)).sum::<f64>() / total as f64;
    variance.sqrt()
}
