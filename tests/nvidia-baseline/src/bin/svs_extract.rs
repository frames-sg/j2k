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
//       [--region-size N] [--format jpeg|ppm|pgm]
//
// Defaults: --limit 256 --stride 7 --quality 85. Near-white background tiles are
// skipped (mean luma > 235).

use std::path::Path;

use j2k_jpeg::encoder::{encode_jpeg_baseline, JpegEncodeOptions, JpegSamples};
use wsi_rs::{
    LevelIdx, PlaneIdx, PlaneSelection, RegionRequest, SceneId, SeriesId, Slide, TileLayout,
    TileOutputPreference, TilePixels, TileRequest,
};

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
    let mut region_size = None;
    let mut output_format = OutputFormat::Jpeg;
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
            "--region-size" => region_size = args.next().and_then(|v| v.parse().ok()),
            "--format" | "--output-format" => {
                let value = args.next().unwrap_or_else(|| "jpeg".to_string());
                output_format = OutputFormat::parse(&value).unwrap_or_else(|error| {
                    eprintln!("{error}");
                    std::process::exit(2);
                });
            }
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
        region_size,
        output_format,
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
    region_size: Option<u32>,
    output_format: OutputFormat,
}

#[derive(Clone, Copy)]
enum OutputFormat {
    Jpeg,
    Ppm,
    Pgm,
}

impl OutputFormat {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "jpeg" | "jpg" => Ok(Self::Jpeg),
            "ppm" => Ok(Self::Ppm),
            "pgm" => Ok(Self::Pgm),
            other => Err(format!(
                "--format must be jpeg, ppm, or pgm; got {other:?}"
            )),
        }
    }

    const fn extension(self) -> &'static str {
        match self {
            Self::Jpeg => "jpg",
            Self::Ppm => "ppm",
            Self::Pgm => "pgm",
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Jpeg => "JPEG",
            Self::Ppm => "PPM",
            Self::Pgm => "PGM",
        }
    }
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
    if matches!(options.region_size, Some(0)) {
        return Err("--region-size must be greater than zero".to_string());
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
    if let Some(region_size) = options.region_size {
        return extract_regions(slide, out_dir, options, grid, region_size);
    }

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

        write_rgb_image(
            out_dir,
            "tile",
            stats.written,
            &rgb,
            (w, h),
            options.output_format,
            jpeg_options,
        )
        .map_err(|e| format!("write tile row {row} col {col}: {e}"))?;
        stats.record_written(fraction);
    }
    Ok(stats)
}

fn extract_regions(
    slide: &Slide,
    out_dir: &Path,
    options: &ExtractOptions,
    grid: &SlideGrid,
    region_size: u32,
) -> Result<ExtractionStats, String> {
    let jpeg_options = JpegEncodeOptions {
        quality: options.quality,
        ..JpegEncodeOptions::default()
    };
    let region_size_u64 = u64::from(region_size);
    if grid.dimensions.0 < region_size_u64 || grid.dimensions.1 < region_size_u64 {
        return Err(format!(
            "--region-size {region_size} exceeds level dimensions {}x{}",
            grid.dimensions.0, grid.dimensions.1
        ));
    }
    let regions_across = ((grid.dimensions.0 - region_size_u64) / region_size_u64) + 1;
    let regions_down = ((grid.dimensions.1 - region_size_u64) / region_size_u64) + 1;
    let region_count = regions_across
        .checked_mul(regions_down)
        .ok_or_else(|| "region grid is too large".to_string())?;

    let mut stats = ExtractionStats::default();
    let mut index = 0u64;
    let stride = u64::try_from(options.stride.max(1)).map_err(|_| "--stride is too large")?;
    while index < region_count && stats.written < options.limit {
        let row = index / regions_across;
        let col = index % regions_across;
        index = index
            .checked_add(stride)
            .ok_or_else(|| "region stride overflow".to_string())?;

        stats.attempted += 1;
        let x = col
            .checked_mul(region_size_u64)
            .ok_or_else(|| "region x overflow".to_string())?;
        let y = row
            .checked_mul(region_size_u64)
            .ok_or_else(|| "region y overflow".to_string())?;
        let Ok(rgb) = read_region_rgb(slide, options, x, y, region_size) else {
            stats.decode_failures += 1;
            continue;
        };
        let fraction = tissue_fraction(&rgb);
        if fraction < options.min_tissue || luma_stddev(&rgb) < 12.0 {
            stats.skipped_blank += 1;
            continue;
        }

        write_rgb_image(
            out_dir,
            "region",
            stats.written,
            &rgb,
            (region_size, region_size),
            options.output_format,
            jpeg_options,
        )
        .map_err(|e| format!("write region x {x} y {y}: {e}"))?;
        stats.record_written(fraction);
    }
    Ok(stats)
}

fn print_stats(out_dir: &str, options: &ExtractOptions, stats: &ExtractionStats) {
    println!(
        "wrote {} {} images to {out_dir} (attempted {}, skipped {} below {:.0}% tissue, {} decode failures)",
        stats.written,
        options.output_format.label(),
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

fn read_region_rgb(
    slide: &Slide,
    options: &ExtractOptions,
    x: u64,
    y: u64,
    size: u32,
) -> Result<Vec<u8>, String> {
    let request = RegionRequest::new(
        SceneId::new(options.scene),
        SeriesId::new(options.series),
        LevelIdx::new(options.level),
        (
            i64::try_from(x).map_err(|_| "region x is too large")?,
            i64::try_from(y).map_err(|_| "region y is too large")?,
        ),
        (size, size),
    )
    .with_plane(PlaneIdx::new(PlaneSelection::default()));
    let rgba = slide
        .read_region_rgba(&request)
        .map_err(|e| format!("read region: {e}"))?
        .into_raw();
    Ok(rgba_to_rgb_on_white(&rgba))
}

fn rgba_to_rgb_on_white(rgba: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(rgba.len() / 4 * 3);
    for px in rgba.chunks_exact(4) {
        let alpha = u16::from(px[3]);
        let inv_alpha = 255_u16 - alpha;
        rgb.push(composite_channel(px[0], alpha, inv_alpha));
        rgb.push(composite_channel(px[1], alpha, inv_alpha));
        rgb.push(composite_channel(px[2], alpha, inv_alpha));
    }
    rgb
}

fn composite_channel(value: u8, alpha: u16, inv_alpha: u16) -> u8 {
    let blended = (u16::from(value) * alpha + 255 * inv_alpha + 127) / 255;
    u8::try_from(blended).unwrap_or(255)
}

fn write_rgb_image(
    out_dir: &Path,
    prefix: &str,
    index: usize,
    rgb: &[u8],
    dimensions: (u32, u32),
    format: OutputFormat,
    jpeg_options: JpegEncodeOptions,
) -> Result<(), String> {
    let (width, height) = dimensions;
    let path = out_dir.join(format!("{prefix}_{index:05}.{}", format.extension()));
    let bytes = match format {
        OutputFormat::Jpeg => {
            encode_jpeg_baseline(
                JpegSamples::Rgb8 {
                    data: rgb,
                    width,
                    height,
                },
                jpeg_options,
            )
            .map_err(|e| format!("encode JPEG: {e}"))?
            .data
        }
        OutputFormat::Ppm => {
            let mut bytes = format!("P6\n{width} {height}\n255\n").into_bytes();
            bytes.extend_from_slice(rgb);
            bytes
        }
        OutputFormat::Pgm => {
            let mut bytes = format!("P5\n{width} {height}\n255\n").into_bytes();
            bytes.extend(rgb_to_gray(rgb));
            bytes
        }
    };
    std::fs::write(&path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
}

fn rgb_to_gray(rgb: &[u8]) -> Vec<u8> {
    rgb.chunks_exact(3)
        .map(|px| {
            let luma =
                77_u16 * u16::from(px[0]) + 150_u16 * u16::from(px[1]) + 29_u16 * u16::from(px[2]);
            u8::try_from((luma + 128) >> 8).unwrap_or(255)
        })
        .collect()
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
