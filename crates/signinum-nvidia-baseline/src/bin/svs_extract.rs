// SPDX-License-Identifier: Apache-2.0
//
// Extract benchmark JPEG tiles from an Aperio .svs whole-slide image.
//
// Many GDC SVS files store their tiles as JPEG 2000 (Aperio compression 33003 /
// 33005), not JPEG, so they cannot feed a JPEG -> HTJ2K transcode benchmark
// directly. This tool decodes each tile (J2K via signinum-j2k-native; original
// JPEG passed through) to RGB and re-encodes a deterministic, tissue-only subset
// as baseline JPEG into an output directory for `transcode_compare`.
//
// Re-encoding adds one lossy step, so the tiles are realistic WSI *content* at
// realistic tile sizes rather than byte-identical originals — fine for a
// throughput benchmark (and the PSNR reference is self-consistent across codecs).
//
// Usage:
//   svs_extract <slide.svs> <out-dir> [--limit N] [--stride S] [--quality Q]
//
// Defaults: --limit 256 --stride 7 --quality 85. Near-white background tiles are
// skipped (mean luma > 235).

use std::path::Path;

use signinum_j2k_native::{DecodeSettings, Image};
use signinum_jpeg::encoder::{encode_jpeg_baseline, JpegEncodeOptions, JpegSamples};

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(svs_path) = args.next() else {
        eprintln!("usage: svs_extract <slide.svs> <out-dir> [--limit N] [--stride S] [--quality Q]");
        std::process::exit(2);
    };
    let Some(out_dir) = args.next() else {
        eprintln!("usage: svs_extract <slide.svs> <out-dir> [--limit N] [--stride S] [--quality Q]");
        std::process::exit(2);
    };
    let mut limit = 256usize;
    let mut stride = 7usize;
    let mut quality = 85u8;
    let mut min_tissue = 0.5f64;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--limit" => limit = args.next().and_then(|v| v.parse().ok()).unwrap_or(limit),
            "--stride" => stride = args.next().and_then(|v| v.parse().ok()).unwrap_or(stride).max(1),
            "--quality" => quality = args.next().and_then(|v| v.parse().ok()).unwrap_or(quality),
            "--min-tissue" => {
                min_tissue = args.next().and_then(|v| v.parse().ok()).unwrap_or(min_tissue);
            }
            other => {
                eprintln!("unknown flag: {other}");
                std::process::exit(2);
            }
        }
    }

    if let Err(error) = run(&svs_path, &out_dir, limit, stride, quality, min_tissue) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run(
    svs_path: &str,
    out_dir: &str,
    limit: usize,
    stride: usize,
    quality: u8,
    min_tissue: f64,
) -> Result<(), String> {
    let bytes = std::fs::read(svs_path).map_err(|e| format!("read {svs_path}: {e}"))?;
    let ifd0 = parse_first_ifd(&bytes)?;
    println!(
        "slide: {}x{} px, {}x{} tiles, compression {}, {} tiles total",
        ifd0.image_width, ifd0.image_height, ifd0.tile_width, ifd0.tile_height,
        ifd0.compression, ifd0.tile_offsets.len()
    );

    std::fs::create_dir_all(out_dir).map_err(|e| format!("create {out_dir}: {e}"))?;

    let options = JpegEncodeOptions {
        quality,
        ..JpegEncodeOptions::default()
    };

    let mut written = 0usize;
    let mut attempted = 0usize;
    let mut skipped_blank = 0usize;
    let mut decode_failures = 0usize;
    let mut min_seen_fraction = 1.0f64;
    let mut sum_fraction = 0.0f64;

    let mut index = 0usize;
    while index < ifd0.tile_offsets.len() && written < limit {
        let offset = ifd0.tile_offsets[index] as usize;
        let count = ifd0.tile_byte_counts[index] as usize;
        index += stride;
        if count == 0 || offset + count > bytes.len() {
            continue;
        }
        attempted += 1;
        let tile = &bytes[offset..offset + count];

        let Some((rgb, w, h)) = decode_tile_rgb(tile, ifd0.compression) else {
            decode_failures += 1;
            continue;
        };
        let fraction = tissue_fraction(&rgb);
        // Require both real tissue coverage and visible structure (texture), so
        // flat homogeneous stroma/background is rejected in favour of cellular
        // tissue with nuclei and edges.
        if fraction < min_tissue || luma_stddev(&rgb) < 12.0 {
            skipped_blank += 1;
            continue;
        }

        let encoded = encode_jpeg_baseline(
            JpegSamples::Rgb8 { data: &rgb, width: w, height: h },
            options,
        )
        .map_err(|e| format!("encode tile {index}: {e}"))?;

        let path = Path::new(out_dir).join(format!("tile_{written:05}.jpg"));
        std::fs::write(&path, &encoded.data).map_err(|e| format!("write {}: {e}", path.display()))?;
        written += 1;
        min_seen_fraction = min_seen_fraction.min(fraction);
        sum_fraction += fraction;
    }

    let mean_fraction = if written > 0 { sum_fraction / written as f64 } else { 0.0 };
    println!(
        "wrote {written} JPEG tiles to {out_dir} (attempted {attempted}, skipped {skipped_blank} below {:.0}% tissue, {decode_failures} decode failures)",
        min_tissue * 100.0
    );
    println!(
        "tissue coverage of written tiles: min {:.0}%, mean {:.0}%",
        min_seen_fraction * 100.0,
        mean_fraction * 100.0
    );
    if written == 0 {
        return Err("no tiles written — decode may be unsupported, or all tiles below the tissue threshold".to_string());
    }
    Ok(())
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

// Aperio JPEG 2000 with YCbCr components (the codestream stores Y/Cb/Cr planes
// directly rather than using J2K's in-stream color transform).
const APERIO_J2K_YCBCR: u16 = 33003;

/// Decode one tile to interleaved RGB. JPEG 2000 / HTJ2K codestreams and JPEG
/// tiles are both handled by the native parser via `Image`. Aperio's J2K-YCbCr
/// tiles decode to YCbCr component values, which are converted to RGB here.
fn decode_tile_rgb(tile: &[u8], compression: u16) -> Option<(Vec<u8>, u32, u32)> {
    let image = Image::new(tile, &DecodeSettings::default()).ok()?;
    let bitmap = image.decode_native().ok()?;
    if bitmap.bytes_per_sample != 1 {
        return None;
    }
    match bitmap.num_components {
        3 => {
            let rgb = if compression == APERIO_J2K_YCBCR {
                ycbcr_to_rgb(&bitmap.data)
            } else {
                bitmap.data
            };
            Some((rgb, bitmap.width, bitmap.height))
        }
        1 => {
            // Expand grayscale to RGB so the encoder path is uniform.
            let rgb = bitmap.data.iter().flat_map(|&g| [g, g, g]).collect();
            Some((rgb, bitmap.width, bitmap.height))
        }
        _ => None,
    }
}

/// JFIF full-range YCbCr -> RGB, interleaved.
fn ycbcr_to_rgb(ycbcr: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(ycbcr.len());
    for px in ycbcr.chunks_exact(3) {
        let y = f32::from(px[0]);
        let cb = f32::from(px[1]) - 128.0;
        let cr = f32::from(px[2]) - 128.0;
        rgb.push((y + 1.402 * cr).clamp(0.0, 255.0) as u8);
        rgb.push((y - 0.344_136 * cb - 0.714_136 * cr).clamp(0.0, 255.0) as u8);
        rgb.push((y + 1.772 * cb).clamp(0.0, 255.0) as u8);
    }
    rgb
}

struct Ifd {
    image_width: u32,
    image_height: u32,
    tile_width: u32,
    tile_height: u32,
    compression: u16,
    tile_offsets: Vec<u32>,
    tile_byte_counts: Vec<u32>,
}

/// Minimal classic little-endian TIFF reader for the first (full-resolution) IFD.
fn parse_first_ifd(bytes: &[u8]) -> Result<Ifd, String> {
    if bytes.len() < 8 || &bytes[0..2] != b"II" {
        return Err("not a little-endian TIFF (expected II byte order)".to_string());
    }
    let magic = u16::from_le_bytes([bytes[2], bytes[3]]);
    if magic != 42 {
        return Err(format!("unsupported TIFF magic {magic} (BigTIFF is not supported)"));
    }
    let ifd_off = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    if ifd_off + 2 > bytes.len() {
        return Err("IFD offset out of range".to_string());
    }
    let entry_count = u16::from_le_bytes([bytes[ifd_off], bytes[ifd_off + 1]]) as usize;

    let mut compression = 1u16;
    let (mut image_width, mut image_height) = (0u32, 0u32);
    let (mut tile_width, mut tile_height) = (0u32, 0u32);
    let mut tile_offsets = Vec::new();
    let mut tile_byte_counts = Vec::new();

    for i in 0..entry_count {
        let base = ifd_off + 2 + i * 12;
        if base + 12 > bytes.len() {
            return Err("IFD entry out of range".to_string());
        }
        let tag = u16::from_le_bytes([bytes[base], bytes[base + 1]]);
        let typ = u16::from_le_bytes([bytes[base + 2], bytes[base + 3]]);
        let count = u32::from_le_bytes([bytes[base + 4], bytes[base + 5], bytes[base + 6], bytes[base + 7]]);
        let value_field = &bytes[base + 8..base + 12];
        match tag {
            256 => image_width = scalar(typ, value_field),
            257 => image_height = scalar(typ, value_field),
            259 => compression = scalar(typ, value_field) as u16,
            322 => tile_width = scalar(typ, value_field),
            323 => tile_height = scalar(typ, value_field),
            324 => tile_offsets = read_u32_array(bytes, typ, count, value_field)?,
            325 => tile_byte_counts = read_u32_array(bytes, typ, count, value_field)?,
            _ => {}
        }
    }

    if tile_offsets.is_empty() || tile_offsets.len() != tile_byte_counts.len() {
        return Err("IFD has no tiles or mismatched tile arrays (is this a tiled SVS?)".to_string());
    }
    Ok(Ifd {
        image_width,
        image_height,
        tile_width,
        tile_height,
        compression,
        tile_offsets,
        tile_byte_counts,
    })
}

/// Read an inline scalar (SHORT=3 or LONG=4) from a 4-byte value field.
fn scalar(typ: u16, value_field: &[u8]) -> u32 {
    match typ {
        3 => u32::from(u16::from_le_bytes([value_field[0], value_field[1]])),
        _ => u32::from_le_bytes([value_field[0], value_field[1], value_field[2], value_field[3]]),
    }
}

/// Read a SHORT/LONG array, inline when it fits in 4 bytes else at the offset.
fn read_u32_array(bytes: &[u8], typ: u16, count: u32, value_field: &[u8]) -> Result<Vec<u32>, String> {
    let count = count as usize;
    let elem = match typ {
        3 => 2usize,
        4 => 4usize,
        other => return Err(format!("unsupported tile-array type {other}")),
    };
    let total = count * elem;
    let data: &[u8] = if total <= 4 {
        value_field
    } else {
        let off = u32::from_le_bytes([value_field[0], value_field[1], value_field[2], value_field[3]]) as usize;
        bytes.get(off..off + total).ok_or_else(|| "tile array out of range".to_string())?
    };
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let v = if elem == 2 {
            u32::from(u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]))
        } else {
            u32::from_le_bytes([data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]])
        };
        out.push(v);
    }
    Ok(out)
}
