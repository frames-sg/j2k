// SPDX-License-Identifier: Apache-2.0

use j2k::J2kDecoder;
use j2k_core::{Downscale, PixelFormat, Rect};
use j2k_native::{encode_htj2k, EncodeOptions};
use j2k_test_support::{gradient_u8, read_pnm_pixels, wrap_codestream_jp2, write_pnm};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        atomic::{AtomicUsize, Ordering},
        OnceLock,
    },
};

#[test]
fn classic_gray_full_decode_matches_grok() {
    let Some(path) = grok_decompress_bin() else {
        return;
    };
    let pixels = gradient_u8(128, 128, 1);
    let jp2 = classic_jp2(&pixels, 128, 128, 1).expect("classic jp2");

    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut out = vec![0_u8; 128 * 128];
    decoder
        .decode_into(&mut out, 128, PixelFormat::Gray8)
        .expect("j2k decode");

    let expected = decode_with_grok(&path, "grok_full_gray", &jp2, ".pgm", &[]);
    assert_eq!(out, expected);
}

#[test]
fn classic_rgb_full_decode_matches_grok() {
    let Some(path) = grok_decompress_bin() else {
        return;
    };
    let pixels = gradient_u8(128, 128, 3);
    let jp2 = classic_jp2(&pixels, 128, 128, 3).expect("classic jp2");

    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut out = vec![0_u8; 128 * 128 * 3];
    decoder
        .decode_into(&mut out, 128 * 3, PixelFormat::Rgb8)
        .expect("j2k decode");

    let expected = decode_with_grok(&path, "grok_full_rgb", &jp2, ".ppm", &[]);
    assert_eq!(out, expected);
}

#[test]
fn classic_gray_region_decode_matches_grok_area_decode() {
    let Some(path) = grok_decompress_bin() else {
        return;
    };
    let pixels = gradient_u8(128, 128, 1);
    let jp2 = classic_jp2(&pixels, 128, 128, 1).expect("classic jp2");
    let roi = Rect {
        x: 16,
        y: 24,
        w: 48,
        h: 48,
    };

    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut out = vec![0_u8; roi.w as usize * roi.h as usize];
    decoder
        .decode_region_into(
            &mut j2k::J2kScratchPool::new(),
            &mut out,
            roi.w as usize,
            PixelFormat::Gray8,
            roi,
        )
        .expect("j2k region decode");

    let expected = decode_with_grok(
        &path,
        "grok_region_gray",
        &jp2,
        ".pgm",
        &[
            "-d",
            &format!("{},{},{},{}", roi.x, roi.y, roi.x + roi.w, roi.y + roi.h),
        ],
    );
    assert_eq!(out, expected);
}

#[test]
fn classic_rgb_region_decode_matches_grok_area_decode() {
    let Some(path) = grok_decompress_bin() else {
        return;
    };
    let pixels = gradient_u8(128, 128, 3);
    let jp2 = classic_jp2(&pixels, 128, 128, 3).expect("classic jp2");
    let roi = Rect {
        x: 16,
        y: 24,
        w: 48,
        h: 48,
    };

    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut out = vec![0_u8; roi.w as usize * roi.h as usize * 3];
    decoder
        .decode_region_into(
            &mut j2k::J2kScratchPool::new(),
            &mut out,
            roi.w as usize * 3,
            PixelFormat::Rgb8,
            roi,
        )
        .expect("j2k region decode");

    let expected = decode_with_grok(
        &path,
        "grok_region_rgb",
        &jp2,
        ".ppm",
        &[
            "-d",
            &format!("{},{},{},{}", roi.x, roi.y, roi.x + roi.w, roi.y + roi.h),
        ],
    );
    assert_eq!(out, expected);
}

#[test]
fn classic_gray_scaled_decode_matches_grok_reduce() {
    let Some(path) = grok_decompress_bin() else {
        return;
    };
    let pixels = gradient_u8(128, 128, 1);
    let jp2 = classic_jp2(&pixels, 128, 128, 1).expect("classic jp2");

    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut out = vec![0_u8; 32 * 32];
    decoder
        .decode_scaled_into(
            &mut j2k::J2kScratchPool::new(),
            &mut out,
            32,
            PixelFormat::Gray8,
            Downscale::Quarter,
        )
        .expect("j2k scaled decode");

    let expected = decode_with_grok(&path, "grok_scaled_gray", &jp2, ".pgm", &["-r", "2"]);
    assert_eq!(out, expected);
}

#[test]
fn classic_rgb_scaled_decode_matches_grok_reduce() {
    let Some(path) = grok_decompress_bin() else {
        return;
    };
    let pixels = gradient_u8(128, 128, 3);
    let jp2 = classic_jp2(&pixels, 128, 128, 3).expect("classic jp2");

    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut out = vec![0_u8; 32 * 32 * 3];
    decoder
        .decode_scaled_into(
            &mut j2k::J2kScratchPool::new(),
            &mut out,
            32 * 3,
            PixelFormat::Rgb8,
            Downscale::Quarter,
        )
        .expect("j2k scaled decode");

    let expected = decode_with_grok(&path, "grok_scaled_rgb", &jp2, ".ppm", &["-r", "2"]);
    assert_eq!(out, expected);
}

#[test]
fn ht_gray_full_decode_matches_grok() {
    let Some(path) = grok_decompress_bin() else {
        return;
    };
    let pixels = gradient_u8(128, 128, 1);
    let jp2 = ht_jp2(&pixels, 128, 128, 1);

    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut out = vec![0_u8; 128 * 128];
    decoder
        .decode_into(&mut out, 128, PixelFormat::Gray8)
        .expect("j2k decode");

    let expected = decode_with_grok(&path, "grok_full_ht_gray", &jp2, ".pgm", &[]);
    assert_eq!(out, expected);
}

#[test]
fn ht_gray_region_decode_matches_grok_area_decode() {
    let Some(path) = grok_decompress_bin() else {
        return;
    };
    let pixels = gradient_u8(128, 128, 1);
    let jp2 = ht_jp2(&pixels, 128, 128, 1);
    let roi = Rect {
        x: 16,
        y: 24,
        w: 48,
        h: 48,
    };

    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut out = vec![0_u8; roi.w as usize * roi.h as usize];
    decoder
        .decode_region_into(
            &mut j2k::J2kScratchPool::new(),
            &mut out,
            roi.w as usize,
            PixelFormat::Gray8,
            roi,
        )
        .expect("j2k region decode");

    let expected = decode_with_grok(
        &path,
        "grok_region_ht_gray",
        &jp2,
        ".pgm",
        &[
            "-d",
            &format!("{},{},{},{}", roi.x, roi.y, roi.x + roi.w, roi.y + roi.h),
        ],
    );
    assert_eq!(out, expected);
}

#[test]
fn ht_gray_scaled_decode_matches_grok_reduce() {
    let Some(path) = grok_decompress_bin() else {
        return;
    };
    let pixels = gradient_u8(128, 128, 1);
    let jp2 = ht_jp2(&pixels, 128, 128, 1);

    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut out = vec![0_u8; 32 * 32];
    decoder
        .decode_scaled_into(
            &mut j2k::J2kScratchPool::new(),
            &mut out,
            32,
            PixelFormat::Gray8,
            Downscale::Quarter,
        )
        .expect("j2k scaled decode");

    let expected = decode_with_grok(&path, "grok_scaled_ht_gray", &jp2, ".pgm", &["-r", "2"]);
    assert_eq!(out, expected);
}

#[test]
fn ht_rgb_full_decode_matches_grok() {
    let Some(path) = grok_decompress_bin() else {
        return;
    };
    let pixels = gradient_u8(128, 128, 3);
    let jp2 = ht_jp2(&pixels, 128, 128, 3);

    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut out = vec![0_u8; 128 * 128 * 3];
    decoder
        .decode_into(&mut out, 128 * 3, PixelFormat::Rgb8)
        .expect("j2k decode");

    let expected = decode_with_grok(&path, "grok_full_ht_rgb", &jp2, ".ppm", &[]);
    assert_eq!(out, expected);
}

#[test]
fn ht_rgb_region_decode_matches_grok_area_decode() {
    let Some(path) = grok_decompress_bin() else {
        return;
    };
    let pixels = gradient_u8(128, 128, 3);
    let jp2 = ht_jp2(&pixels, 128, 128, 3);
    let roi = Rect {
        x: 16,
        y: 24,
        w: 48,
        h: 48,
    };

    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut out = vec![0_u8; roi.w as usize * roi.h as usize * 3];
    decoder
        .decode_region_into(
            &mut j2k::J2kScratchPool::new(),
            &mut out,
            roi.w as usize * 3,
            PixelFormat::Rgb8,
            roi,
        )
        .expect("j2k region decode");

    let expected = decode_with_grok(
        &path,
        "grok_region_ht_rgb",
        &jp2,
        ".ppm",
        &[
            "-d",
            &format!("{},{},{},{}", roi.x, roi.y, roi.x + roi.w, roi.y + roi.h),
        ],
    );
    assert_eq!(out, expected);
}

#[test]
fn ht_rgb_scaled_decode_matches_grok_reduce() {
    let Some(path) = grok_decompress_bin() else {
        return;
    };
    let pixels = gradient_u8(128, 128, 3);
    let jp2 = ht_jp2(&pixels, 128, 128, 3);

    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut out = vec![0_u8; 32 * 32 * 3];
    decoder
        .decode_scaled_into(
            &mut j2k::J2kScratchPool::new(),
            &mut out,
            32 * 3,
            PixelFormat::Rgb8,
            Downscale::Quarter,
        )
        .expect("j2k scaled decode");

    let expected = decode_with_grok(&path, "grok_scaled_ht_rgb", &jp2, ".ppm", &["-r", "2"]);
    assert_eq!(out, expected);
}

fn classic_jp2(pixels: &[u8], width: u32, height: u32, components: u8) -> Option<Vec<u8>> {
    let bin = grok_compress_bin()?;
    let dir = temp_dir();
    let id = next_temp_id();
    let src_path = dir.join(if components == 1 {
        format!("grok_classic_input_{id}.pgm")
    } else {
        format!("grok_classic_input_{id}.ppm")
    });
    let out_path = dir.join(format!("grok_classic_output_{id}.jp2"));
    write_pnm(&src_path, pixels, width, height, usize::from(components)).ok()?;
    let status = Command::new(bin)
        .arg("-i")
        .arg(&src_path)
        .arg("-o")
        .arg(&out_path)
        .arg("-n")
        .arg("4")
        .status()
        .ok()?;
    if !status.success() {
        assert!(
            !require_grok(),
            "J2K_REQUIRE_GROK is set but grk_compress failed"
        );
        return None;
    }
    fs::read(out_path).ok()
}

fn next_temp_id() -> usize {
    static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

fn ht_jp2(pixels: &[u8], width: u32, height: u32, components: u8) -> Vec<u8> {
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 3,
        ..EncodeOptions::default()
    };
    let codestream =
        encode_htj2k(pixels, width, height, components, 8, false, &options).expect("encode ht");
    wrap_codestream_jp2(&codestream, width, height, u16::from(components), 8, 17)
}

fn grok_decompress_bin() -> Option<PathBuf> {
    static GROK: OnceLock<Option<PathBuf>> = OnceLock::new();
    let path = GROK.get_or_init(discover_grok_decompress_bin).clone();
    assert!(
        path.is_some() || !require_grok(),
        "J2K_REQUIRE_GROK is set but grk_decompress was not found"
    );
    path
}

fn grok_compress_bin() -> Option<PathBuf> {
    static GROK: OnceLock<Option<PathBuf>> = OnceLock::new();
    let path = GROK.get_or_init(discover_grok_compress_bin).clone();
    assert!(
        path.is_some() || !require_grok(),
        "J2K_REQUIRE_GROK is set but grk_compress was not found"
    );
    path
}

fn require_grok() -> bool {
    std::env::var_os("J2K_REQUIRE_GROK").is_some()
}

fn discover_grok_decompress_bin() -> Option<PathBuf> {
    std::env::var_os("J2K_GROK_BIN")
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from("/opt/homebrew/bin/grk_decompress")))
        .or_else(|| Some(PathBuf::from("/usr/local/bin/grk_decompress")))
        .filter(|path| path.exists())
}

fn discover_grok_compress_bin() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("J2K_GROK_COMPRESS_BIN")
        .map(PathBuf::from)
        .filter(|path| path.exists())
    {
        return Some(path);
    }
    if let Some(path) = std::env::var_os("J2K_GROK_BIN")
        .map(PathBuf::from)
        .filter(|path| path.exists())
    {
        let sibling = path.with_file_name("grk_compress");
        if sibling.exists() {
            return Some(sibling);
        }
    }
    [
        "/opt/homebrew/bin/grk_compress",
        "/usr/local/bin/grk_compress",
    ]
    .into_iter()
    .map(PathBuf::from)
    .find(|path| path.exists())
}

fn decode_with_grok(
    bin: &Path,
    stem: &str,
    jp2: &[u8],
    output_ext: &str,
    extra_args: &[&str],
) -> Vec<u8> {
    let dir = temp_dir();
    let input_path = dir.join(format!("{stem}.jp2"));
    let output_path = dir.join(format!("{stem}{output_ext}"));
    fs::write(&input_path, jp2).expect("write jp2");
    let mut command = Command::new(bin);
    command.arg("-i").arg(&input_path);
    command.arg("-o").arg(&output_path);
    command.args(extra_args);
    let status = command.status().expect("run grk_decompress");
    assert!(status.success(), "grk_decompress failed");
    read_pnm_pixels(&output_path).expect("read pnm")
}

fn temp_dir() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("j2k-grok-parity-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    })
}
