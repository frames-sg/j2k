// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::J2kDecoder;
use j2k_core::{Downscale, PixelFormat, Rect};
use j2k_test_support::{fnv1a64_hex, gradient_u8, write_pnm};
use std::{
    fs,
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
    process::Command,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Barrier, OnceLock,
    },
};

const APERIO_TILE_OFFSET: u64 = 12_696_074;
const APERIO_TILE_BYTES: usize = 23_080;
const APERIO_TILE_FNV1A64: &str = "36dc1f181d439cd5";

#[test]
fn aperio_lossy_wsi_tile_matches_openjpeg() {
    let Some(path) = std::env::var_os("J2K_WSI_SVS_PATH") else {
        eprintln!("J2K_WSI_SVS_PATH is unset; skipping external Aperio parity fixture");
        return;
    };
    let mut file = fs::File::open(path).expect("open Aperio JP2K slide");
    file.seek(SeekFrom::Start(APERIO_TILE_OFFSET))
        .expect("seek to level-zero tile (14, 16)");
    let mut codestream = vec![0_u8; APERIO_TILE_BYTES];
    file.read_exact(&mut codestream)
        .expect("read level-zero tile (14, 16)");
    assert_eq!(fnv1a64_hex(&codestream), APERIO_TILE_FNV1A64);

    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let ours = decoder
        .decode_native_components()
        .expect("j2k native component decode");
    let reference = j2k_compare::openjpeg::decode_components(&codestream)
        .expect("OpenJPEG native component decode");

    assert_eq!(ours.dimensions(), reference.dimensions);
    assert_eq!(ours.planes().len(), reference.components.len());
    for (index, (actual, expected)) in ours.planes().iter().zip(&reference.components).enumerate() {
        let sampling = actual.sampling();
        assert_eq!(
            (u32::from(sampling.0), u32::from(sampling.1)),
            expected.sampling
        );
        assert_eq!(actual.bit_depth(), expected.bit_depth);
        assert_eq!(actual.signed(), expected.signed);
        assert_eq!(actual.bytes_per_sample(), 1);
        assert_eq!(actual.dimensions(), ours.dimensions());
        let actual_dimensions = actual.dimensions();
        let mismatch =
            actual
                .data()
                .iter()
                .enumerate()
                .find_map(|(sample_index, actual_sample)| {
                    let x = sample_index % actual_dimensions.0 as usize;
                    let y = sample_index / actual_dimensions.0 as usize;
                    let expected_index = (y / sampling.1 as usize) * expected.dimensions.0 as usize
                        + x / sampling.0 as usize;
                    let expected_sample = expected.samples[expected_index];
                    (i32::from(*actual_sample) != expected_sample).then_some((
                        x,
                        y,
                        *actual_sample,
                        expected_sample,
                    ))
                });
        if let Some((x, y, actual_sample, expected_sample)) = mismatch {
            panic!(
                "component {index} differs at ({x}, {y}): native={actual_sample}, OpenJPEG={expected_sample}"
            );
        }
    }

    let ours_rgb = j2k_rgb(&codestream);
    let reference_rgb =
        j2k_compare::openjpeg::decode_rgb(&codestream).expect("OpenJPEG RGB decode");
    assert_eq!(ours_rgb.len(), reference_rgb.len());
    if let Some((index, (actual, expected))) = ours_rgb
        .iter()
        .zip(&reference_rgb)
        .enumerate()
        .find(|(_, (actual, expected))| actual != expected)
    {
        panic!("RGB output differs at byte {index}: native={actual}, OpenJPEG={expected}");
    }
    assert_eq!(
        fnv1a64_hex(&ours_rgb),
        "ce5021d9b3a33766",
        "unexpected exact RGB checksum"
    );
}

#[test]
fn openjpeg_in_process_matches_j2k_rgb_fixture() {
    let Some(input) = bench_fixture_rgb() else {
        return;
    };
    let ours = j2k_rgb(&input);
    let theirs = j2k_compare::openjpeg::decode_rgb(&input).expect("openjpeg");
    assert_eq!(ours, theirs);
}

#[test]
fn openjpeg_in_process_region_matches_j2k_rgb_fixture() {
    let Some(input) = bench_fixture_rgb() else {
        return;
    };
    let roi = Rect {
        x: 16,
        y: 24,
        w: 64,
        h: 64,
    };
    let ours = j2k_rgb_region(&input, roi);
    let theirs = j2k_compare::openjpeg::decode_rgb_region(&input, roi).expect("openjpeg");
    assert_eq!(ours, theirs);
}

#[test]
fn openjpeg_in_process_scaled_matches_j2k_rgb_fixture() {
    let Some(input) = bench_fixture_rgb() else {
        return;
    };
    let ours = j2k_rgb_scaled_q4(&input);
    let theirs = j2k_compare::openjpeg::decode_rgb_scaled(&input, 2).expect("openjpeg");
    assert_eq!(ours, theirs);
}

#[test]
fn openjpeg_in_process_region_scaled_matches_j2k_rgb_fixture() {
    let Some(input) = bench_fixture_rgb() else {
        return;
    };
    let roi = Rect {
        x: 16,
        y: 24,
        w: 64,
        h: 64,
    };
    let ours = j2k_rgb_region_scaled_q4(&input, roi);
    let theirs = j2k_compare::openjpeg::decode_rgb_region_scaled(&input, roi, 2).expect("openjpeg");
    assert_eq!(ours, theirs);
}

#[test]
fn grok_in_process_matches_j2k_rgb_fixture() {
    if !j2k_compare::grok::is_available() {
        assert!(
            !require_grok(),
            "J2K_REQUIRE_GROK is set but in-process Grok is unavailable"
        );
        return;
    }
    let Some(input) = bench_fixture_rgb() else {
        return;
    };
    let ours = j2k_rgb(&input);
    let theirs = j2k_compare::grok::decode_rgb(&input).expect("grok");
    assert_eq!(ours, theirs);
}

#[test]
fn grok_in_process_region_matches_j2k_rgb_fixture() {
    if !j2k_compare::grok::is_available() {
        assert!(
            !require_grok(),
            "J2K_REQUIRE_GROK is set but in-process Grok is unavailable"
        );
        return;
    }
    let Some(input) = bench_fixture_rgb() else {
        return;
    };
    let roi = Rect {
        x: 16,
        y: 24,
        w: 64,
        h: 64,
    };
    let ours = j2k_rgb_region(&input, roi);
    let theirs = j2k_compare::grok::decode_rgb_region(&input, roi).expect("grok");
    assert_eq!(ours, theirs);
}

#[test]
fn grok_in_process_scaled_matches_j2k_rgb_fixture() {
    if !j2k_compare::grok::is_available() {
        assert!(
            !require_grok(),
            "J2K_REQUIRE_GROK is set but in-process Grok is unavailable"
        );
        return;
    }
    let Some(input) = bench_fixture_rgb() else {
        return;
    };
    let ours = j2k_rgb_scaled_q4(&input);
    let theirs = j2k_compare::grok::decode_rgb_scaled(&input, 2).expect("grok");
    assert_eq!(ours, theirs);
}

#[test]
fn grok_in_process_region_scaled_matches_j2k_rgb_fixture() {
    if !j2k_compare::grok::is_available() {
        assert!(
            !require_grok(),
            "J2K_REQUIRE_GROK is set but in-process Grok is unavailable"
        );
        return;
    }
    let Some(input) = bench_fixture_rgb() else {
        return;
    };
    let roi = Rect {
        x: 16,
        y: 24,
        w: 64,
        h: 64,
    };
    let ours = j2k_rgb_region_scaled_q4(&input, roi);
    let theirs = j2k_compare::grok::decode_rgb_region_scaled(&input, roi, 2).expect("grok");
    assert_eq!(ours, theirs);
}

#[test]
fn grok_concurrent_decodes_share_the_initialized_runtime() {
    if !j2k_compare::grok::is_available() {
        assert!(
            !require_grok(),
            "J2K_REQUIRE_GROK is set but in-process Grok is unavailable"
        );
        return;
    }
    let Some(input) = bench_fixture_rgb() else {
        return;
    };
    let expected = j2k_rgb(&input);
    let workers = 8;
    let barrier = Barrier::new(workers);
    std::thread::scope(|scope| {
        let handles = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    barrier.wait();
                    j2k_compare::grok::decode_rgb(&input)
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            let decoded = handle
                .join()
                .expect("Grok decode worker must not panic")
                .expect("concurrent Grok decode");
            assert_eq!(decoded, expected);
        }
    });
}

fn bench_fixture_rgb() -> Option<Vec<u8>> {
    let pixels = gradient_u8(128, 128, 3);
    openjpeg_encode_jp2("in_process_parity_rgb", &pixels, 128, 128)
}

fn j2k_rgb(bytes: &[u8]) -> Vec<u8> {
    let mut decoder = J2kDecoder::new(bytes).expect("decoder");
    let dims = decoder.info().dimensions;
    let mut out = vec![0_u8; dims.0 as usize * dims.1 as usize * 3];
    decoder
        .decode_into(&mut out, dims.0 as usize * 3, PixelFormat::Rgb8)
        .expect("decode");
    out
}

fn j2k_rgb_region(bytes: &[u8], roi: Rect) -> Vec<u8> {
    let mut decoder = J2kDecoder::new(bytes).expect("decoder");
    let mut out = vec![0_u8; roi.w as usize * roi.h as usize * 3];
    decoder
        .decode_region_into(
            &mut j2k::J2kScratchPool::new(),
            &mut out,
            roi.w as usize * 3,
            PixelFormat::Rgb8,
            roi,
        )
        .expect("region decode");
    out
}

fn j2k_rgb_scaled_q4(bytes: &[u8]) -> Vec<u8> {
    let mut decoder = J2kDecoder::new(bytes).expect("decoder");
    let dims = decoder.info().dimensions;
    let scaled = (dims.0.div_ceil(4), dims.1.div_ceil(4));
    let mut out = vec![0_u8; scaled.0 as usize * scaled.1 as usize * 3];
    decoder
        .decode_scaled_into(
            &mut j2k::J2kScratchPool::new(),
            &mut out,
            scaled.0 as usize * 3,
            PixelFormat::Rgb8,
            Downscale::Quarter,
        )
        .expect("scaled decode");
    out
}

fn j2k_rgb_region_scaled_q4(bytes: &[u8], roi: Rect) -> Vec<u8> {
    let mut decoder = J2kDecoder::new(bytes).expect("decoder");
    let scaled = roi.scaled_covering(Downscale::Quarter);
    let mut out = vec![0_u8; scaled.w as usize * scaled.h as usize * 3];
    decoder
        .decode_region_scaled_into(
            &mut j2k::J2kScratchPool::new(),
            &mut out,
            scaled.w as usize * 3,
            PixelFormat::Rgb8,
            roi,
            Downscale::Quarter,
        )
        .expect("region scaled decode");
    out
}

fn openjpeg_encode_jp2(name: &str, pixels: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
    let Some(bin) = openjpeg_compress_bin() else {
        assert!(
            !require_openjpeg(),
            "J2K_REQUIRE_OPENJPEG is set but opj_compress was not found"
        );
        return None;
    };
    let dir = openjpeg_temp_dir();
    let unique = next_temp_suffix();
    let src_path = dir.join(format!("{name}-{unique}.ppm"));
    let out_path = dir.join(format!("{name}-{unique}.jp2"));
    write_pnm(&src_path, pixels, width, height, 3).ok()?;
    let status = Command::new(bin)
        .arg("-i")
        .arg(&src_path)
        .arg("-o")
        .arg(&out_path)
        .status()
        .ok()?;
    if !status.success() {
        assert!(
            !require_openjpeg(),
            "J2K_REQUIRE_OPENJPEG is set but opj_compress failed"
        );
        return None;
    }
    fs::read(out_path).ok()
}

fn next_temp_suffix() -> usize {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

fn openjpeg_compress_bin() -> Option<PathBuf> {
    static OPENJPEG_COMPRESS: OnceLock<Option<PathBuf>> = OnceLock::new();
    OPENJPEG_COMPRESS
        .get_or_init(|| {
            if let Some(path) = std::env::var_os("J2K_OPENJPEG_COMPRESS_BIN") {
                let path = PathBuf::from(path);
                if path.exists() {
                    return Some(path);
                }
            }
            let default = PathBuf::from("/opt/homebrew/bin/opj_compress");
            default.exists().then_some(default)
        })
        .clone()
}

fn require_openjpeg() -> bool {
    std::env::var_os("J2K_REQUIRE_OPENJPEG").is_some()
}

fn require_grok() -> bool {
    std::env::var_os("J2K_REQUIRE_GROK").is_some()
}

fn openjpeg_temp_dir() -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::temp_dir().join("j2k-openjpeg-tests");
        fs::create_dir_all(&dir).expect("create openjpeg temp dir");
        dir
    })
    .clone()
}
