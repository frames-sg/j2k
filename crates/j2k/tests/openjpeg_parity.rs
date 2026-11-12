// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    encode_j2k_lossless, EncodeBackendPreference, J2kDecoder, J2kLosslessEncodeOptions,
    J2kLosslessSamples,
};
use j2k_core::{Downscale, PixelFormat, Rect};
use j2k_test_support::{gradient_u8, read_pnm_pixels, write_pnm};
#[path = "support/deep_samples.rs"]
mod deep_samples;
use deep_samples::{
    deep_sample_bytes, deep_samples_from_bytes, hostile_deep_samples, DEEP_HEIGHT, DEEP_PRECISIONS,
    DEEP_WIDTH,
};

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::OnceLock,
};

#[test]
fn classic_gray_full_decode_matches_openjpeg() {
    let Some(paths) = OpenJpegPaths::discover() else {
        return;
    };
    let pixels = gradient_u8(128, 128, 1);
    let jp2 = encode_with_openjpeg(&paths, "parity_full_gray", &pixels, 128, 128, 1);

    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut out = vec![0_u8; 128 * 128];
    decoder
        .decode_into(&mut out, 128, PixelFormat::Gray8)
        .expect("j2k decode");

    let expected = decode_with_openjpeg(&paths, "parity_full_gray", &jp2, ".pgm", &[]);
    assert_eq!(out, expected);
}

#[test]
fn classic_rgb_full_decode_matches_openjpeg() {
    let Some(paths) = OpenJpegPaths::discover() else {
        return;
    };
    let pixels = gradient_u8(128, 128, 3);
    let jp2 = encode_with_openjpeg(&paths, "parity_full_rgb", &pixels, 128, 128, 3);

    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut out = vec![0_u8; 128 * 128 * 3];
    decoder
        .decode_into(&mut out, 128 * 3, PixelFormat::Rgb8)
        .expect("j2k decode");

    let expected = decode_with_openjpeg(&paths, "parity_full_rgb", &jp2, ".ppm", &[]);
    assert_eq!(out, expected);
}

#[test]
fn classic_gray_region_decode_matches_openjpeg_area_decode() {
    let Some(paths) = OpenJpegPaths::discover() else {
        return;
    };
    let pixels = gradient_u8(128, 128, 1);
    let jp2 = encode_with_openjpeg(&paths, "parity_region_gray", &pixels, 128, 128, 1);
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

    let expected = decode_with_openjpeg(
        &paths,
        "parity_region_gray",
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
fn classic_rgb_region_decode_matches_openjpeg_area_decode() {
    let Some(paths) = OpenJpegPaths::discover() else {
        return;
    };
    let pixels = gradient_u8(128, 128, 3);
    let jp2 = encode_with_openjpeg(&paths, "parity_region_rgb", &pixels, 128, 128, 3);
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

    let expected = decode_with_openjpeg(
        &paths,
        "parity_region_rgb",
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
fn classic_gray_scaled_decode_matches_openjpeg_reduce() {
    let Some(paths) = OpenJpegPaths::discover() else {
        return;
    };
    let pixels = gradient_u8(128, 128, 1);
    let jp2 = encode_with_openjpeg(&paths, "parity_scaled_gray", &pixels, 128, 128, 1);

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

    let expected = decode_with_openjpeg(&paths, "parity_scaled_gray", &jp2, ".pgm", &["-r", "2"]);
    assert_eq!(out, expected);
}

#[test]
fn classic_rgb_scaled_decode_matches_openjpeg_reduce() {
    let Some(paths) = OpenJpegPaths::discover() else {
        return;
    };
    let pixels = gradient_u8(128, 128, 3);
    let jp2 = encode_with_openjpeg(&paths, "parity_scaled_rgb", &pixels, 128, 128, 3);

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

    let expected = decode_with_openjpeg(&paths, "parity_scaled_rgb", &jp2, ".ppm", &["-r", "2"]);
    assert_eq!(out, expected);
}

#[test]
fn classic_lossless_encode_decodes_with_openjpeg() {
    let Some(paths) = OpenJpegPaths::discover() else {
        return;
    };
    let pixels = gradient_u8(64, 64, 3);
    let samples = J2kLosslessSamples::new(&pixels, 64, 64, 3, 8, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions::default().with_backend(EncodeBackendPreference::CpuOnly),
    )
    .expect("j2k encode");

    let decoded = decode_j2k_with_openjpeg(&paths, "j2k_encode_rgb", &encoded.codestream);
    assert_eq!(decoded, pixels);
}

/// One unsigned component plane as an `OpenJPEG` PGX file: a text header,
/// then big-endian samples of four bytes above 16 bits.
fn write_pgx(path: &Path, plane: &[i64], depth: u8) {
    assert!(depth > 16, "these tests write only four-byte PGX samples");
    let mut bytes = format!("PG ML + {depth} {DEEP_WIDTH} {DEEP_HEIGHT}\n").into_bytes();
    for &sample in plane {
        let sample = u32::try_from(sample).expect("unsigned sample fits 32 bits");
        bytes.extend_from_slice(&sample.to_be_bytes());
    }
    fs::write(path, bytes).expect("write pgx");
}

/// The samples of a PGX file `OpenJPEG` wrote.
fn read_pgx(path: &Path, depth: u8, signed: bool) -> Vec<i64> {
    let bytes = fs::read(path).expect("read pgx");
    let header_end = bytes
        .iter()
        .position(|&byte| byte == b'\n')
        .expect("pgx header");
    let header = std::str::from_utf8(&bytes[..header_end]).expect("pgx header text");
    let fields: Vec<&str> = header.split_whitespace().collect();
    assert_eq!(fields[0], "PG");
    assert_eq!(fields[1], "ML", "big-endian PGX");
    let precision: u8 = fields[fields.len() - 3]
        .trim_start_matches(['+', '-'])
        .parse()
        .unwrap();
    assert_eq!(precision, depth);
    assert_eq!(header.contains('-'), signed);
    let shift = 64 - u32::from(depth);
    let (samples, rest) = bytes[header_end + 1..].as_chunks::<4>();
    assert!(rest.is_empty(), "whole four-byte PGX samples");
    samples
        .iter()
        .map(|&sample| {
            let raw = u64::from(u32::from_be_bytes(sample)) << shift;
            if signed {
                raw.cast_signed() >> shift
            } else {
                (raw >> shift).cast_signed()
            }
        })
        .collect()
}

/// The samples `OpenJPEG` decodes `codestream` to, interleaved.
fn decode_deep_with_openjpeg(
    paths: &OpenJpegPaths,
    stem: &str,
    codestream: &[u8],
    depth: u8,
    signed: bool,
    components: usize,
) -> Vec<i64> {
    let dir = temp_dir();
    let input_path = dir.join(format!("{stem}.j2k"));
    fs::write(&input_path, codestream).expect("write j2k");
    let status = Command::new(&paths.decompress)
        .arg("-i")
        .arg(&input_path)
        .arg("-o")
        .arg(dir.join(format!("{stem}.pgx")))
        .arg("-quiet")
        .status()
        .expect("run opj_decompress");
    assert!(status.success(), "opj_decompress failed");
    let planes: Vec<Vec<i64>> = (0..components)
        .map(|component| {
            let numbered = dir.join(format!("{stem}_{component}.pgx"));
            let path = if numbered.exists() {
                numbered
            } else {
                dir.join(format!("{stem}.pgx"))
            };
            read_pgx(&path, depth, signed)
        })
        .collect();
    (0..planes[0].len())
        .flat_map(|at| planes.iter().map(move |plane| plane[at]))
        .collect()
}

#[test]
fn deep_lossless_encode_decodes_exactly_with_openjpeg() {
    let Some(paths) = OpenJpegPaths::discover() else {
        return;
    };
    for depth in DEEP_PRECISIONS {
        for (signed, components) in [(false, 1), (false, 3), (true, 1)] {
            let source = hostile_deep_samples(depth, signed, components);
            let pixels = deep_sample_bytes(&source, depth);
            let samples = J2kLosslessSamples::new(
                &pixels,
                DEEP_WIDTH,
                DEEP_HEIGHT,
                u16::try_from(components).unwrap(),
                depth,
                signed,
            )
            .unwrap();
            let encoded = encode_j2k_lossless(
                samples,
                &J2kLosslessEncodeOptions::default().with_backend(EncodeBackendPreference::CpuOnly),
            )
            .unwrap_or_else(|error| panic!("{depth}-bit x{components} encode: {error}"));

            let stem = format!("deep_encode_{depth}_{components}_{signed}");
            let decoded = decode_deep_with_openjpeg(
                &paths,
                &stem,
                &encoded.codestream,
                depth,
                signed,
                components,
            );
            assert!(
                decoded == source,
                "{depth}-bit x{components} signed={signed}: OpenJPEG decodes other samples"
            );
        }
    }
}

#[test]
fn openjpeg_deep_lossless_encode_decodes_as_openjpeg_decodes_it() {
    let Some(paths) = OpenJpegPaths::discover() else {
        return;
    };
    // Unsigned only: `opj_compress` reads signed PGX samples deeper than 16
    // bits as 16-bit ones. The reference is OpenJPEG's own decode of its
    // codestream, not the source: on this input at 24 bits its encoder
    // overflows, and its codestream no longer holds the source.
    for depth in DEEP_PRECISIONS {
        let source = hostile_deep_samples(depth, false, 1);
        let dir = temp_dir();
        let stem = format!("deep_decode_{depth}");
        let pgx = dir.join(format!("{stem}.pgx"));
        let j2k = dir.join(format!("{stem}.j2k"));
        write_pgx(&pgx, &source, depth);
        let status = Command::new(&paths.compress)
            .arg("-i")
            .arg(&pgx)
            .arg("-o")
            .arg(&j2k)
            .status()
            .expect("run opj_compress");
        assert!(status.success(), "opj_compress failed");
        let codestream = fs::read(&j2k).expect("read j2k");
        let expected = decode_deep_with_openjpeg(
            &paths,
            &format!("{stem}_reference"),
            &codestream,
            depth,
            false,
            1,
        );
        if depth < 24 {
            assert!(expected == source, "{depth}-bit: OpenJPEG round trip");
        }

        let image = j2k_native::Image::new(&codestream, &j2k_native::DecodeSettings::default())
            .expect("parse");
        let decoded = image.decode_native().expect("native decode");
        assert_eq!(decoded.bit_depth, depth);
        assert!(
            deep_samples_from_bytes(&decoded.data, depth, false) == expected,
            "{depth}-bit: other samples than OpenJPEG decodes"
        );
    }
}

#[derive(Clone)]
struct OpenJpegPaths {
    compress: PathBuf,
    decompress: PathBuf,
}

impl OpenJpegPaths {
    fn discover() -> Option<Self> {
        let paths = discover_openjpeg_paths();
        assert!(
            paths.is_some() || !require_openjpeg(),
            "J2K_REQUIRE_OPENJPEG is set but opj_compress/opj_decompress were not found"
        );
        paths
    }
}

fn discover_openjpeg_paths() -> Option<OpenJpegPaths> {
    let compress = std::env::var_os("J2K_OPENJPEG_COMPRESS_BIN")
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from("/opt/homebrew/bin/opj_compress")))
        .filter(|path| path.exists())?;
    let decompress = std::env::var_os("J2K_OPENJPEG_BIN")
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from("/opt/homebrew/bin/opj_decompress")))
        .filter(|path| path.exists())?;
    Some(OpenJpegPaths {
        compress,
        decompress,
    })
}

fn require_openjpeg() -> bool {
    std::env::var_os("J2K_REQUIRE_OPENJPEG").is_some()
}

fn encode_with_openjpeg(
    paths: &OpenJpegPaths,
    stem: &str,
    pixels: &[u8],
    width: u32,
    height: u32,
    channels: usize,
) -> Vec<u8> {
    let dir = temp_dir();
    let src_path = dir.join(if channels == 1 {
        format!("{stem}.pgm")
    } else {
        format!("{stem}.ppm")
    });
    let out_path = dir.join(format!("{stem}.jp2"));
    write_pnm(&src_path, pixels, width, height, channels).expect("write pnm");
    let status = Command::new(&paths.compress)
        .arg("-i")
        .arg(&src_path)
        .arg("-o")
        .arg(&out_path)
        .status()
        .expect("run opj_compress");
    assert!(status.success(), "opj_compress failed");
    fs::read(out_path).expect("read jp2")
}

fn decode_with_openjpeg(
    paths: &OpenJpegPaths,
    stem: &str,
    jp2: &[u8],
    output_ext: &str,
    extra_args: &[&str],
) -> Vec<u8> {
    let dir = temp_dir();
    let input_path = dir.join(format!("{stem}.jp2"));
    let output_path = dir.join(format!("{stem}{output_ext}"));
    fs::write(&input_path, jp2).expect("write jp2");
    let mut command = Command::new(&paths.decompress);
    command.arg("-i").arg(&input_path);
    command.arg("-o").arg(&output_path);
    command.arg("-quiet");
    command.args(extra_args);
    let status = command.status().expect("run opj_decompress");
    assert!(status.success(), "opj_decompress failed");
    read_pnm_pixels(&output_path).expect("read pnm")
}

fn decode_j2k_with_openjpeg(paths: &OpenJpegPaths, stem: &str, codestream: &[u8]) -> Vec<u8> {
    let dir = temp_dir();
    let input_path = dir.join(format!("{stem}.j2k"));
    let output_path = dir.join(format!("{stem}.ppm"));
    fs::write(&input_path, codestream).expect("write j2k");
    let status = Command::new(&paths.decompress)
        .arg("-i")
        .arg(&input_path)
        .arg("-o")
        .arg(&output_path)
        .arg("-quiet")
        .status()
        .expect("run opj_decompress");
    assert!(status.success(), "opj_decompress failed");
    read_pnm_pixels(&output_path).expect("read pnm")
}

fn temp_dir() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("j2k-parity-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    })
}
