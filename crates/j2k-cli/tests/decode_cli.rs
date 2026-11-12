// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `opj_decompress`-compatible decoder, run as `j2k decode` and through
//! an `opj_decompress` symlink. References are in-repo fixtures, including
//! the reference decoder's own reduced decodes; `opj_decompress` itself is
//! not needed.

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use j2k_test_support::{OPENJPEG_ORIGIN_23X17_AT_3_5, ORIGIN_REDUCED_RGB, ORIGIN_SOURCE_23X17_RGB};

const OPENJPH: &str = "../j2k-test-support/fixtures/htj2k/openjph_batch";

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("j2k-cli-decode-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create test dir");
    dir
}

/// `opj_decompress` in `dir`: a symlink to the j2k binary, or on Windows,
/// where symlinks need a privilege, a copy of it named `opj_decompress.exe`.
/// The binary recognizes either by its file stem.
#[cfg(unix)]
fn opj_link(dir: &Path) -> PathBuf {
    let link = dir.join("opj_decompress");
    std::os::unix::fs::symlink(env!("CARGO_BIN_EXE_j2k"), &link).expect("link opj_decompress");
    link
}

#[cfg(not(unix))]
fn opj_link(dir: &Path) -> PathBuf {
    let copy = dir.join(format!("opj_decompress{}", std::env::consts::EXE_SUFFIX));
    fs::copy(env!("CARGO_BIN_EXE_j2k"), &copy).expect("copy opj_decompress");
    copy
}

fn run(program: &Path, args: &[&str]) -> Output {
    Command::new(program).args(args).output().expect("run")
}

fn j2k(args: &[&str]) -> Output {
    let mut all = vec!["decode"];
    all.extend_from_slice(args);
    run(Path::new(env!("CARGO_BIN_EXE_j2k")), &all)
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn origin_fixture(dir: &Path) -> PathBuf {
    // No extension: Cantaloupe passes source-cache paths as they are.
    let path = dir.join("origin-image");
    fs::write(&path, OPENJPEG_ORIGIN_23X17_AT_3_5).expect("write fixture");
    path
}

fn rgb(path: &Path) -> (u32, u32, Vec<u8>) {
    let image = image::open(path).expect("read output").to_rgb8();
    (image.width(), image.height(), image.into_raw())
}

/// Rows `y0..y0+h` and columns `x0..x0+w` of a packed RGB image `width` wide.
fn crop(samples: &[u8], width: u32, (x0, y0, w, h): (u32, u32, u32, u32)) -> Vec<u8> {
    samples
        .chunks_exact(width as usize * 3)
        .skip(y0 as usize)
        .take(h as usize)
        .flat_map(|row| &row[x0 as usize * 3..(x0 + w) as usize * 3])
        .copied()
        .collect()
}

#[test]
fn the_opj_decompress_link_decodes_like_openjpeg() {
    let dir = temp_dir("link");
    let opj = opj_link(&dir);
    let input = origin_fixture(&dir);
    let input = input.to_str().expect("utf-8 path");
    for format in ["bmp", "png", "ppm", "tif"] {
        let output = dir.join(format!("full.{format}"));
        let result = run(
            &opj,
            &["-quiet", "-i", input, "-o", output.to_str().expect("path")],
        );
        assert!(
            result.status.success(),
            "{format}: {}",
            text(&result.stderr)
        );
        assert!(
            result.stdout.is_empty() && result.stderr.is_empty(),
            "{format}"
        );
        assert_eq!(
            rgb(&output),
            (23, 17, ORIGIN_SOURCE_23X17_RGB.to_vec()),
            "{format}"
        );
    }
    // OpenJPEG's own reduced decodes, at every reduction.
    for (levels, ((width, height), reference)) in (1..).zip(ORIGIN_REDUCED_RGB) {
        let output = dir.join(format!("r{levels}.bmp"));
        let levels = levels.to_string();
        let result = run(
            &opj,
            &[
                "-quiet",
                "-i",
                input,
                "-r",
                &levels,
                "-o",
                output.to_str().expect("path"),
            ],
        );
        assert!(
            result.status.success(),
            "-r {levels}: {}",
            text(&result.stderr)
        );
        assert_eq!(
            rgb(&output),
            (width, height, reference.to_vec()),
            "-r {levels}"
        );
    }
}

#[test]
fn decode_areas_use_reference_grid_coordinates_and_t800_reduction() {
    let dir = temp_dir("area");
    let input = origin_fixture(&dir);
    let input = input.to_str().expect("utf-8 path");
    // The image occupies x 3..26 and y 5..22 of the reference grid.
    let output = dir.join("area.ppm");
    let result = j2k(&[
        "-quiet",
        "-i",
        input,
        "-d",
        "4,6,20,15",
        "-o",
        output.to_str().expect("path"),
    ]);
    assert!(result.status.success(), "{}", text(&result.stderr));
    assert_eq!(
        rgb(&output),
        (16, 9, crop(ORIGIN_SOURCE_23X17_RGB, 23, (1, 1, 16, 9)))
    );
    // At reduction r the area keeps [ceil(x0 / 2^r), ceil(x1 / 2^r)) of the
    // reduced grid, whose image starts at ceil(origin / 2^r).
    for (levels, ((width, _), reference)) in (1_u32..).zip(ORIGIN_REDUCED_RGB) {
        let scale = 1 << levels;
        let (x, y) = (
            4_u32.div_ceil(scale) - 3_u32.div_ceil(scale),
            6_u32.div_ceil(scale) - 5_u32.div_ceil(scale),
        );
        let (w, h) = (
            20_u32.div_ceil(scale) - 4_u32.div_ceil(scale),
            15_u32.div_ceil(scale) - 6_u32.div_ceil(scale),
        );
        let output = dir.join(format!("area-r{levels}.png"));
        let result = j2k(&[
            "-quiet",
            "-i",
            input,
            "-d",
            "4,6,20,15",
            "-r",
            &levels.to_string(),
            "-o",
            output.to_str().expect("path"),
        ]);
        assert!(result.status.success(), "{}", text(&result.stderr));
        assert_eq!(
            rgb(&output),
            (w, h, crop(reference, width, (x, y, w, h))),
            "-r {levels}"
        );
    }
    // Coordinates before the origin or past the extent are clamped with a warning.
    let output = dir.join("clamped.bmp");
    let result = j2k(&[
        "-i",
        input,
        "-d",
        "0,0,99,99",
        "-o",
        output.to_str().expect("path"),
    ]);
    assert!(result.status.success(), "{}", text(&result.stderr));
    assert!(text(&result.stderr).contains("warning: the decoded area starts at x=0"));
    assert_eq!(rgb(&output), (23, 17, ORIGIN_SOURCE_23X17_RGB.to_vec()));
}

#[test]
fn deep_and_signed_samples_keep_their_precision() {
    let dir = temp_dir("deep");
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join(OPENJPH);
    for (name, signed) in [("gray_u12_53", false), ("gray_s12_53", true)] {
        let input = fixtures.join(format!("{name}.j2c"));
        let oracle = fs::read(fixtures.join(format!("{name}.oracle.raw"))).expect("oracle");
        let expected = oracle
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&pair| {
                let value = i32::from(u16::from_le_bytes(pair));
                if signed {
                    (value << 16 >> 16) + 2048
                } else {
                    value
                }
            })
            .collect::<Vec<_>>();
        let input = input.to_str().expect("path");
        let pgm = dir.join(format!("{name}.pgm"));
        let result = j2k(&["-quiet", "-i", input, "-o", pgm.to_str().expect("path")]);
        assert!(result.status.success(), "{name}: {}", text(&result.stderr));
        let bytes = fs::read(&pgm).expect("pgm");
        assert!(text(&bytes[..16]).contains("4095"), "{name}: maxval");
        let samples = bytes[bytes.len() - expected.len() * 2..]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&pair| i32::from(u16::from_be_bytes(pair)))
            .collect::<Vec<_>>();
        assert_eq!(samples, expected, "{name}: PGM keeps 12-bit samples");
        let png = dir.join(format!("{name}.png"));
        let result = j2k(&["-quiet", "-i", input, "-o", png.to_str().expect("path")]);
        assert!(result.status.success(), "{name}: {}", text(&result.stderr));
        let image = image::open(&png).expect("png").to_luma16().into_raw();
        // OpenJPEG scales unsigned samples by 65535 / 4095 and signed ones by
        // 2^15 / 2^11 before offsetting them by 2^15.
        let scaled = expected
            .iter()
            .map(|&v| u16::try_from(if signed { v * 16 } else { v * 65535 / 4095 }).expect("u16"));
        assert_eq!(
            image,
            scaled.collect::<Vec<_>>(),
            "{name}: PNG scales 12 bits to 16"
        );
        let tif = dir.join(format!("{name}.tif"));
        let result = j2k(&["-quiet", "-i", input, "-o", tif.to_str().expect("path")]);
        assert!(result.status.success(), "{name}: {}", text(&result.stderr));
        let image = image::open(&tif).expect("tif").to_luma16().into_raw();
        let raw = expected.iter().map(|&v| u16::try_from(v).expect("u16"));
        assert_eq!(
            image,
            raw.collect::<Vec<_>>(),
            "{name}: TIFF keeps 12-bit samples"
        );
    }
}

#[test]
fn signed_8_bit_samples_keep_their_values() {
    let dir = temp_dir("signed8");
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join(OPENJPH);
    let input = fixtures.join("gray_s8_53.j2c");
    let input = input.to_str().expect("path");
    let gray = |path: &Path| {
        let image = image::open(path).expect("pgm").to_luma8();
        (image.width(), image.height(), image.into_raw())
    };
    // OpenJPEG's PGM offsets signed gray samples by 2^(precision - 1).
    let oracle = fs::read(fixtures.join("gray_s8_53.oracle.raw")).expect("oracle");
    let expected = oracle
        .iter()
        .map(|&byte| u8::try_from(i16::from(i8::from_le_bytes([byte])) + 128).expect("u8"))
        .collect::<Vec<_>>();
    let full = dir.join("full.pgm");
    let result = j2k(&["-quiet", "-i", input, "-o", full.to_str().expect("path")]);
    assert!(result.status.success(), "{}", text(&result.stderr));
    assert_eq!(gray(&full), (19, 13, expected));
    // A reduced area is the matching part of the reduced image: x [1, 5) and
    // y [1, 4) of the 10x7 image at -r 1.
    let reduced = dir.join("r1.pgm");
    let result = j2k(&[
        "-quiet",
        "-i",
        input,
        "-r",
        "1",
        "-o",
        reduced.to_str().expect("path"),
    ]);
    assert!(result.status.success(), "{}", text(&result.stderr));
    let (width, height, samples) = gray(&reduced);
    assert_eq!((width, height), (10, 7));
    let area = dir.join("r1-area.pgm");
    let result = j2k(&[
        "-quiet",
        "-i",
        input,
        "-d",
        "2,2,9,7",
        "-r",
        "1",
        "-o",
        area.to_str().expect("path"),
    ]);
    assert!(result.status.success(), "{}", text(&result.stderr));
    let window = samples
        .as_chunks::<10>()
        .0
        .iter()
        .skip(1)
        .take(3)
        .flat_map(|row| &row[1..5])
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(gray(&area), (4, 3, window));
}

/// Image servers name `/dev/stdout` as the output file, which only Unix has.
#[cfg(unix)]
#[test]
fn an_output_linked_to_stdout_carries_only_the_image() {
    use std::process::Stdio;

    let dir = temp_dir("stdout");
    let opj = opj_link(&dir);
    let input = origin_fixture(&dir);
    let link = dir.join("out.bmp");
    std::os::unix::fs::symlink("/dev/stdout", &link).expect("link stdout");
    for quiet in [true, false] {
        let mut command = Command::new(&opj);
        if quiet {
            command.arg("-quiet");
        }
        let result = command
            .args([
                "-i",
                input.to_str().expect("path"),
                "-d",
                "4,6,20,15",
                "-r",
                "1",
                "-o",
            ])
            .arg(&link)
            .stdout(Stdio::piped())
            .output()
            .expect("run");
        assert!(result.status.success(), "{}", text(&result.stderr));
        let image = image::load_from_memory_with_format(&result.stdout, image::ImageFormat::Bmp)
            .expect("stdout is one BMP");
        assert_eq!((image.width(), image.height()), (8, 5));
        assert_eq!(result.stderr.is_empty(), quiet, "{}", text(&result.stderr));
    }
}

#[test]
fn threads_and_force_rgb_do_not_change_samples() {
    let dir = temp_dir("options");
    let input = origin_fixture(&dir);
    let input = input.to_str().expect("path");
    let reference = dir.join("one.ppm");
    assert!(j2k(&[
        "-quiet",
        "-i",
        input,
        "-o",
        reference.to_str().expect("path")
    ])
    .status
    .success());
    for threads in ["1", "3", "ALL_CPUS"] {
        let output = dir.join(format!("t{threads}.ppm"));
        let result = j2k(&[
            "-quiet",
            "-threads",
            threads,
            "-i",
            input,
            "-o",
            output.to_str().expect("path"),
        ]);
        assert!(result.status.success(), "{}", text(&result.stderr));
        assert_eq!(
            fs::read(&output).expect("out"),
            fs::read(&reference).expect("ref")
        );
    }
    let gray = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(OPENJPH)
        .join("gray_u8_53.j2c");
    let output = dir.join("gray.ppm");
    let result = j2k(&[
        "-quiet",
        "-force-rgb",
        "-i",
        gray.to_str().expect("path"),
        "-o",
        output.to_str().expect("path"),
    ]);
    assert!(result.status.success(), "{}", text(&result.stderr));
    let image = image::open(&output).expect("ppm");
    assert_eq!(image.color(), image::ColorType::Rgb8);
    let pixels = image.to_rgb8().into_raw();
    assert!(pixels
        .as_chunks::<3>()
        .0
        .iter()
        .all(|pixel| pixel[0] == pixel[1] && pixel[1] == pixel[2]));
}

#[test]
fn failures_exit_nonzero_with_opj_decompress_messages() {
    let dir = temp_dir("errors");
    let opj = opj_link(&dir);
    let input = origin_fixture(&dir);
    let input = input.to_str().expect("path");
    let junk = dir.join("junk.jp2");
    fs::write(&junk, b"not a JPEG 2000 file at all").expect("junk");
    let deep = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../j2k-test-support/fixtures/openjpeg_origin/gray29_23x17_origin_3_3.j2k");
    let out = dir.join("out.png");
    let out = out.to_str().expect("path");
    for (args, needle) in [
        (vec![], "Required parameters are missing"),
        (
            vec!["-i", junk.to_str().expect("path"), "-o", out],
            "Unknown input file format",
        ),
        (
            vec!["-i", "/nonexistent/image.jp2", "-o", out],
            "infile cannot be read",
        ),
        (
            vec!["-i", input, "-o", out, "-r", "4"],
            "reduce factor 4 exceeds",
        ),
        (
            vec!["-i", input, "-o", out, "-d", "40,40,50,50"],
            "after the image area",
        ),
        (vec!["-i", input, "-o", out, "-l", "1"], "-l 1"),
        (vec!["-i", input, "-o", out, "-t", "0"], "-t"),
        (vec!["-i", input, "-o", out, "-c", "0"], "-c"),
        (vec!["-i", input, "-o", out, "-upsample"], "-upsample"),
        (
            vec!["-i", input, "-o", out, "-allow-partial"],
            "-allow-partial",
        ),
        (vec!["-i", input, "-o", "out.raw"], ".raw"),
        (
            vec!["-i", deep.to_str().expect("path"), "-o", out],
            "precision 29 is larger than 16",
        ),
    ] {
        let result = run(&opj, &args);
        assert_eq!(
            result.status.code(),
            Some(1),
            "{args:?}: {}",
            text(&result.stderr)
        );
        assert!(
            text(&result.stderr).contains(needle),
            "{args:?}: {}",
            text(&result.stderr)
        );
        assert!(result.stdout.is_empty(), "{args:?}");
    }
    assert!(!Path::new(out).exists(), "no output after failures");
    // j2k decode keeps j2k's contract: 2 for arguments, 1 at run time.
    assert_eq!(
        j2k(&["-i", input, "-o", out, "-t", "0"]).status.code(),
        Some(2)
    );
    assert_eq!(
        j2k(&["-i", input, "-o", out, "-r", "4"]).status.code(),
        Some(1)
    );
}

#[test]
fn help_follows_each_contract() {
    let dir = temp_dir("help");
    let opj = opj_link(&dir);
    let result = run(&opj, &["-h"]);
    assert_eq!(result.status.code(), Some(1), "opj_decompress -h exits 1");
    let help = text(&result.stdout);
    assert!(
        help.contains("-i <file>") && help.contains("-threads"),
        "{help}"
    );
    // Cantaloupe reads a "vX.Y.Z" from this output and stops passing -quiet below 2.2.
    assert!(!help.contains(" v0") && !help.contains(" v1"), "{help}");
    let result = j2k(&["-h"]);
    assert_eq!(result.status.code(), Some(0));
    assert!(result.stdout.is_empty() && text(&result.stderr).contains("-d <x0,y0,x1,y1>"));
}
