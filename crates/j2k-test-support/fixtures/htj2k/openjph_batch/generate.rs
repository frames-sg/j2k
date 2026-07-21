// SPDX-License-Identifier: MIT OR Apache-2.0

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const WIDTH: u32 = 19;
const HEIGHT: u32 = 13;

#[derive(Clone, Copy)]
struct FixtureSpec {
    components: u32,
    precision: u8,
    signed: bool,
}

fn sample(spec: FixtureSpec, x: u32, y: u32, component: u32) -> i32 {
    let modulus = 1_i32 << spec.precision;
    let value = i32::try_from(
        (x * 37 + y * 73 + component * 109 + x * y * 3 + component * y * 11)
            % u32::try_from(modulus).expect("fixture modulus is positive"),
    )
    .expect("fixture sample fits i32");
    if spec.signed {
        value - modulus / 2
    } else {
        value
    }
}

fn irreversible_sample(spec: FixtureSpec, x: u32, y: u32, component: u32) -> i32 {
    let value = i32::try_from(
        (x * 37 + y * 73 + component * 109 + x * y * 3 + component * y * 11) % 256,
    )
    .expect("fixture sample fits i32");
    if spec.signed {
        value - 128
    } else {
        value
    }
}

fn append_sample(bytes: &mut Vec<u8>, sample: i32, precision: u8, signed: bool) {
    if precision <= 8 {
        if signed {
            bytes.push(i8::try_from(sample).expect("signed 8-bit fixture sample") as u8);
        } else {
            bytes.push(u8::try_from(sample).expect("unsigned 8-bit fixture sample"));
        }
    } else if signed {
        bytes.extend_from_slice(
            &i16::try_from(sample)
                .expect("signed 16-bit fixture sample")
                .to_le_bytes(),
        );
    } else {
        bytes.extend_from_slice(
            &u16::try_from(sample)
                .expect("unsigned 16-bit fixture sample")
                .to_le_bytes(),
        );
    }
}

fn planar_source_bytes(spec: FixtureSpec) -> Vec<u8> {
    let bytes_per_sample = if spec.precision <= 8 { 1 } else { 2 };
    let mut bytes = Vec::with_capacity(
        WIDTH as usize * HEIGHT as usize * spec.components as usize * bytes_per_sample,
    );
    for component in 0..spec.components {
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let value = sample(spec, x, y, component);
                if spec.components == 3 && spec.signed && (9..16).contains(&spec.precision) {
                    // OpenJPH's YUV reader loads 9-16-bit samples as unsigned
                    // containers even when the codestream component is signed.
                    // Store the precision-bit two's-complement code instead of
                    // a sign-extended i16 container so the independent encoder
                    // receives the intended signed sample modulo 2^precision.
                    let mask = (1_u16 << spec.precision) - 1;
                    bytes.extend_from_slice(&((value as u16) & mask).to_le_bytes());
                } else {
                    append_sample(&mut bytes, value, spec.precision, spec.signed);
                }
            }
        }
    }
    bytes
}

fn pnm_source_bytes(spec: FixtureSpec) -> Vec<u8> {
    assert!(!spec.signed);
    let magic = if spec.components == 1 { "P5" } else { "P6" };
    let max_value = (1_u32 << spec.precision) - 1;
    let mut bytes = format!("{magic}\n{WIDTH} {HEIGHT}\n{max_value}\n").into_bytes();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            for component in 0..spec.components {
                let sample = u16::try_from(sample(spec, x, y, component))
                    .expect("unsigned PNM fixture sample");
                if spec.precision <= 8 {
                    bytes.push(u8::try_from(sample).expect("unsigned 8-bit PNM sample"));
                } else {
                    bytes.extend_from_slice(&sample.to_be_bytes());
                }
            }
        }
    }
    bytes
}

fn irreversible_source_bytes(spec: FixtureSpec) -> Vec<u8> {
    if spec.signed {
        let mut bytes = Vec::new();
        for component in 0..spec.components {
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    append_sample(
                        &mut bytes,
                        irreversible_sample(spec, x, y, component),
                        spec.precision,
                        true,
                    );
                }
            }
        }
        return bytes;
    }

    let magic = if spec.components == 1 { "P5" } else { "P6" };
    let max_value = (1_u32 << spec.precision) - 1;
    let mut bytes = format!("{magic}\n{WIDTH} {HEIGHT}\n{max_value}\n").into_bytes();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            for component in 0..spec.components {
                let sample = u16::try_from(irreversible_sample(spec, x, y, component))
                    .expect("unsigned irreversible fixture sample");
                if spec.precision <= 8 {
                    bytes.push(u8::try_from(sample).expect("8-bit irreversible sample"));
                } else {
                    bytes.extend_from_slice(&sample.to_be_bytes());
                }
            }
        }
    }
    bytes
}

fn interleaved_native_bytes(spec: FixtureSpec) -> Vec<u8> {
    let mut bytes = Vec::new();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            for component in 0..spec.components {
                append_sample(
                    &mut bytes,
                    sample(spec, x, y, component),
                    spec.precision,
                    spec.signed,
                );
            }
        }
    }
    bytes
}

fn write_sources(output: &Path) {
    fs::create_dir_all(output).expect("create fixture output directory");

    for components in [1, 3] {
        for signed in [false, true] {
            if components == 3 && signed {
                continue;
            }
            for precision in [8, 12, 16] {
                let shape = if components == 1 { "gray" } else { "rgb" };
                let sign = if signed { "s" } else { "u" };
                let name = format!("{shape}_{sign}{precision}_53");
                let spec = FixtureSpec {
                    components,
                    precision,
                    signed,
                };
                let extension = match (components, signed) {
                    (1, false) => "pgm",
                    (3, false) => "ppm",
                    (1, true) => "raw",
                    (3, true) => "yuv",
                    _ => unreachable!("fixture catalog uses one or three components"),
                };
                let source = if signed {
                    planar_source_bytes(spec)
                } else {
                    pnm_source_bytes(spec)
                };
                fs::write(output.join(format!("{name}.source.{extension}")), source)
                    .expect("write reversible fixture source");
            }
        }
    }

    let irreversible = FixtureSpec {
        components: 3,
        precision: 8,
        signed: false,
    };
    fs::write(
        output.join("rgb_u8_97.source.ppm"),
        pnm_source_bytes(irreversible),
    )
    .expect("write irreversible fixture source");

    for (components, precision, signed) in [
        (1, 12, false),
        (1, 16, false),
        (1, 12, true),
        (1, 16, true),
        (3, 12, false),
    ] {
        let shape = if components == 1 { "gray" } else { "rgb" };
        let sign = if signed { "s" } else { "u" };
        let extension = if signed {
            "raw"
        } else if components == 1 {
            "pgm"
        } else {
            "ppm"
        };
        let spec = FixtureSpec {
            components,
            precision,
            signed,
        };
        fs::write(
            output.join(format!("{shape}_{sign}{precision}_97.source.{extension}")),
            irreversible_source_bytes(spec),
        )
        .expect("write >8-bit irreversible fixture source");
    }
}

fn pfm_payload(bytes: &[u8]) -> (&[u8], u32, u32) {
    let mut lines = bytes.splitn(4, |byte| *byte == b'\n');
    let magic = lines.next().expect("PFM magic");
    assert!(magic == b"Pf" || magic == b"PF", "PFM magic");
    let dimensions = std::str::from_utf8(lines.next().expect("PFM dimensions"))
        .expect("PFM dimensions are ASCII");
    let mut dimensions = dimensions.split_ascii_whitespace();
    let width = dimensions
        .next()
        .expect("PFM width")
        .parse::<u32>()
        .expect("PFM width is an integer");
    let height = dimensions
        .next()
        .expect("PFM height")
        .parse::<u32>()
        .expect("PFM height is an integer");
    let scale = std::str::from_utf8(lines.next().expect("PFM scale"))
        .expect("PFM scale is ASCII")
        .parse::<f32>()
        .expect("PFM scale is numeric");
    assert!(
        scale.is_sign_negative(),
        "fixture PFM must be little-endian"
    );
    (lines.next().expect("PFM payload"), width, height)
}

fn write_oracle(input: &Path, output: &Path, precision: u8, signed: bool) {
    let bytes = fs::read(input).expect("read OpenJPH PFM oracle");
    let (payload, width, height) = pfm_payload(&bytes);
    let pixels = usize::try_from(width * height).expect("fixture pixels fit usize");
    assert!(payload.len() == pixels * 4 || payload.len() == pixels * 3 * 4);
    let components = payload.len() / (pixels * 4);
    let shift = u32::from(32 - precision);
    let mut oracle = Vec::with_capacity(pixels * components * if precision <= 8 { 1 } else { 2 });
    for y in (0..height).rev() {
        for x in 0..width {
            for component in 0..components {
                let index =
                    ((y as usize * width as usize + x as usize) * components + component) * 4;
                let bytes = payload[index..index + 4]
                    .try_into()
                    .expect("PFM sample is four bytes");
                let value = if signed {
                    i32::from_le_bytes(bytes) >> shift
                } else {
                    i32::try_from(u32::from_le_bytes(bytes) >> shift)
                        .expect("unsigned fixture oracle fits i32")
                };
                append_sample(&mut oracle, value, precision, signed);
            }
        }
    }
    fs::write(output, oracle).expect("write raw OpenJPH oracle");
}

fn append_box(output: &mut Vec<u8>, box_type: &[u8; 4], payload: &[u8]) {
    let length = u32::try_from(payload.len() + 8).expect("fixture box length fits u32");
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(box_type);
    output.extend_from_slice(payload);
}

fn write_jph(input: &Path, output: &Path, components: u16, precision: u8, signed: bool) {
    let codestream = fs::read(input).expect("read OpenJPH codestream");
    let mut jph = Vec::with_capacity(codestream.len() + 100);
    append_box(&mut jph, b"jP  ", &[0x0d, 0x0a, 0x87, 0x0a]);
    let mut file_type = Vec::with_capacity(12);
    file_type.extend_from_slice(b"jph ");
    file_type.extend_from_slice(&0_u32.to_be_bytes());
    file_type.extend_from_slice(b"jph ");
    append_box(&mut jph, b"ftyp", &file_type);

    let mut image_header = Vec::with_capacity(14);
    image_header.extend_from_slice(&HEIGHT.to_be_bytes());
    image_header.extend_from_slice(&WIDTH.to_be_bytes());
    image_header.extend_from_slice(&components.to_be_bytes());
    image_header.push((precision - 1) | if signed { 0x80 } else { 0 });
    image_header.extend_from_slice(&[7, 0, 0]);
    let mut jp2_header = Vec::with_capacity(45);
    append_box(&mut jp2_header, b"ihdr", &image_header);
    let color_space = if components == 1 { 17_u32 } else { 16_u32 };
    let mut color = vec![1, 0, 0];
    color.extend_from_slice(&color_space.to_be_bytes());
    append_box(&mut jp2_header, b"colr", &color);
    append_box(&mut jph, b"jp2h", &jp2_header);
    append_box(&mut jph, b"jp2c", &codestream);
    fs::write(output, jph).expect("write independently boxed JPH fixture");
}

fn parse_bool(value: &str) -> bool {
    match value {
        "true" => true,
        "false" => false,
        _ => panic!("signedness must be true or false"),
    }
}

fn next_u32(arguments: &mut impl Iterator<Item = std::ffi::OsString>, what: &str) -> u32 {
    arguments
        .next()
        .unwrap_or_else(|| panic!("missing {what}"))
        .to_str()
        .unwrap_or_else(|| panic!("{what} is UTF-8"))
        .parse::<u32>()
        .unwrap_or_else(|_| panic!("{what} is an integer"))
}

fn main() {
    let mut arguments = env::args_os().skip(1);
    let mode = arguments
        .next()
        .expect("usage: generate sources DIR | oracle IN.pfm OUT.raw PRECISION SIGNED | verify ORACLE COMPONENTS PRECISION SIGNED | jph IN.j2c OUT.jph COMPONENTS PRECISION SIGNED");
    match mode.to_str().expect("mode is UTF-8") {
        "sources" => {
            let output = PathBuf::from(arguments.next().expect("sources output directory"));
            write_sources(&output);
        }
        "oracle" => {
            let input = PathBuf::from(arguments.next().expect("PFM input"));
            let output = PathBuf::from(arguments.next().expect("raw oracle output"));
            let precision = arguments
                .next()
                .expect("oracle precision")
                .to_str()
                .expect("precision is UTF-8")
                .parse::<u8>()
                .expect("precision is an integer");
            let signed = parse_bool(
                arguments
                    .next()
                    .expect("oracle signedness")
                    .to_str()
                    .expect("signedness is UTF-8"),
            );
            write_oracle(&input, &output, precision, signed);
        }
        "verify" => {
            let input = PathBuf::from(arguments.next().expect("raw oracle input"));
            let components = next_u32(&mut arguments, "component count");
            let precision =
                u8::try_from(next_u32(&mut arguments, "precision")).expect("precision fits u8");
            let signed = parse_bool(
                arguments
                    .next()
                    .expect("oracle signedness")
                    .to_str()
                    .expect("signedness is UTF-8"),
            );
            let actual = fs::read(input).expect("read raw OpenJPH oracle");
            let expected = interleaved_native_bytes(FixtureSpec {
                components,
                precision,
                signed,
            });
            assert_eq!(actual, expected, "reversible OpenJPH oracle");
        }
        "jph" => {
            let input = PathBuf::from(arguments.next().expect("codestream input"));
            let output = PathBuf::from(arguments.next().expect("JPH output"));
            let components = arguments
                .next()
                .expect("JPH component count")
                .to_str()
                .expect("component count is UTF-8")
                .parse::<u16>()
                .expect("component count is an integer");
            let precision = arguments
                .next()
                .expect("JPH precision")
                .to_str()
                .expect("precision is UTF-8")
                .parse::<u8>()
                .expect("precision is an integer");
            let signed = parse_bool(
                arguments
                    .next()
                    .expect("JPH signedness")
                    .to_str()
                    .expect("signedness is UTF-8"),
            );
            write_jph(&input, &output, components, precision, signed);
        }
        mode => panic!("unknown generator mode {mode}"),
    }
    assert!(arguments.next().is_none(), "unexpected generator arguments");
}
