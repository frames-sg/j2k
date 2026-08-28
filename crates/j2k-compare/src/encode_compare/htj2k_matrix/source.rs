// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path};

use super::{
    types::{SampleFormat, SourceImage},
    HEIGHT, WIDTH,
};

pub(super) fn generated_source(format: SampleFormat) -> SourceImage {
    let mut pixels_le = Vec::new();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            match format {
                SampleFormat::Gray8 => {
                    pixels_le.push((x * 37 + y * 73 + x * y).to_le_bytes()[0]);
                }
                SampleFormat::Rgb8 => {
                    pixels_le.push((x * 5 + y * 3).to_le_bytes()[0]);
                    pixels_le.push((x * 11 + y * 17 + 41).to_le_bytes()[0]);
                    pixels_le.push((x * 23 + y * 7 + x * y).to_le_bytes()[0]);
                }
                SampleFormat::Gray16 => {
                    let value = (x * 400 + y * 150).to_le_bytes();
                    pixels_le.extend_from_slice(&value[..2]);
                }
            }
        }
    }
    let magic = if format == SampleFormat::Rgb8 {
        "P6"
    } else {
        "P5"
    };
    let max_value = if format == SampleFormat::Gray16 {
        65_535
    } else {
        255
    };
    let mut pnm_bytes = format!("{magic}\n{WIDTH} {HEIGHT}\n{max_value}\n").into_bytes();
    if format == SampleFormat::Gray16 {
        for sample in pixels_le.chunks_exact(2) {
            pnm_bytes.extend_from_slice(&u16::from_le_bytes([sample[0], sample[1]]).to_be_bytes());
        }
    } else {
        pnm_bytes.extend_from_slice(&pixels_le);
    }
    SourceImage {
        format,
        pixels_le,
        pnm_bytes,
    }
}

pub(super) fn read_pnm_as_le(path: &Path, expected: SampleFormat) -> Result<Vec<u8>, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let mut cursor = 0;
    let magic = pnm_token(&bytes, &mut cursor)?;
    let expected_magic = if expected == SampleFormat::Rgb8 {
        b"P6".as_slice()
    } else {
        b"P5".as_slice()
    };
    if magic != expected_magic {
        return Err(format!("{} has unexpected PNM magic", path.display()));
    }
    let width = parse_pnm_number(pnm_token(&bytes, &mut cursor)?, "width")?;
    let height = parse_pnm_number(pnm_token(&bytes, &mut cursor)?, "height")?;
    let max_value = parse_pnm_number(pnm_token(&bytes, &mut cursor)?, "max value")?;
    let expected_max = if expected == SampleFormat::Gray16 {
        65_535
    } else {
        255
    };
    if width != WIDTH || height != HEIGHT || max_value != expected_max {
        return Err(format!(
            "{} has PNM profile {width}x{height} max={max_value}",
            path.display()
        ));
    }
    consume_pnm_separator(&bytes, &mut cursor)?;
    let payload = &bytes[cursor..];
    let width = usize::try_from(WIDTH).map_err(|_| "PNM width exceeds usize".to_string())?;
    let height = usize::try_from(HEIGHT).map_err(|_| "PNM height exceeds usize".to_string())?;
    let samples = width * height * usize::from(expected.components());
    let expected_len = samples
        * usize::from(if expected == SampleFormat::Gray16 {
            2_u8
        } else {
            1
        });
    if payload.len() != expected_len {
        return Err(format!(
            "{} PNM payload length {} != {expected_len}",
            path.display(),
            payload.len()
        ));
    }
    if expected == SampleFormat::Gray16 {
        Ok(payload
            .chunks_exact(2)
            .flat_map(|sample| u16::from_be_bytes([sample[0], sample[1]]).to_le_bytes())
            .collect())
    } else {
        Ok(payload.to_vec())
    }
}

fn pnm_token<'a>(bytes: &'a [u8], cursor: &mut usize) -> Result<&'a [u8], String> {
    loop {
        while bytes.get(*cursor).is_some_and(u8::is_ascii_whitespace) {
            *cursor += 1;
        }
        if bytes.get(*cursor) != Some(&b'#') {
            break;
        }
        while bytes.get(*cursor).is_some_and(|byte| *byte != b'\n') {
            *cursor += 1;
        }
    }
    let start = *cursor;
    while bytes
        .get(*cursor)
        .is_some_and(|byte| !byte.is_ascii_whitespace() && *byte != b'#')
    {
        *cursor += 1;
    }
    if start == *cursor {
        return Err("truncated PNM header".to_string());
    }
    Ok(&bytes[start..*cursor])
}

fn parse_pnm_number(token: &[u8], label: &str) -> Result<u32, String> {
    std::str::from_utf8(token)
        .map_err(|error| format!("PNM {label} is not UTF-8: {error}"))?
        .parse()
        .map_err(|error| format!("invalid PNM {label}: {error}"))
}

fn consume_pnm_separator(bytes: &[u8], cursor: &mut usize) -> Result<(), String> {
    let separator = *bytes
        .get(*cursor)
        .ok_or_else(|| "PNM header has no pixel separator".to_string())?;
    if !separator.is_ascii_whitespace() {
        return Err("PNM header is not followed by whitespace".to_string());
    }
    *cursor += 1;
    if separator == b'\r' && bytes.get(*cursor) == Some(&b'\n') {
        *cursor += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{generated_source, read_pnm_as_le, SampleFormat};

    #[test]
    fn generated_pnm_round_trips_matrix_sample_endianness() {
        for format in [
            SampleFormat::Gray8,
            SampleFormat::Rgb8,
            SampleFormat::Gray16,
        ] {
            let source = generated_source(format);
            let path = std::env::temp_dir().join(format!(
                "j2k-openjph-matrix-pnm-{}-{}",
                std::process::id(),
                format.label()
            ));
            std::fs::write(&path, &source.pnm_bytes).expect("write generated PNM");
            let parsed = read_pnm_as_le(&path, format).expect("parse generated PNM");
            assert_eq!(parsed, source.pixels_le);
        }
    }
}
