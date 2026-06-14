// SPDX-License-Identifier: Apache-2.0

//! Decode a JPEG through `signinum-jpeg` and write a viewable 24-bit BMP.
//!
//! Run with:
//! `cargo run -p signinum-jpeg --example dump_bmp`
//!
//! Or pass an input and output path:
//! `cargo run -p signinum-jpeg --example dump_bmp -- input.jpg output.bmp`

use std::{
    env, fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use signinum_jpeg::{Decoder, PixelFormat};
use signinum_test_support::JPEG_BASELINE_420_16X16;

const DEFAULT_OUTPUT: &str = "target/signinum-jpeg-baseline-420.bmp";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let (input_label, bytes, output_path) =
        match args.as_slice() {
            [] => (
                "baseline_420_16x16 fixture".to_string(),
                JPEG_BASELINE_420_16X16.to_vec(),
                PathBuf::from(DEFAULT_OUTPUT),
            ),
            [input, output] => (input.clone(), fs::read(input)?, PathBuf::from(output)),
            _ => return Err(
                "usage: cargo run -p signinum-jpeg --example dump_bmp -- [input.jpg output.bmp]"
                    .into(),
            ),
        };

    let decoder = Decoder::new(&bytes)?;
    let (rgb, outcome) = decoder.decode(PixelFormat::Rgb8)?;
    let width = outcome.decoded.w as usize;
    let height = outcome.decoded.h as usize;

    write_rgb8_bmp(&output_path, width, height, &rgb)?;
    println!(
        "decoded {input_label} as {}x{} Rgb8 and wrote {}",
        width,
        height,
        output_path.display()
    );
    Ok(())
}

fn write_rgb8_bmp(
    path: &Path,
    width: usize,
    height: usize,
    rgb: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let expected_len = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or("decoded image dimensions overflow")?;
    if rgb.len() != expected_len {
        return Err(format!("expected {expected_len} RGB bytes, got {}", rgb.len()).into());
    }

    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }

    let row_stride = width
        .checked_mul(3)
        .and_then(|row| row.checked_add(3))
        .map(|row| row & !3)
        .ok_or("BMP row stride overflow")?;
    let pixel_bytes = row_stride
        .checked_mul(height)
        .ok_or("BMP pixel buffer size overflow")?;
    let file_size = 54usize
        .checked_add(pixel_bytes)
        .ok_or("BMP file size overflow")?;

    let mut out = BufWriter::new(fs::File::create(path)?);
    out.write_all(b"BM")?;
    write_u32(&mut out, u32::try_from(file_size)?)?;
    write_u16(&mut out, 0)?;
    write_u16(&mut out, 0)?;
    write_u32(&mut out, 54)?;

    write_u32(&mut out, 40)?;
    write_i32(&mut out, i32::try_from(width)?)?;
    write_i32(&mut out, i32::try_from(height)?)?;
    write_u16(&mut out, 1)?;
    write_u16(&mut out, 24)?;
    write_u32(&mut out, 0)?;
    write_u32(&mut out, u32::try_from(pixel_bytes)?)?;
    write_i32(&mut out, 0)?;
    write_i32(&mut out, 0)?;
    write_u32(&mut out, 0)?;
    write_u32(&mut out, 0)?;

    let padding = vec![0_u8; row_stride - width * 3];
    for y in (0..height).rev() {
        let row = &rgb[y * width * 3..(y + 1) * width * 3];
        for pixel in row.chunks_exact(3) {
            out.write_all(&[pixel[2], pixel[1], pixel[0]])?;
        }
        out.write_all(&padding)?;
    }
    Ok(())
}

fn write_u16(out: &mut dyn Write, value: u16) -> Result<(), std::io::Error> {
    out.write_all(&value.to_le_bytes())
}

fn write_u32(out: &mut dyn Write, value: u32) -> Result<(), std::io::Error> {
    out.write_all(&value.to_le_bytes())
}

fn write_i32(out: &mut dyn Write, value: i32) -> Result<(), std::io::Error> {
    out.write_all(&value.to_le_bytes())
}
