// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]

use std::{
    fs,
    io::{self, ErrorKind},
    path::Path,
};

mod corpus;
mod cuda;
mod fixtures;
mod jpeg_fixtures;
mod metal_shader;
mod pixels;

pub use corpus::{collect_jpeg_paths, is_jpeg_path, paths_from_env};
pub use cuda::{
    cuda_bench_required, cuda_htj2k_strict_required, cuda_jpeg_hardware_decode_required,
    cuda_runtime_required,
};
pub use fixtures::{
    baseline_grayscale_jpeg, jpeg_baseline_420_16x16, jpeg_baseline_420_16x16_rgb,
    jpeg_baseline_420_restart_32x16, jpeg_baseline_420_restart_32x16_rgb, jpeg_baseline_422_16x8,
    jpeg_baseline_422_16x8_rgb, jpeg_baseline_444_8x8, jpeg_baseline_444_8x8_rgb,
    jpeg_grayscale_8x8, jpeg_grayscale_8x8_gray, minimal_baseline_jpeg,
    minimal_baseline_jpeg_with_restart_interval, minimal_gray8_jpeg,
    minimal_grayscale_jpeg_with_dimensions, minimal_j2k_codestream, minimal_jp2,
    restart_coded_grayscale_jpeg, wrap_jp2_codestream, wrap_jp2_rgba_codestream,
    JPEG_BASELINE_420_16X16, JPEG_BASELINE_420_16X16_RGB, JPEG_BASELINE_420_RESTART_32X16,
    JPEG_BASELINE_420_RESTART_32X16_RGB, JPEG_BASELINE_422_16X8, JPEG_BASELINE_422_16X8_RGB,
    JPEG_BASELINE_444_8X8, JPEG_BASELINE_444_8X8_RGB, JPEG_GRAYSCALE_8X8, JPEG_GRAYSCALE_8X8_GRAY,
};
#[cfg(feature = "j2k-native-fixtures")]
pub use fixtures::{
    classic_j2k_gray8_fixture, htj2k_gray8_97_fixture, htj2k_gray8_fixture,
    htj2k_gray8_large_fixture, htj2k_rgb8_97_fixture, htj2k_rgb8_fixture,
    htj2k_rgb8_fixture_with_pixels, htj2k_rgb8_pattern_fixture, openhtj2k_refinement_odd_fixture,
};
pub use jpeg_fixtures::*;
pub use metal_shader::{host_compiles_metal_pipeline, metal_kernel_names, unwired_metal_kernels};
pub use pixels::{
    centered_rect, crop_interleaved_bytes, crop_interleaved_u16, crop_interleaved_u8,
    project_scaled_interleaved_u16, project_scaled_interleaved_u8, rgb16le_to_rgba16le,
    rgb16ne_to_opaque_rgba16ne, rgb8_to_rgba8, scaled_rect_covering, u16_samples_to_le_bytes,
    PixelRect,
};

/// Generates deterministic RGB8 pixels for tests and benches.
pub fn patterned_rgb8(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 17 + y * 3) & 0xFF) as u8);
            pixels.push(((x * 5 + y * 11 + 40) & 0xFF) as u8);
            pixels.push(((x * 13 + y * 7 + 90) & 0xFF) as u8);
        }
    }
    pixels
}

/// Generates deterministic grayscale pixels for tests and benches.
pub fn patterned_gray8(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 19 + y * 23) & 0xFF) as u8);
        }
    }
    pixels
}

/// Generates a simple deterministic gradient for reference codec parity cases.
pub fn gradient_u8(width: u32, height: u32, channels: usize) -> Vec<u8> {
    gradient_variant_u8(width, height, channels, 0)
}

/// Generates a deterministic gradient variant keyed by `seed`.
pub fn gradient_variant_u8(width: u32, height: u32, channels: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(width as usize * height as usize * channels);
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                out.push(((x + y + seed * 13 + (c as u32 * 17)) & 0xFF) as u8);
            }
        }
    }
    out
}

/// Generates deterministic RGB8 pixels used by GPU upload/decode benches.
pub fn gpu_bench_rgb8(width: u32, height: u32) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            rgb.push(((x * 13 + y * 3) & 0xff) as u8);
            rgb.push(((x * 5 + y * 11 + (x ^ y)) & 0xff) as u8);
            rgb.push(((x * 7 + y * 17 + (x.wrapping_mul(y) >> 5)) & 0xff) as u8);
        }
    }
    rgb
}

/// Generates contiguous RGB8 tiles for JPEG baseline encode benches.
pub fn patterned_rgb8_tiles(width: u32, height: u32, tile_count: usize) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(width as usize * height as usize * 3 * tile_count);
    for tile in 0..tile_count as u32 {
        for y in 0..height {
            for x in 0..width {
                rgb.push(((x * 13 + y * 3 + tile * 29) & 0xff) as u8);
                rgb.push(((x * 5 + y * 11 + (x ^ y) + tile * 17) & 0xff) as u8);
                rgb.push(((x * 7 + y * 17 + (x.wrapping_mul(y) >> 5) + tile * 23) & 0xff) as u8);
            }
        }
    }
    rgb
}

/// Compatibility alias for the old JP2 fixture helper name.
#[must_use]
pub fn wrap_codestream_jp2(
    codestream: &[u8],
    width: u32,
    height: u32,
    components: u16,
    bit_depth: u8,
    colorspace_enum: u32,
) -> Vec<u8> {
    fixtures::wrap_jp2_codestream(
        codestream,
        width,
        height,
        components,
        bit_depth,
        colorspace_enum,
    )
}

/// Builds a binary PGM (`P5`) or PPM (`P6`) fixture from raw 8-bit pixels.
///
/// # Errors
///
/// Returns an error when `channels` is not `1` or `3`, when the dimensions
/// overflow, or when `pixels.len()` does not match the requested image shape.
pub fn pnm_bytes(pixels: &[u8], width: u32, height: u32, channels: usize) -> io::Result<Vec<u8>> {
    let magic = match channels {
        1 => "P5",
        3 => "P6",
        _ => {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "PNM fixtures support only 1 or 3 channels",
            ));
        }
    };
    let expected_len = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidInput, "PNM dimensions overflow"))?;
    if pixels.len() != expected_len {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            format!(
                "PNM pixel length {} does not match expected {expected_len}",
                pixels.len()
            ),
        ));
    }

    let mut bytes = format!("{magic}\n{width} {height}\n255\n").into_bytes();
    bytes.extend_from_slice(pixels);
    Ok(bytes)
}

/// Writes a binary PGM (`P5`) or PPM (`P6`) fixture.
///
/// # Errors
///
/// Returns an error when [`pnm_bytes`] rejects the image shape or when the file
/// cannot be written.
pub fn write_pnm(
    path: impl AsRef<Path>,
    pixels: &[u8],
    width: u32,
    height: u32,
    channels: usize,
) -> io::Result<()> {
    fs::write(path, pnm_bytes(pixels, width, height, channels)?)
}

/// Reads pixel payload bytes from a binary PGM (`P5`) or PPM (`P6`) fixture.
///
/// # Errors
///
/// Returns an error when the file cannot be read or the PNM header is missing
/// the expected `P5`/`P6` magic, dimensions, or max-value fields.
pub fn read_pnm_pixels(path: impl AsRef<Path>) -> io::Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    let mut cursor = 0;

    let magic = read_pnm_token(&bytes, &mut cursor)
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "PNM missing magic"))?;
    if magic != b"P5" && magic != b"P6" {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "PNM magic must be P5 or P6",
        ));
    }

    read_pnm_token(&bytes, &mut cursor)
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "PNM missing width"))?;
    read_pnm_token(&bytes, &mut cursor)
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "PNM missing height"))?;
    read_pnm_token(&bytes, &mut cursor)
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "PNM missing max value"))?;

    while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
        cursor += 1;
    }

    Ok(bytes[cursor..].to_vec())
}

fn read_pnm_token<'a>(bytes: &'a [u8], cursor: &mut usize) -> Option<&'a [u8]> {
    skip_pnm_separators(bytes, cursor);
    if *cursor >= bytes.len() {
        return None;
    }

    let start = *cursor;
    while *cursor < bytes.len() {
        let byte = bytes[*cursor];
        if byte.is_ascii_whitespace() || byte == b'#' {
            break;
        }
        *cursor += 1;
    }

    (start != *cursor).then_some(&bytes[start..*cursor])
}

fn skip_pnm_separators(bytes: &[u8], cursor: &mut usize) {
    while *cursor < bytes.len() {
        let byte = bytes[*cursor];
        if byte.is_ascii_whitespace() {
            *cursor += 1;
            continue;
        }
        if byte == b'#' {
            *cursor += 1;
            while *cursor < bytes.len() && bytes[*cursor] != b'\n' {
                *cursor += 1;
            }
            continue;
        }
        break;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        minimal_j2k_codestream, minimal_jp2, pnm_bytes, read_pnm_pixels, wrap_codestream_jp2,
        write_pnm,
    };

    #[test]
    fn minimal_j2k_codestream_has_j2k_magic_and_siz_marker() {
        let codestream = minimal_j2k_codestream();

        assert!(codestream.starts_with(&[0xFF, 0x4F]));
        assert!(codestream.windows(2).any(|marker| marker == [0xFF, 0x51]));
    }

    #[test]
    fn minimal_jp2_wraps_the_minimal_codestream() {
        let jp2 = minimal_jp2();

        assert!(jp2.starts_with(&[0, 0, 0, 12, b'j', b'P', b' ', b' ']));
        assert!(jp2.windows(4).any(|box_type| box_type == b"jp2c"));
        assert!(jp2.windows(2).any(|marker| marker == [0xFF, 0x4F]));
    }

    #[test]
    fn jp2_wrapper_writes_image_header_dimensions_and_colorspace() {
        let jp2 = wrap_codestream_jp2(&[0xFF, 0x4F], 320, 240, 3, 8, 16);

        assert!(jp2.windows(4).any(|box_type| box_type == b"jp2h"));
        assert!(jp2.windows(4).any(|value| value == 240u32.to_be_bytes()));
        assert!(jp2.windows(4).any(|value| value == 320u32.to_be_bytes()));
        assert!(jp2.windows(4).any(|value| value == 16u32.to_be_bytes()));
    }

    #[test]
    fn pnm_bytes_writes_p5_and_p6_headers() {
        assert_eq!(
            pnm_bytes(&[1, 2], 2, 1, 1).unwrap(),
            b"P5\n2 1\n255\n\x01\x02"
        );
        assert_eq!(
            pnm_bytes(&[1, 2, 3, 4, 5, 6], 1, 2, 3).unwrap(),
            b"P6\n1 2\n255\n\x01\x02\x03\x04\x05\x06"
        );
    }

    #[test]
    fn pnm_bytes_rejects_unsupported_shape() {
        assert!(pnm_bytes(&[1, 2], 1, 1, 2).is_err());
        assert!(pnm_bytes(&[1, 2], 2, 2, 1).is_err());
    }

    #[test]
    fn write_and_read_pnm_round_trips_pixels_with_comments() {
        let path =
            std::env::temp_dir().join(format!("j2k-test-support-pnm-{}.ppm", std::process::id()));
        let bytes = b"P6\n# generated by test\n2 1\n255\n\x01\x02\x03\x04\x05\x06";
        std::fs::write(&path, bytes).expect("write commented pnm fixture");

        assert_eq!(
            read_pnm_pixels(&path).expect("read pnm pixels"),
            vec![1, 2, 3, 4, 5, 6]
        );

        write_pnm(&path, &[7, 8], 2, 1, 1).expect("write pnm");
        assert_eq!(read_pnm_pixels(&path).expect("read pnm pixels"), vec![7, 8]);
        let _ = std::fs::remove_file(path);
    }
}
