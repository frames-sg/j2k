//! Integration tests for the OpenJPH-RS encode/decode pipeline.
//!
//! Tests end-to-end encoding and decoding of synthetic images.

mod common;

use openjph_core::codestream::Codestream;
use openjph_core::file::{MemInfile, MemOutfile};
use openjph_core::types::{Point, Size};

use common::mse_pae::{find_mse_pae, ImgInfo};

/// Helper: encode a single-component 8-bit unsigned image to a J2K byte buffer.
fn encode_8bit_gray(width: u32, height: u32, pixels: &[i32], num_decomps: u32) -> Vec<u8> {
    let mut cs = Codestream::new();

    // Configure SIZ
    {
        let siz = cs.access_siz_mut();
        siz.set_image_extent(Point::new(width, height));
        siz.set_tile_size(Size::new(width, height));
        siz.set_num_components(1);
        siz.set_comp_info(0, Point::new(1, 1), 8, false);
    }

    // Configure COD
    {
        let cod = cs.access_cod_mut();
        cod.set_reversible(true);
        cod.set_color_transform(false);
        cod.set_num_decomposition(num_decomps);
        cod.set_block_dims(64, 64);
    }

    // Write headers
    let mut outfile = MemOutfile::new();
    cs.write_headers(&mut outfile, &[]).unwrap();

    // Push image lines
    for y in 0..height {
        let start = (y * width) as usize;
        let end = start + width as usize;
        cs.exchange(&pixels[start..end], 0).unwrap();
    }

    // Flush
    cs.flush(&mut outfile).unwrap();

    outfile.get_data().to_vec()
}

/// Helper: decode a J2K byte buffer to a single-component image.
fn decode_8bit_gray(j2k_data: &[u8], width: u32, height: u32) -> Vec<i32> {
    let mut cs = Codestream::new();
    let mut infile = MemInfile::new(j2k_data);
    cs.read_headers(&mut infile).unwrap();
    cs.create(&mut infile).unwrap();

    let mut pixels = Vec::with_capacity((width * height) as usize);
    for _ in 0..height {
        if let Some(line) = cs.pull(0) {
            pixels.extend_from_slice(&line);
        }
    }

    pixels
}

#[test]
fn integration_roundtrip_constant_image() {
    let width = 16u32;
    let height = 16u32;
    let val = 128;
    let pixels: Vec<i32> = vec![val; (width * height) as usize];

    let j2k = encode_8bit_gray(width, height, &pixels, 2);
    assert!(!j2k.is_empty(), "encoded data should not be empty");

    let decoded = decode_8bit_gray(&j2k, width, height);
    assert_eq!(decoded.len(), pixels.len());

    // Reversible coding should be lossless
    let img_orig = ImgInfo::from_samples(
        width as usize,
        height as usize,
        8,
        false,
        vec![pixels.clone()],
    );
    let img_dec = ImgInfo::from_samples(
        width as usize,
        height as usize,
        8,
        false,
        vec![decoded.clone()],
    );
    let results = find_mse_pae(&img_orig, &img_dec);
    assert_eq!(
        results[0].mse, 0.0,
        "reversible coding must be lossless for constant image"
    );
    assert_eq!(results[0].pae, 0);
}

#[test]
fn integration_roundtrip_gradient_image() {
    // Test with gradient image that produces non-trivial DWT coefficients.
    // Uses small values to stay within block coder comfort zone.
    let width = 8u32;
    let height = 8u32;
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push(128i32 + ((x as i32 + y as i32) % 8) - 4);
        }
    }

    let j2k = encode_8bit_gray(width, height, &pixels, 2);
    assert!(!j2k.is_empty());

    let decoded = decode_8bit_gray(&j2k, width, height);
    assert_eq!(decoded.len(), pixels.len());

    let img_orig = ImgInfo::from_samples(
        width as usize,
        height as usize,
        8,
        false,
        vec![pixels.clone()],
    );
    let img_dec = ImgInfo::from_samples(
        width as usize,
        height as usize,
        8,
        false,
        vec![decoded.clone()],
    );
    let results = find_mse_pae(&img_orig, &img_dec);
    assert_eq!(
        results[0].mse, 0.0,
        "reversible coding must be lossless. PAE={}",
        results[0].pae
    );
}

#[test]
fn integration_roundtrip_no_dwt() {
    // Test with 0 decomposition levels (no DWT) — constant value
    let width = 8u32;
    let height = 8u32;
    let pixels = vec![100i32; (width * height) as usize];

    let j2k = encode_8bit_gray(width, height, &pixels, 0);
    assert!(!j2k.is_empty());

    let decoded = decode_8bit_gray(&j2k, width, height);
    assert_eq!(decoded.len(), pixels.len());

    let img_orig = ImgInfo::from_samples(
        width as usize,
        height as usize,
        8,
        false,
        vec![pixels.clone()],
    );
    let img_dec = ImgInfo::from_samples(
        width as usize,
        height as usize,
        8,
        false,
        vec![decoded.clone()],
    );
    let results = find_mse_pae(&img_orig, &img_dec);
    assert_eq!(results[0].mse, 0.0, "no-DWT roundtrip must be lossless");
}

#[test]
fn integration_mse_pae_api() {
    // Test that the MSE/PAE API works correctly with known differences
    let width = 4usize;
    let height = 4usize;
    let original = vec![
        100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250,
    ];
    // Introduce known errors
    let decoded = vec![
        101, 110, 118, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 245,
    ];

    let img1 = ImgInfo::from_samples(width, height, 8, false, vec![original]);
    let img2 = ImgInfo::from_samples(width, height, 8, false, vec![decoded]);

    let results = find_mse_pae(&img1, &img2);
    assert_eq!(results.len(), 1);
    // Errors: 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5
    // SSE = 1+0+4+0+...+25 = 30, MSE = 30/16 = 1.875
    assert!((results[0].mse - 1.875).abs() < 0.001);
    assert_eq!(results[0].pae, 5);
}

#[test]
fn integration_compare_files_api() {
    use common::compare_files::{compare_j2k_bytes, CompareResult};

    // Identical data should match
    let data = vec![0xFF, 0x4F, 0xFF, 0x51, 0x00, 0x04, 0x01, 0x02];
    assert_eq!(compare_j2k_bytes(&data, &data), CompareResult::Match);

    // Different data should mismatch
    let mut data2 = data.clone();
    data2[7] = 0x03;
    assert!(matches!(
        compare_j2k_bytes(&data, &data2),
        CompareResult::Mismatch { .. }
    ));
}

#[test]
fn integration_codestream_header_roundtrip() {
    // Test that headers written can be read back
    let mut cs = Codestream::new();
    {
        let siz = cs.access_siz_mut();
        siz.set_image_extent(Point::new(32, 32));
        siz.set_tile_size(Size::new(32, 32));
        siz.set_num_components(1);
        siz.set_comp_info(0, Point::new(1, 1), 8, false);
    }
    {
        let cod = cs.access_cod_mut();
        cod.set_reversible(true);
        cod.set_color_transform(false);
        cod.set_num_decomposition(3);
    }

    let mut outfile = MemOutfile::new();
    cs.write_headers(&mut outfile, &[]).unwrap();

    // Read it back
    let data = outfile.get_data().to_vec();
    let mut cs2 = Codestream::new();
    let mut infile = MemInfile::new(&data);
    // read_headers will stop at SOT (or EOF if no tiles)
    // Since we didn't flush tiles, we may get an error at the SOT check.
    // That's okay — the headers should still be parsed correctly.
    let _ = cs2.read_headers(&mut infile);

    // Verify the read parameters match
    assert_eq!(cs2.access_siz().get_num_components(), 1);
    assert_eq!(cs2.access_cod().get_num_decompositions(), 3);
    assert!(cs2.access_cod().is_reversible());
}

#[test]
fn integration_roundtrip_no_dwt_varied_values() {
    // Test no-DWT with a variety of pixel values (not just constant)
    let width = 8u32;
    let height = 8u32;
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for i in 0..(width * height) {
        pixels.push((i * 4 % 256) as i32);
    }

    let j2k = encode_8bit_gray(width, height, &pixels, 0);
    let decoded = decode_8bit_gray(&j2k, width, height);

    let img_orig = ImgInfo::from_samples(
        width as usize,
        height as usize,
        8,
        false,
        vec![pixels.clone()],
    );
    let img_dec = ImgInfo::from_samples(width as usize, height as usize, 8, false, vec![decoded]);
    let results = find_mse_pae(&img_orig, &img_dec);
    assert_eq!(results[0].pae, 0, "no-DWT varied values must be lossless");
}

#[test]
fn integration_roundtrip_dwt1_small() {
    // Single DWT level on a small 4x4 image
    let width = 4u32;
    let height = 4u32;
    let pixels = vec![
        100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250,
    ];

    let j2k = encode_8bit_gray(width, height, &pixels, 1);
    let decoded = decode_8bit_gray(&j2k, width, height);

    let img_orig = ImgInfo::from_samples(
        width as usize,
        height as usize,
        8,
        false,
        vec![pixels.clone()],
    );
    let img_dec = ImgInfo::from_samples(width as usize, height as usize, 8, false, vec![decoded]);
    let results = find_mse_pae(&img_orig, &img_dec);
    assert_eq!(results[0].pae, 0, "1-level DWT on 4x4 must be lossless");
}
