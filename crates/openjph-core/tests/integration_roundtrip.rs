//! End-to-end roundtrip tests for the OpenJPH-RS codec.
//!
//! These tests generate synthetic PPM/PGM-like images in memory, encode to
//! J2K via the Codestream API, decode back, and verify MSE/PAE within
//! tolerance. Reversible (5/3) expects exact lossless; irreversible (9/7)
//! expects bounded error.

mod common;

use openjph_core::codestream::Codestream;
use openjph_core::file::{MemInfile, MemOutfile};
use openjph_core::types::{Point, Size};

use common::mse_pae::{find_mse_pae, ImgInfo};

// ============================================================================
// Helpers
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn encode_multicomp(
    width: u32,
    height: u32,
    num_comps: u32,
    bit_depth: u32,
    is_signed: bool,
    reversible: bool,
    color_transform: bool,
    num_decomps: u32,
    block_w: u32,
    block_h: u32,
    tile_w: u32,
    tile_h: u32,
    qstep: f32,
    components: &[Vec<i32>],
) -> Vec<u8> {
    let mut cs = Codestream::new();

    {
        let siz = cs.access_siz_mut();
        siz.set_image_extent(Point::new(width, height));
        let tw = if tile_w > 0 { tile_w } else { width };
        let th = if tile_h > 0 { tile_h } else { height };
        siz.set_tile_size(Size::new(tw, th));
        siz.set_num_components(num_comps);
        for c in 0..num_comps {
            siz.set_comp_info(c, Point::new(1, 1), bit_depth, is_signed);
        }
    }

    {
        let cod = cs.access_cod_mut();
        cod.set_reversible(reversible);
        cod.set_color_transform(color_transform && num_comps >= 3);
        cod.set_num_decomposition(num_decomps);
        cod.set_block_dims(block_w, block_h);
    }

    if !reversible {
        cs.access_qcd_mut().set_delta(qstep);
    }

    let mut outfile = MemOutfile::new();
    cs.write_headers(&mut outfile, &[]).unwrap();

    for y in 0..height {
        for c in 0..num_comps {
            let start = (y * width) as usize;
            let end = start + width as usize;
            cs.exchange(&components[c as usize][start..end], c).unwrap();
        }
    }

    cs.flush(&mut outfile).unwrap();
    outfile.get_data().to_vec()
}

fn decode_multicomp(j2k_data: &[u8], width: u32, height: u32, num_comps: u32) -> Vec<Vec<i32>> {
    let mut cs = Codestream::new();
    let mut infile = MemInfile::new(j2k_data);
    cs.read_headers(&mut infile).unwrap();
    cs.create(&mut infile).unwrap();

    let mut components: Vec<Vec<i32>> = (0..num_comps)
        .map(|_| Vec::with_capacity((width * height) as usize))
        .collect();

    for _ in 0..height {
        for c in 0..num_comps {
            if let Some(line) = cs.pull(c) {
                components[c as usize].extend_from_slice(&line);
            }
        }
    }
    components
}

fn verify_lossless(
    width: u32,
    height: u32,
    bit_depth: u32,
    is_signed: bool,
    original: &[Vec<i32>],
    decoded: &[Vec<i32>],
) {
    let img_orig = ImgInfo::from_samples(
        width as usize,
        height as usize,
        bit_depth,
        is_signed,
        original.to_vec(),
    );
    let img_dec = ImgInfo::from_samples(
        width as usize,
        height as usize,
        bit_depth,
        is_signed,
        decoded.to_vec(),
    );
    let results = find_mse_pae(&img_orig, &img_dec);
    for (c, res) in results.iter().enumerate() {
        assert_eq!(
            res.mse, 0.0,
            "component {}: lossless roundtrip failed (MSE={})",
            c, res.mse
        );
        assert_eq!(
            res.pae, 0,
            "component {}: lossless roundtrip failed (PAE={})",
            c, res.pae
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn verify_lossy(
    width: u32,
    height: u32,
    bit_depth: u32,
    is_signed: bool,
    original: &[Vec<i32>],
    decoded: &[Vec<i32>],
    max_mse_ratio: f64,
    max_pae: u32,
) {
    let img_orig = ImgInfo::from_samples(
        width as usize,
        height as usize,
        bit_depth,
        is_signed,
        original.to_vec(),
    );
    let img_dec = ImgInfo::from_samples(
        width as usize,
        height as usize,
        bit_depth,
        is_signed,
        decoded.to_vec(),
    );
    let results = find_mse_pae(&img_orig, &img_dec);
    let range = ((1u64 << bit_depth) - 1) as f64;
    let max_mse = range * range * max_mse_ratio;
    for (c, res) in results.iter().enumerate() {
        assert!(
            (res.mse as f64) < max_mse,
            "component {}: MSE {} exceeds max {} (ratio {})",
            c,
            res.mse,
            max_mse,
            max_mse_ratio
        );
        if max_pae > 0 {
            assert!(
                res.pae <= max_pae,
                "component {}: PAE {} exceeds max {}",
                c,
                res.pae,
                max_pae
            );
        }
    }
}

// ============================================================================
// Gradient image generators
// ============================================================================

fn gen_gradient_gray(w: u32, h: u32, bit_depth: u32) -> Vec<Vec<i32>> {
    let max_val = (1i32 << bit_depth) - 1;
    let n = (w * h) as usize;
    let mut pix = Vec::with_capacity(n);
    for _y in 0..h {
        for x in 0..w {
            pix.push(((x as i64 * max_val as i64) / w.max(1) as i64) as i32);
        }
    }
    vec![pix]
}

fn gen_gradient_rgb(w: u32, h: u32, bit_depth: u32) -> Vec<Vec<i32>> {
    let max_val = (1i32 << bit_depth) - 1;
    let n = (w * h) as usize;
    let mut comps = Vec::new();
    for c in 0..3u32 {
        let mut pix = Vec::with_capacity(n);
        for y in 0..h {
            for x in 0..w {
                let val = match c {
                    0 => ((x as i64 * max_val as i64) / w.max(1) as i64) as i32,
                    1 => ((y as i64 * max_val as i64) / h.max(1) as i64) as i32,
                    _ => (((x + y) as i64 * (max_val as i64 / 2)) / (w + h).max(1) as i64) as i32,
                };
                pix.push(val);
            }
        }
        comps.push(pix);
    }
    comps
}

fn gen_checkerboard_gray(w: u32, h: u32, bit_depth: u32, block_sz: u32) -> Vec<Vec<i32>> {
    let max_val = (1i32 << bit_depth) - 1;
    let n = (w * h) as usize;
    let mut pix = Vec::with_capacity(n);
    for y in 0..h {
        for x in 0..w {
            let bx = x / block_sz;
            let by = y / block_sz;
            pix.push(if (bx + by) % 2 == 0 {
                max_val / 4
            } else {
                max_val * 3 / 4
            });
        }
    }
    vec![pix]
}

fn gen_random_gray(w: u32, h: u32, bit_depth: u32, seed: u32) -> Vec<Vec<i32>> {
    let max_val = (1u32 << bit_depth) - 1;
    let n = (w * h) as usize;
    let mut pix = Vec::with_capacity(n);
    let mut rng = seed;
    for _ in 0..n {
        // Simple LCG PRNG
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        pix.push(((rng >> 16) % (max_val + 1)) as i32);
    }
    vec![pix]
}

fn gen_signed_gray(w: u32, h: u32, bit_depth: u32) -> Vec<Vec<i32>> {
    let half = 1i32 << (bit_depth - 1);
    let n = (w * h) as usize;
    let mut pix = Vec::with_capacity(n);
    for y in 0..h {
        for x in 0..w {
            let val = ((x as i64 + y as i64) % (2 * half as i64)) as i32 - half;
            pix.push(val);
        }
    }
    vec![pix]
}

// ============================================================================
// 8-bit grayscale roundtrip tests
// ============================================================================

#[test]
fn roundtrip_lossless_gray_gradient_8bit() {
    let w = 64;
    let h = 64;
    let comps = gen_gradient_gray(w, h, 8);
    let j2k = encode_multicomp(w, h, 1, 8, false, true, false, 5, 64, 64, 0, 0, 0.0, &comps);
    let decoded = decode_multicomp(&j2k, w, h, 1);
    verify_lossless(w, h, 8, false, &comps, &decoded);
}

#[test]
fn roundtrip_lossless_gray_checkerboard_8bit() {
    let w = 64;
    let h = 64;
    let comps = gen_checkerboard_gray(w, h, 8, 8);
    let j2k = encode_multicomp(w, h, 1, 8, false, true, false, 5, 64, 64, 0, 0, 0.0, &comps);
    let decoded = decode_multicomp(&j2k, w, h, 1);
    verify_lossless(w, h, 8, false, &comps, &decoded);
}

#[test]
fn roundtrip_lossless_gray_random_8bit() {
    let w = 64;
    let h = 64;
    let comps = gen_random_gray(w, h, 8, 42);
    let j2k = encode_multicomp(w, h, 1, 8, false, true, false, 5, 64, 64, 0, 0, 0.0, &comps);
    let decoded = decode_multicomp(&j2k, w, h, 1);
    verify_lossless(w, h, 8, false, &comps, &decoded);
}

#[test]
fn roundtrip_lossless_gray_constant_8bit() {
    let w = 32;
    let h = 32;
    let comps = vec![vec![128i32; (w * h) as usize]];
    let j2k = encode_multicomp(w, h, 1, 8, false, true, false, 3, 64, 64, 0, 0, 0.0, &comps);
    let decoded = decode_multicomp(&j2k, w, h, 1);
    verify_lossless(w, h, 8, false, &comps, &decoded);
}

// ============================================================================
// 8-bit RGB roundtrip tests
// ============================================================================

#[test]
fn roundtrip_lossless_rgb_gradient_8bit() {
    let w = 64;
    let h = 64;
    let comps = gen_gradient_rgb(w, h, 8);
    let j2k = encode_multicomp(w, h, 3, 8, false, true, true, 5, 64, 64, 0, 0, 0.0, &comps);
    let decoded = decode_multicomp(&j2k, w, h, 3);
    verify_lossless(w, h, 8, false, &comps, &decoded);
}

#[test]
fn roundtrip_lossy_rgb_gradient_8bit() {
    let w = 64;
    let h = 64;
    let comps = gen_gradient_rgb(w, h, 8);
    let j2k = encode_multicomp(
        w, h, 3, 8, false, false, true, 5, 64, 64, 0, 0, 0.01, &comps,
    );
    let decoded = decode_multicomp(&j2k, w, h, 3);
    // 1% of range² = 0.01 * 255² ≈ 650
    verify_lossy(w, h, 8, false, &comps, &decoded, 0.01, 255);
}

// ============================================================================
// 16-bit roundtrip tests
// ============================================================================

#[test]
fn roundtrip_lossless_gray_16bit() {
    let w = 32;
    let h = 32;
    let comps = gen_gradient_gray(w, h, 16);
    let j2k = encode_multicomp(
        w, h, 1, 16, false, true, false, 5, 64, 64, 0, 0, 0.0, &comps,
    );
    let decoded = decode_multicomp(&j2k, w, h, 1);
    verify_lossless(w, h, 16, false, &comps, &decoded);
}

#[test]
fn roundtrip_lossless_rgb_16bit() {
    let w = 32;
    let h = 32;
    let comps = gen_gradient_rgb(w, h, 16);
    let j2k = encode_multicomp(w, h, 3, 16, false, true, true, 5, 64, 64, 0, 0, 0.0, &comps);
    let decoded = decode_multicomp(&j2k, w, h, 3);
    verify_lossless(w, h, 16, false, &comps, &decoded);
}

#[test]
fn roundtrip_lossy_gray_16bit() {
    let w = 32;
    let h = 32;
    let comps = gen_gradient_gray(w, h, 16);
    let j2k = encode_multicomp(
        w, h, 1, 16, false, false, false, 5, 64, 64, 0, 0, 0.01, &comps,
    );
    let decoded = decode_multicomp(&j2k, w, h, 1);
    verify_lossy(w, h, 16, false, &comps, &decoded, 0.01, 65535);
}

// ============================================================================
// 10-bit and 12-bit roundtrip tests
// ============================================================================

#[test]
fn roundtrip_lossless_gray_10bit() {
    let w = 32;
    let h = 32;
    let comps = gen_gradient_gray(w, h, 10);
    let j2k = encode_multicomp(
        w, h, 1, 10, false, true, false, 4, 64, 64, 0, 0, 0.0, &comps,
    );
    let decoded = decode_multicomp(&j2k, w, h, 1);
    verify_lossless(w, h, 10, false, &comps, &decoded);
}

#[test]
fn roundtrip_lossless_gray_12bit() {
    let w = 32;
    let h = 32;
    let comps = gen_gradient_gray(w, h, 12);
    let j2k = encode_multicomp(
        w, h, 1, 12, false, true, false, 4, 64, 64, 0, 0, 0.0, &comps,
    );
    let decoded = decode_multicomp(&j2k, w, h, 1);
    verify_lossless(w, h, 12, false, &comps, &decoded);
}

#[test]
fn roundtrip_lossless_rgb_10bit() {
    let w = 32;
    let h = 32;
    let comps = gen_gradient_rgb(w, h, 10);
    let j2k = encode_multicomp(w, h, 3, 10, false, true, true, 4, 64, 64, 0, 0, 0.0, &comps);
    let decoded = decode_multicomp(&j2k, w, h, 3);
    verify_lossless(w, h, 10, false, &comps, &decoded);
}

#[test]
fn roundtrip_lossless_rgb_12bit() {
    let w = 32;
    let h = 32;
    let comps = gen_gradient_rgb(w, h, 12);
    let j2k = encode_multicomp(w, h, 3, 12, false, true, true, 4, 64, 64, 0, 0, 0.0, &comps);
    let decoded = decode_multicomp(&j2k, w, h, 3);
    verify_lossless(w, h, 12, false, &comps, &decoded);
}

// ============================================================================
// Signed sample roundtrip tests
// ============================================================================

#[test]
fn roundtrip_lossless_signed_gray_8bit() {
    let w = 32;
    let h = 32;
    let comps = gen_signed_gray(w, h, 8);
    let j2k = encode_multicomp(w, h, 1, 8, true, true, false, 4, 64, 64, 0, 0, 0.0, &comps);
    let decoded = decode_multicomp(&j2k, w, h, 1);
    verify_lossless(w, h, 8, true, &comps, &decoded);
}

#[test]
fn roundtrip_lossless_signed_gray_16bit() {
    let w = 32;
    let h = 32;
    let comps = gen_signed_gray(w, h, 16);
    let j2k = encode_multicomp(w, h, 1, 16, true, true, false, 4, 64, 64, 0, 0, 0.0, &comps);
    let decoded = decode_multicomp(&j2k, w, h, 1);
    verify_lossless(w, h, 16, true, &comps, &decoded);
}

// ============================================================================
// Tiled roundtrip tests
// ============================================================================

#[test]
fn roundtrip_lossless_tiles_rgb() {
    let w = 64;
    let h = 64;
    let comps = gen_gradient_rgb(w, h, 8);
    let j2k = encode_multicomp(
        w, h, 3, 8, false, true, true, 3, 64, 64, 33, 33, 0.0, &comps,
    );
    let decoded = decode_multicomp(&j2k, w, h, 3);
    verify_lossless(w, h, 8, false, &comps, &decoded);
}

#[test]
fn roundtrip_lossy_tiles_rgb() {
    let w = 64;
    let h = 64;
    let comps = gen_gradient_rgb(w, h, 8);
    let j2k = encode_multicomp(
        w, h, 3, 8, false, false, true, 3, 64, 64, 33, 33, 0.01, &comps,
    );
    let decoded = decode_multicomp(&j2k, w, h, 3);
    verify_lossy(w, h, 8, false, &comps, &decoded, 0.01, 255);
}

#[test]
fn roundtrip_lossless_tiles_gray() {
    let w = 64;
    let h = 64;
    let comps = gen_gradient_gray(w, h, 8);
    let j2k = encode_multicomp(
        w, h, 1, 8, false, true, false, 3, 64, 64, 33, 33, 0.0, &comps,
    );
    let decoded = decode_multicomp(&j2k, w, h, 1);
    verify_lossless(w, h, 8, false, &comps, &decoded);
}

// ============================================================================
// Decomposition level sweep
// ============================================================================

#[test]
fn roundtrip_lossless_decomp_0() {
    let w = 32;
    let h = 32;
    let comps = gen_gradient_gray(w, h, 8);
    let j2k = encode_multicomp(w, h, 1, 8, false, true, false, 0, 64, 64, 0, 0, 0.0, &comps);
    let decoded = decode_multicomp(&j2k, w, h, 1);
    verify_lossless(w, h, 8, false, &comps, &decoded);
}

#[test]
fn roundtrip_lossless_decomp_1() {
    let w = 32;
    let h = 32;
    let comps = gen_gradient_gray(w, h, 8);
    let j2k = encode_multicomp(w, h, 1, 8, false, true, false, 1, 64, 64, 0, 0, 0.0, &comps);
    let decoded = decode_multicomp(&j2k, w, h, 1);
    verify_lossless(w, h, 8, false, &comps, &decoded);
}

#[test]
fn roundtrip_lossless_decomp_3() {
    let w = 64;
    let h = 64;
    let comps = gen_gradient_gray(w, h, 8);
    let j2k = encode_multicomp(w, h, 1, 8, false, true, false, 3, 64, 64, 0, 0, 0.0, &comps);
    let decoded = decode_multicomp(&j2k, w, h, 1);
    verify_lossless(w, h, 8, false, &comps, &decoded);
}

#[test]
fn roundtrip_lossless_decomp_5() {
    let w = 64;
    let h = 64;
    let comps = gen_gradient_gray(w, h, 8);
    let j2k = encode_multicomp(w, h, 1, 8, false, true, false, 5, 64, 64, 0, 0, 0.0, &comps);
    let decoded = decode_multicomp(&j2k, w, h, 1);
    verify_lossless(w, h, 8, false, &comps, &decoded);
}

// ============================================================================
// Block size sweep (lossless)
// ============================================================================

#[test]
fn roundtrip_lossless_block_4x4() {
    let w = 32;
    let h = 32;
    let comps = gen_gradient_gray(w, h, 8);
    let j2k = encode_multicomp(w, h, 1, 8, false, true, false, 3, 4, 4, 0, 0, 0.0, &comps);
    let decoded = decode_multicomp(&j2k, w, h, 1);
    verify_lossless(w, h, 8, false, &comps, &decoded);
}

#[test]
fn roundtrip_lossless_block_16x16() {
    let w = 32;
    let h = 32;
    let comps = gen_gradient_gray(w, h, 8);
    let j2k = encode_multicomp(w, h, 1, 8, false, true, false, 3, 16, 16, 0, 0, 0.0, &comps);
    let decoded = decode_multicomp(&j2k, w, h, 1);
    verify_lossless(w, h, 8, false, &comps, &decoded);
}

#[test]
fn roundtrip_lossless_block_32x32() {
    let w = 32;
    let h = 32;
    let comps = gen_gradient_gray(w, h, 8);
    let j2k = encode_multicomp(w, h, 1, 8, false, true, false, 3, 32, 32, 0, 0, 0.0, &comps);
    let decoded = decode_multicomp(&j2k, w, h, 1);
    verify_lossless(w, h, 8, false, &comps, &decoded);
}

// ============================================================================
// Odd-dimension roundtrip tests
// ============================================================================

#[test]
fn roundtrip_lossless_odd_7x93() {
    let w = 7;
    let h = 93;
    let comps = gen_gradient_gray(w, h, 8);
    let j2k = encode_multicomp(w, h, 1, 8, false, true, false, 3, 64, 64, 0, 0, 0.0, &comps);
    let decoded = decode_multicomp(&j2k, w, h, 1);
    verify_lossless(w, h, 8, false, &comps, &decoded);
}

#[test]
fn roundtrip_lossless_odd_127x93_rgb() {
    let w = 127;
    let h = 93;
    let comps = gen_gradient_rgb(w, h, 8);
    let j2k = encode_multicomp(w, h, 3, 8, false, true, true, 5, 64, 64, 0, 0, 0.0, &comps);
    let decoded = decode_multicomp(&j2k, w, h, 3);
    verify_lossless(w, h, 8, false, &comps, &decoded);
}

#[test]
fn roundtrip_lossless_odd_1x1() {
    let w = 1;
    let h = 1;
    let comps = vec![vec![42i32]];
    let j2k = encode_multicomp(w, h, 1, 8, false, true, false, 0, 64, 64, 0, 0, 0.0, &comps);
    let decoded = decode_multicomp(&j2k, w, h, 1);
    verify_lossless(w, h, 8, false, &comps, &decoded);
}

#[test]
fn roundtrip_lossless_odd_3x3() {
    let w = 3;
    let h = 3;
    let comps = vec![vec![10, 20, 30, 40, 50, 60, 70, 80, 90]];
    let j2k = encode_multicomp(w, h, 1, 8, false, true, false, 1, 64, 64, 0, 0, 0.0, &comps);
    let decoded = decode_multicomp(&j2k, w, h, 1);
    verify_lossless(w, h, 8, false, &comps, &decoded);
}

// ============================================================================
// Various quality step (qstep) values for lossy
// ============================================================================

#[test]
fn roundtrip_lossy_qstep_001() {
    let w = 64;
    let h = 64;
    let comps = gen_gradient_rgb(w, h, 8);
    let j2k = encode_multicomp(
        w, h, 3, 8, false, false, true, 5, 64, 64, 0, 0, 0.001, &comps,
    );
    let decoded = decode_multicomp(&j2k, w, h, 3);
    verify_lossy(w, h, 8, false, &comps, &decoded, 0.001, 128);
}

#[test]
fn roundtrip_lossy_qstep_01() {
    let w = 64;
    let h = 64;
    let comps = gen_gradient_rgb(w, h, 8);
    let j2k = encode_multicomp(
        w, h, 3, 8, false, false, true, 5, 64, 64, 0, 0, 0.01, &comps,
    );
    let decoded = decode_multicomp(&j2k, w, h, 3);
    verify_lossy(w, h, 8, false, &comps, &decoded, 0.01, 255);
}

#[test]
fn roundtrip_lossy_qstep_1() {
    let w = 64;
    let h = 64;
    let comps = gen_gradient_rgb(w, h, 8);
    let j2k = encode_multicomp(w, h, 3, 8, false, false, true, 5, 64, 64, 0, 0, 0.1, &comps);
    let decoded = decode_multicomp(&j2k, w, h, 3);
    verify_lossy(w, h, 8, false, &comps, &decoded, 0.05, 255);
}

// ============================================================================
// Color transform on/off
// ============================================================================

#[test]
fn roundtrip_lossless_rgb_no_color_transform() {
    let w = 32;
    let h = 32;
    let comps = gen_gradient_rgb(w, h, 8);
    let j2k = encode_multicomp(w, h, 3, 8, false, true, false, 3, 64, 64, 0, 0, 0.0, &comps);
    let decoded = decode_multicomp(&j2k, w, h, 3);
    verify_lossless(w, h, 8, false, &comps, &decoded);
}

#[test]
fn roundtrip_lossless_rgb_with_color_transform() {
    let w = 32;
    let h = 32;
    let comps = gen_gradient_rgb(w, h, 8);
    let j2k = encode_multicomp(w, h, 3, 8, false, true, true, 3, 64, 64, 0, 0, 0.0, &comps);
    let decoded = decode_multicomp(&j2k, w, h, 3);
    verify_lossless(w, h, 8, false, &comps, &decoded);
}

// ============================================================================
// Larger images (256x256) — realistic workload
// ============================================================================

#[test]
fn roundtrip_lossless_256x256_rgb() {
    let w = 256;
    let h = 256;
    let comps = gen_gradient_rgb(w, h, 8);
    let j2k = encode_multicomp(w, h, 3, 8, false, true, true, 5, 64, 64, 0, 0, 0.0, &comps);
    let decoded = decode_multicomp(&j2k, w, h, 3);
    verify_lossless(w, h, 8, false, &comps, &decoded);
}

#[test]
fn roundtrip_lossy_256x256_rgb() {
    let w = 256;
    let h = 256;
    let comps = gen_gradient_rgb(w, h, 8);
    let j2k = encode_multicomp(
        w, h, 3, 8, false, false, true, 5, 64, 64, 0, 0, 0.01, &comps,
    );
    let decoded = decode_multicomp(&j2k, w, h, 3);
    verify_lossy(w, h, 8, false, &comps, &decoded, 0.01, 255);
}

// ============================================================================
// Header parameter verification
// ============================================================================

#[test]
fn roundtrip_verify_headers() {
    let w = 32;
    let h = 32;
    let comps = gen_gradient_gray(w, h, 8);
    let j2k = encode_multicomp(w, h, 1, 8, false, true, false, 3, 32, 32, 0, 0, 0.0, &comps);

    // Read back and verify header parameters
    let mut cs = Codestream::new();
    let mut infile = MemInfile::new(&j2k);
    cs.read_headers(&mut infile).unwrap();

    let siz = cs.access_siz();
    assert_eq!(siz.get_num_components(), 1);
    assert_eq!(siz.get_bit_depth(0), 8);
    assert!(!siz.is_signed(0));
    let extent = siz.get_image_extent();
    assert_eq!(extent.x, w);
    assert_eq!(extent.y, h);

    let cod = cs.access_cod();
    assert!(cod.is_reversible());
    assert_eq!(cod.get_num_decompositions(), 3);
}

#[test]
fn roundtrip_verify_headers_rgb_16bit() {
    let w = 64;
    let h = 48;
    let comps = gen_gradient_rgb(w, h, 16);
    let j2k = encode_multicomp(w, h, 3, 16, false, true, true, 5, 64, 64, 0, 0, 0.0, &comps);

    let mut cs = Codestream::new();
    let mut infile = MemInfile::new(&j2k);
    cs.read_headers(&mut infile).unwrap();

    let siz = cs.access_siz();
    assert_eq!(siz.get_num_components(), 3);
    for c in 0..3 {
        assert_eq!(siz.get_bit_depth(c), 16);
        assert!(!siz.is_signed(c));
    }

    let cod = cs.access_cod();
    assert!(cod.is_reversible());
    assert_eq!(cod.get_num_decompositions(), 5);
    assert!(cod.is_employing_color_transform());
}
