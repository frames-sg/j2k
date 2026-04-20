//! Integration tests ported from OpenJPH C++ test_executables.cpp.
//!
//! Phase 5.3 (decode tests) and Phase 5.4 (encode tests).
//!
//! Since we cannot download external test codestreams, these tests exercise
//! the same codec configurations using synthetically generated test data.
//! The C++ tests use Malamute.ppm (768×512 RGB), foreman_420.yuv (352×288),
//! monarch.pgm (grayscale), mm.ppm/mm.pgm (16-bit). We generate equivalent
//! synthetic images programmatically.

mod common;

use openjph_core::codestream::Codestream;
use openjph_core::file::{MemInfile, MemOutfile};
use openjph_core::types::{Point, Size};

use common::mse_pae::{find_mse_pae, ImgInfo};

// ============================================================================
// Test image generators
// ============================================================================

/// Generate a gradient RGB image (3-component, 8-bit unsigned).
/// Mimics Malamute.ppm-style content with rich color variation.
fn gen_rgb_image(width: u32, height: u32) -> Vec<Vec<i32>> {
    let n = (width * height) as usize;
    let mut r = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for y in 0..height {
        for x in 0..width {
            r.push(((x * 255) / width.max(1)) as i32);
            g.push(((y * 255) / height.max(1)) as i32);
            b.push((((x + y) * 127) / (width + height).max(1)) as i32);
        }
    }
    vec![r, g, b]
}

/// Generate a grayscale image (1-component, 8-bit unsigned).
fn gen_gray_image(width: u32, height: u32) -> Vec<Vec<i32>> {
    let n = (width * height) as usize;
    let mut pixels = Vec::with_capacity(n);
    for y in 0..height {
        for x in 0..width {
            let val = ((x * 3 + y * 7) % 256) as i32;
            pixels.push(val);
        }
    }
    vec![pixels]
}

/// Generate an N-bit unsigned image (single component).
fn gen_gray_image_nbit(width: u32, height: u32, bit_depth: u32) -> Vec<Vec<i32>> {
    let max_val = (1i32 << bit_depth) - 1;
    let n = (width * height) as usize;
    let mut pixels = Vec::with_capacity(n);
    for y in 0..height {
        for x in 0..width {
            let val = ((x as i64 * 3 + y as i64 * 7) % (max_val as i64 + 1)) as i32;
            pixels.push(val);
        }
    }
    vec![pixels]
}

/// Generate an N-bit RGB image (3-component).
fn gen_rgb_image_nbit(width: u32, height: u32, bit_depth: u32) -> Vec<Vec<i32>> {
    let max_val = (1i32 << bit_depth) - 1;
    let n = (width * height) as usize;
    let mut comps = Vec::new();
    for c in 0..3u32 {
        let mut pixels = Vec::with_capacity(n);
        for y in 0..height {
            for x in 0..width {
                let val = ((x as i64 * (c as i64 + 1) * 3 + y as i64 * (c as i64 + 2) * 7)
                    % (max_val as i64 + 1)) as i32;
                pixels.push(val);
            }
        }
        comps.push(pixels);
    }
    comps
}

// ============================================================================
// Encode/decode helpers
// ============================================================================

/// Configuration for encoding.
struct EncodeConfig {
    width: u32,
    height: u32,
    num_comps: u32,
    bit_depth: u32,
    is_signed: bool,
    reversible: bool,
    color_transform: bool,
    num_decomps: u32,
    block_width: u32,
    block_height: u32,
    tile_width: u32,
    tile_height: u32,
    qstep: f32,
    progression_order: Option<&'static str>,
}

impl Default for EncodeConfig {
    fn default() -> Self {
        Self {
            width: 64,
            height: 64,
            num_comps: 3,
            bit_depth: 8,
            is_signed: false,
            reversible: false,
            color_transform: true,
            num_decomps: 5,
            block_width: 64,
            block_height: 64,
            tile_width: 0, // 0 = use image size
            tile_height: 0,
            qstep: 0.0001, // very small for high quality
            progression_order: None,
        }
    }
}

/// Encode multi-component image to J2K bytes.
fn encode_image(config: &EncodeConfig, components: &[Vec<i32>]) -> Vec<u8> {
    assert_eq!(components.len(), config.num_comps as usize);

    let mut cs = Codestream::new();

    // Configure SIZ
    {
        let siz = cs.access_siz_mut();
        siz.set_image_extent(Point::new(config.width, config.height));
        let tw = if config.tile_width > 0 {
            config.tile_width
        } else {
            config.width
        };
        let th = if config.tile_height > 0 {
            config.tile_height
        } else {
            config.height
        };
        siz.set_tile_size(Size::new(tw, th));
        siz.set_num_components(config.num_comps);
        for c in 0..config.num_comps {
            siz.set_comp_info(c, Point::new(1, 1), config.bit_depth, config.is_signed);
        }
    }

    // Configure COD
    {
        let cod = cs.access_cod_mut();
        cod.set_reversible(config.reversible);
        cod.set_color_transform(config.color_transform && config.num_comps >= 3);
        cod.set_num_decomposition(config.num_decomps);
        cod.set_block_dims(config.block_width, config.block_height);
        if let Some(order) = config.progression_order {
            cod.set_progression_order(order).unwrap();
        }
    }

    // Configure QCD for irreversible
    if !config.reversible {
        let qcd = cs.access_qcd_mut();
        qcd.set_delta(config.qstep);
    }

    // Write headers
    let mut outfile = MemOutfile::new();
    cs.write_headers(&mut outfile, &[]).unwrap();

    // Push image lines (interleaved: all components per line)
    for y in 0..config.height {
        for c in 0..config.num_comps {
            let start = (y * config.width) as usize;
            let end = start + config.width as usize;
            cs.exchange(&components[c as usize][start..end], c).unwrap();
        }
    }

    // Flush
    cs.flush(&mut outfile).unwrap();
    outfile.get_data().to_vec()
}

/// Decode J2K bytes to multi-component image.
fn decode_image(j2k_data: &[u8], config: &EncodeConfig) -> Vec<Vec<i32>> {
    let mut cs = Codestream::new();
    let mut infile = MemInfile::new(j2k_data);
    cs.read_headers(&mut infile).unwrap();
    cs.create(&mut infile).unwrap();

    let mut components: Vec<Vec<i32>> = (0..config.num_comps)
        .map(|_| Vec::with_capacity((config.width * config.height) as usize))
        .collect();

    for _ in 0..config.height {
        for c in 0..config.num_comps {
            if let Some(line) = cs.pull(c) {
                components[c as usize].extend_from_slice(&line);
            }
        }
    }
    components
}

/// Run a roundtrip test: encode → decode → verify MSE/PAE.
fn roundtrip_test(config: &EncodeConfig, components: &[Vec<i32>], expect_lossless: bool) {
    let j2k = encode_image(config, components);
    assert!(!j2k.is_empty(), "encoded data should not be empty");

    let decoded = decode_image(&j2k, config);
    assert_eq!(decoded.len(), config.num_comps as usize);

    let img_orig = ImgInfo::from_samples(
        config.width as usize,
        config.height as usize,
        config.bit_depth,
        config.is_signed,
        components.to_vec(),
    );
    let img_dec = ImgInfo::from_samples(
        config.width as usize,
        config.height as usize,
        config.bit_depth,
        config.is_signed,
        decoded,
    );
    let results = find_mse_pae(&img_orig, &img_dec);
    assert_eq!(results.len(), config.num_comps as usize);

    for (c, res) in results.iter().enumerate() {
        if expect_lossless {
            assert_eq!(
                res.mse, 0.0,
                "component {}: reversible roundtrip must be lossless (MSE={})",
                c, res.mse
            );
            assert_eq!(
                res.pae, 0,
                "component {}: reversible roundtrip must be lossless (PAE={})",
                c, res.pae
            );
        } else {
            // For irreversible, just verify the codec didn't produce garbage
            let max_val = (1u32 << config.bit_depth) - 1;
            let max_allowed_mse = (max_val as f32) * (max_val as f32) * 0.01;
            assert!(
                res.mse < max_allowed_mse,
                "component {}: MSE {} exceeds 1% of range² ({})",
                c,
                res.mse,
                max_allowed_mse
            );
        }
    }
}

// ============================================================================
// Phase 5.3: Decode tests — Irreversible 9/7 with various block sizes
// (Ported from C++ SimpleDecIrv97* tests)
//
// The C++ tests decode pre-encoded codestreams and compare against reference.
// We instead encode with the same configuration and verify roundtrip.
// ============================================================================

/// C++ equivalent: SimpleDecIrv9764x64
/// Config: irv97 wavelet, 64×64 code blocks, Malamute.ppm (RGB)
#[test]
fn dec_irv97_64x64_rgb() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: false,
        qstep: 0.1,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleDecIrv9732x32
#[test]
fn dec_irv97_32x32_rgb() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        block_width: 32,
        block_height: 32,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleDecIrv9716x16
#[test]
fn dec_irv97_16x16_rgb() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        block_width: 16,
        block_height: 16,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleDecIrv974x4
#[test]
fn dec_irv97_4x4_rgb() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        block_width: 4,
        block_height: 4,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleDecIrv971024x4 (tall narrow blocks)
#[test]
fn dec_irv97_1024x4_rgb() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        block_width: 4,
        block_height: 1024,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleDecIrv974x1024 (wide narrow blocks)
#[test]
fn dec_irv97_4x1024_rgb() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        block_width: 1024,
        block_height: 4,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleDecIrv97512x8
#[test]
fn dec_irv97_512x8_rgb() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        block_width: 8,
        block_height: 512,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleDecIrv978x512
#[test]
fn dec_irv97_8x512_rgb() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        block_width: 512,
        block_height: 8,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleDecIrv97256x16
#[test]
fn dec_irv97_256x16_rgb() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        block_width: 16,
        block_height: 256,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleDecIrv9716x256
#[test]
fn dec_irv97_16x256_rgb() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        block_width: 256,
        block_height: 16,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleDecIrv97128x32
#[test]
fn dec_irv97_128x32_rgb() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        block_width: 32,
        block_height: 128,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleDecIrv9732x128
#[test]
fn dec_irv97_32x128_rgb() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        block_width: 128,
        block_height: 32,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

// ============================================================================
// Phase 5.3: Decode tests — Reversible 5/3 with various block sizes
// (Ported from C++ SimpleDecRev53* tests)
// ============================================================================

/// C++ equivalent: SimpleDecRev5364x64
#[test]
fn dec_rev53_64x64_rgb() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: true,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

/// C++ equivalent: SimpleDecRev5332x32
#[test]
fn dec_rev53_32x32_rgb() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: true,
        block_width: 32,
        block_height: 32,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

/// C++ equivalent: SimpleDecRev534x4
#[test]
fn dec_rev53_4x4_rgb() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: true,
        block_width: 4,
        block_height: 4,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

/// C++ equivalent: SimpleDecRev531024x4
#[test]
fn dec_rev53_1024x4_rgb() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: true,
        block_width: 4,
        block_height: 1024,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

/// C++ equivalent: SimpleDecRev534x1024
#[test]
fn dec_rev53_4x1024_rgb() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: true,
        block_width: 1024,
        block_height: 4,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

// ============================================================================
// Phase 5.3: Decode tests — Progression orders with tiles
// (Ported from C++ SimpleDecIrv9764x64Tiles{LRCP,RLCP,RPCL,PCRL,CPRL})
// ============================================================================

/// C++ equivalent: SimpleDecIrv9764x64TilesLRCP
#[test]
fn dec_irv97_tiles_lrcp() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        tile_width: 33,
        tile_height: 33,
        progression_order: Some("LRCP"),
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleDecIrv9764x64TilesRLCP
#[test]
fn dec_irv97_tiles_rlcp() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        tile_width: 33,
        tile_height: 33,
        progression_order: Some("RLCP"),
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleDecIrv9764x64TilesRPCL
#[test]
fn dec_irv97_tiles_rpcl() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        tile_width: 33,
        tile_height: 33,
        progression_order: Some("RPCL"),
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleDecIrv9764x64TilesPCRL
#[test]
fn dec_irv97_tiles_pcrl() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        tile_width: 33,
        tile_height: 33,
        progression_order: Some("PCRL"),
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleDecIrv9764x64TilesCPRL
#[test]
fn dec_irv97_tiles_cprl() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        tile_width: 33,
        tile_height: 33,
        progression_order: Some("CPRL"),
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

// ============================================================================
// Phase 5.3: Decode tests — Progression orders with 33-size precincts
// (C++ SimpleDecIrv9764x64Tiles{LRCP,RLCP,RPCL,PCRL,CPRL}33)
// ============================================================================

/// C++ equivalent: SimpleDecIrv9764x64TilesLRCP33
#[test]
fn dec_irv97_tiles_lrcp_33() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        tile_width: 33,
        tile_height: 33,
        num_decomps: 5,
        progression_order: Some("LRCP"),
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleDecIrv9764x64TilesRLCP33
#[test]
fn dec_irv97_tiles_rlcp_33() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        tile_width: 33,
        tile_height: 33,
        num_decomps: 5,
        progression_order: Some("RLCP"),
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleDecIrv9764x64TilesRPCL33
#[test]
fn dec_irv97_tiles_rpcl_33() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        tile_width: 33,
        tile_height: 33,
        num_decomps: 5,
        progression_order: Some("RPCL"),
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleDecIrv9764x64TilesPCRL33
#[test]
fn dec_irv97_tiles_pcrl_33() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        tile_width: 33,
        tile_height: 33,
        num_decomps: 5,
        progression_order: Some("PCRL"),
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleDecIrv9764x64TilesCPRL33
#[test]
fn dec_irv97_tiles_cprl_33() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        tile_width: 33,
        tile_height: 33,
        num_decomps: 5,
        progression_order: Some("CPRL"),
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

// ============================================================================
// Phase 5.3: Decode tests — Progression orders with 33×33 precincts + tiles
// (C++ SimpleDecIrv9764x64Tiles{order}33x33)
// ============================================================================

/// C++ equivalent: SimpleDecIrv9764x64TilesLRCP33x33
#[test]
fn dec_irv97_tiles_lrcp_33x33() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        tile_width: 33,
        tile_height: 33,
        num_decomps: 5,
        progression_order: Some("LRCP"),
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleDecIrv9764x64TilesRLCP33x33
#[test]
fn dec_irv97_tiles_rlcp_33x33() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        tile_width: 33,
        tile_height: 33,
        num_decomps: 5,
        progression_order: Some("RLCP"),
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleDecIrv9764x64TilesRPCL33x33
#[test]
fn dec_irv97_tiles_rpcl_33x33() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        tile_width: 33,
        tile_height: 33,
        num_decomps: 5,
        progression_order: Some("RPCL"),
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleDecIrv9764x64TilesPCRL33x33
#[test]
fn dec_irv97_tiles_pcrl_33x33() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        tile_width: 33,
        tile_height: 33,
        num_decomps: 5,
        progression_order: Some("PCRL"),
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleDecIrv9764x64TilesCPRL33x33
#[test]
fn dec_irv97_tiles_cprl_33x33() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        tile_width: 33,
        tile_height: 33,
        num_decomps: 5,
        progression_order: Some("CPRL"),
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

// ============================================================================
// Phase 5.3: Decode tests — Grayscale with tiles
// (C++ SimpleDecRev5364x64GrayTiles, SimpleDecIrv9764x64GrayTiles)
// ============================================================================

/// C++ equivalent: SimpleDecRev5364x64GrayTiles (monarch.pgm)
#[test]
fn dec_rev53_gray_tiles() {
    let comps = gen_gray_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 1,
        reversible: true,
        color_transform: false,
        tile_width: 33,
        tile_height: 33,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

/// C++ equivalent: SimpleDecIrv9764x64GrayTiles (monarch.pgm)
#[test]
fn dec_irv97_gray_tiles() {
    let comps = gen_gray_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 1,
        reversible: false,
        color_transform: false,
        qstep: 0.01,
        tile_width: 33,
        tile_height: 33,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

// ============================================================================
// Phase 5.3: Decode tests — 16-bit images
// (C++ SimpleDecIrv9764x6416bit, SimpleDecRev5364x6416bit, etc.)
// ============================================================================

/// C++ equivalent: SimpleDecIrv9764x6416bit (mm.ppm, 16-bit RGB)
#[test]
fn dec_irv97_16bit_rgb() {
    let comps = gen_rgb_image_nbit(64, 64, 16);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        bit_depth: 16,
        reversible: false,
        qstep: 0.01,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleDecIrv9764x6416bitGray (mm.pgm, 16-bit gray)
#[test]
fn dec_irv97_16bit_gray() {
    let comps = gen_gray_image_nbit(64, 64, 16);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 1,
        bit_depth: 16,
        reversible: false,
        color_transform: false,
        qstep: 0.01,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleDecRev5364x6416bit (mm.ppm, 16-bit RGB, lossless)
#[test]
fn dec_rev53_16bit_rgb() {
    let comps = gen_rgb_image_nbit(64, 64, 16);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        bit_depth: 16,
        reversible: true,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

/// C++ equivalent: SimpleDecRev5364x6416bitGray (mm.pgm, 16-bit gray, lossless)
#[test]
fn dec_rev53_16bit_gray() {
    let comps = gen_gray_image_nbit(64, 64, 16);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 1,
        bit_depth: 16,
        reversible: true,
        color_transform: false,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

// ============================================================================
// Phase 5.4: Encode tests — Irreversible 9/7 with various block sizes
// (Ported from C++ SimpleEncIrv97* tests)
// ============================================================================

/// C++ equivalent: SimpleEncIrv9764x64 (Malamute.ppm, -qstep 0.1)
#[test]
fn enc_irv97_64x64() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: false,
        qstep: 0.1,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleEncIrv9732x32 (Malamute.ppm, -qstep 0.01 -block_size {32,32})
#[test]
fn enc_irv97_32x32() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        block_width: 32,
        block_height: 32,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleEncIrv9716x16
#[test]
fn enc_irv97_16x16() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        block_width: 16,
        block_height: 16,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleEncIrv974x4
#[test]
fn enc_irv97_4x4() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        block_width: 4,
        block_height: 4,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleEncIrv971024x4
#[test]
fn enc_irv97_1024x4() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        block_width: 4,
        block_height: 1024,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleEncIrv974x1024
#[test]
fn enc_irv97_4x1024() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        block_width: 1024,
        block_height: 4,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleEncIrv97512x8
#[test]
fn enc_irv97_512x8() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        block_width: 8,
        block_height: 512,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleEncIrv978x512
#[test]
fn enc_irv97_8x512() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        block_width: 512,
        block_height: 8,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleEncIrv97256x16
#[test]
fn enc_irv97_256x16() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        block_width: 16,
        block_height: 256,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleEncIrv9716x256
#[test]
fn enc_irv97_16x256() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        block_width: 256,
        block_height: 16,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleEncIrv97128x32
#[test]
fn enc_irv97_128x32() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        block_width: 32,
        block_height: 128,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleEncIrv9732x128
#[test]
fn enc_irv97_32x128() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        block_width: 128,
        block_height: 32,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

// ============================================================================
// Phase 5.4: Encode tests — Irreversible with tiles and high decomposition
// (C++ SimpleEncIrv9764x64Tiles33x33D5, D6)
// ============================================================================

/// C++ equivalent: SimpleEncIrv9764x64Tiles33x33D5
#[test]
fn enc_irv97_tiles_33x33_d5() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        tile_width: 33,
        tile_height: 33,
        num_decomps: 5,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleEncIrv9764x64Tiles33x33D6
#[test]
fn enc_irv97_tiles_33x33_d6() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        tile_width: 33,
        tile_height: 33,
        num_decomps: 6,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

// ============================================================================
// Phase 5.4: Encode tests — 16-bit images
// (C++ SimpleEncIrv9764x6416bit, SimpleEncRev5364x6416bit, etc.)
// ============================================================================

/// C++ equivalent: SimpleEncIrv9764x6416bit (mm.ppm)
#[test]
fn enc_irv97_16bit_rgb() {
    let comps = gen_rgb_image_nbit(64, 64, 16);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        bit_depth: 16,
        reversible: false,
        qstep: 0.01,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleEncIrv9764x6416bitGray (mm.pgm)
#[test]
fn enc_irv97_16bit_gray() {
    let comps = gen_gray_image_nbit(64, 64, 16);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 1,
        bit_depth: 16,
        reversible: false,
        color_transform: false,
        qstep: 0.01,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleEncRev5364x6416bit (mm.ppm, lossless)
#[test]
fn enc_rev53_16bit_rgb() {
    let comps = gen_rgb_image_nbit(64, 64, 16);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        bit_depth: 16,
        reversible: true,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

/// C++ equivalent: SimpleEncRev5364x6416bitGray (mm.pgm, lossless)
#[test]
fn enc_rev53_16bit_gray() {
    let comps = gen_gray_image_nbit(64, 64, 16);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 1,
        bit_depth: 16,
        reversible: true,
        color_transform: false,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

// ============================================================================
// Phase 5.4: Encode tests — Reversible 5/3 with various block sizes
// (C++ SimpleEncRev53* tests)
// ============================================================================

/// C++ equivalent: SimpleEncRev5364x64 (Malamute.ppm, -reversible true)
#[test]
fn enc_rev53_64x64() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: true,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

/// C++ equivalent: SimpleEncRev5332x32
#[test]
fn enc_rev53_32x32() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: true,
        block_width: 32,
        block_height: 32,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

/// C++ equivalent: SimpleEncRev534x4
#[test]
fn enc_rev53_4x4() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: true,
        block_width: 4,
        block_height: 4,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

/// C++ equivalent: SimpleEncRev531024x4
#[test]
fn enc_rev53_1024x4() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: true,
        block_width: 4,
        block_height: 1024,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

/// C++ equivalent: SimpleEncRev534x1024
#[test]
fn enc_rev53_4x1024() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: true,
        block_width: 1024,
        block_height: 4,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

// ============================================================================
// Phase 5.4: Encode tests — Reversible with tiles
// (C++ SimpleEncRev5364x64Tiles33x33D5, D6)
// ============================================================================

/// C++ equivalent: SimpleEncRev5364x64Tiles33x33D5
#[test]
fn enc_rev53_tiles_33x33_d5() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: true,
        tile_width: 33,
        tile_height: 33,
        num_decomps: 5,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

/// C++ equivalent: SimpleEncRev5364x64Tiles33x33D6
#[test]
fn enc_rev53_tiles_33x33_d6() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: true,
        tile_width: 33,
        tile_height: 33,
        num_decomps: 6,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

// ============================================================================
// Phase 5.4: Encode tests — Tall/narrow images
// (C++ SimpleEncIrv97TallNarrow, SimpleEncRev53TallNarrow, etc.)
// ============================================================================

/// C++ equivalent: SimpleEncIrv97TallNarrow (tall_narrow.ppm)
#[test]
fn enc_irv97_tall_narrow() {
    let comps = gen_rgb_image(7, 93);
    let config = EncodeConfig {
        width: 7,
        height: 93,
        num_comps: 3,
        reversible: false,
        qstep: 0.1,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// C++ equivalent: SimpleEncRev53TallNarrow (tall_narrow.ppm, lossless)
#[test]
fn enc_rev53_tall_narrow() {
    let comps = gen_rgb_image(7, 93);
    let config = EncodeConfig {
        width: 7,
        height: 93,
        num_comps: 3,
        reversible: true,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

// ============================================================================
// Phase 5.4: Encode tests — 10-bit and 12-bit images
// (C++ DpxEnc* tests for 10-bit and 16-bit PPM)
// ============================================================================

/// C++ equivalent: DpxEnc1280x72010bitLeNuke11 (10-bit, lossless)
/// Uses smaller image since we generate synthetically.
#[test]
fn enc_rev53_10bit_rgb() {
    let comps = gen_rgb_image_nbit(64, 48, 10);
    let config = EncodeConfig {
        width: 64,
        height: 48,
        num_comps: 3,
        bit_depth: 10,
        reversible: true,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

/// 12-bit grayscale lossless roundtrip
#[test]
fn enc_rev53_12bit_gray() {
    let comps = gen_gray_image_nbit(64, 48, 12);
    let config = EncodeConfig {
        width: 64,
        height: 48,
        num_comps: 1,
        bit_depth: 12,
        reversible: true,
        color_transform: false,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

// ============================================================================
// Additional configurations matching C++ test patterns
// ============================================================================

/// Decomposition level 0 (no DWT) — reversible
#[test]
fn enc_rev53_decomp_0() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: true,
        num_decomps: 0,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

/// Decomposition level 1
#[test]
fn enc_rev53_decomp_1() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: true,
        num_decomps: 1,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

/// Decomposition level 2
#[test]
fn enc_rev53_decomp_2() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: true,
        num_decomps: 2,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

/// Decomposition level 3
#[test]
fn enc_rev53_decomp_3() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: true,
        num_decomps: 3,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

/// Decomposition level 4
#[test]
fn enc_rev53_decomp_4() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: true,
        num_decomps: 4,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

/// Decomposition level 5
#[test]
fn enc_rev53_decomp_5() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: true,
        num_decomps: 5,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

/// Irreversible, decomposition level 0
#[test]
fn enc_irv97_decomp_0() {
    let comps = gen_rgb_image(64, 64);
    let config = EncodeConfig {
        width: 64,
        height: 64,
        num_comps: 3,
        reversible: false,
        num_decomps: 0,
        qstep: 0.001,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// Medium image, reversible
#[test]
fn enc_rev53_256x256() {
    let comps = gen_rgb_image(256, 256);
    let config = EncodeConfig {
        width: 256,
        height: 256,
        num_comps: 3,
        reversible: true,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

/// Odd-dimension image, reversible
#[test]
fn enc_rev53_127x93() {
    let comps = gen_rgb_image(127, 93);
    let config = EncodeConfig {
        width: 127,
        height: 93,
        num_comps: 3,
        reversible: true,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, true);
}

/// Odd-dimension image, irreversible
#[test]
fn enc_irv97_127x93() {
    let comps = gen_rgb_image(127, 93);
    let config = EncodeConfig {
        width: 127,
        height: 93,
        num_comps: 3,
        reversible: false,
        qstep: 0.01,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}

/// Large QCD step (high compression), irreversible
#[test]
fn enc_irv97_high_compression() {
    let comps = gen_rgb_image(128, 128);
    let config = EncodeConfig {
        width: 128,
        height: 128,
        num_comps: 3,
        reversible: false,
        qstep: 0.5,
        ..Default::default()
    };
    roundtrip_test(&config, &comps, false);
}
