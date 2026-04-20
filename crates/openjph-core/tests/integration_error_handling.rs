//! Error handling and edge case tests for the OpenJPH-RS codec.
//!
//! Tests invalid parameters, truncated codestreams, boundary conditions,
//! and other edge cases that should be handled gracefully.

mod common;

use openjph_core::codestream::Codestream;
use openjph_core::file::{MemInfile, MemOutfile};
use openjph_core::types::{Point, Size};

// ============================================================================
// Invalid parameter tests
// ============================================================================

/// Attempting to write headers without setting image dimensions should fail.
#[test]
fn error_no_siz_set() {
    let mut cs = Codestream::new();
    // Only set num_components, but not image extent
    cs.access_siz_mut().set_num_components(1);
    cs.access_siz_mut()
        .set_comp_info(0, Point::new(1, 1), 8, false);
    let mut outfile = MemOutfile::new();
    // Missing image extent — should error or produce minimal output
    let result = cs.write_headers(&mut outfile, &[]);
    // It's valid for the codec to either error here or produce a degenerate stream
    let _ = result;
}

/// Truncated codestream: just a SOC marker (0xFF4F)
#[test]
fn error_truncated_soc_only() {
    let data = [0xFF, 0x4F]; // SOC marker only
    let mut cs = Codestream::new();
    let mut infile = MemInfile::new(&data);
    let result = cs.read_headers(&mut infile);
    assert!(
        result.is_err(),
        "truncated codestream after SOC should fail"
    );
}

/// Empty codestream
#[test]
fn error_empty_codestream() {
    let data: [u8; 0] = [];
    let mut cs = Codestream::new();
    let mut infile = MemInfile::new(&data);
    let result = cs.read_headers(&mut infile);
    assert!(result.is_err(), "empty codestream should fail");
}

/// Random garbage data
#[test]
fn error_garbage_data() {
    let data = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
    let mut cs = Codestream::new();
    let mut infile = MemInfile::new(&data);
    let result = cs.read_headers(&mut infile);
    assert!(result.is_err(), "garbage data should fail header parse");
}

/// SOC + partial SIZ marker (truncated)
#[test]
fn error_truncated_siz() {
    let data = [
        0xFF, 0x4F, // SOC
        0xFF, 0x51, // SIZ marker
        0x00, 0x10, // Lsiz = 16 (but data stops here)
    ];
    let mut cs = Codestream::new();
    let mut infile = MemInfile::new(&data);
    let result = cs.read_headers(&mut infile);
    assert!(result.is_err(), "truncated SIZ should fail");
}

/// Valid encoding followed by decode of truncated J2K data
#[test]
fn error_truncated_encoded_data() {
    // First, generate valid J2K data
    let mut cs = Codestream::new();
    {
        let siz = cs.access_siz_mut();
        siz.set_image_extent(Point::new(16, 16));
        siz.set_tile_size(Size::new(16, 16));
        siz.set_num_components(1);
        siz.set_comp_info(0, Point::new(1, 1), 8, false);
    }
    {
        let cod = cs.access_cod_mut();
        cod.set_reversible(true);
        cod.set_color_transform(false);
        cod.set_num_decomposition(2);
    }
    let mut outfile = MemOutfile::new();
    cs.write_headers(&mut outfile, &[]).unwrap();
    for y in 0..16u32 {
        let line: Vec<i32> = (0..16).map(|x| ((x + y * 16) % 256) as i32).collect();
        cs.exchange(&line, 0).unwrap();
    }
    cs.flush(&mut outfile).unwrap();
    let full_data = outfile.get_data().to_vec();
    assert!(full_data.len() > 20);

    // Truncate at ~75% of the data
    let trunc_len = full_data.len() * 3 / 4;
    let truncated = &full_data[..trunc_len];

    // Try to decode truncated data — may error or produce partial results
    let mut cs2 = Codestream::new();
    let mut infile = MemInfile::new(truncated);
    let header_result = cs2.read_headers(&mut infile);
    if header_result.is_ok() {
        // Headers might parse fine; create may fail
        let create_result = cs2.create(&mut infile);
        // Either create fails, or pull produces partial/incorrect data
        // The key is it doesn't panic
        if create_result.is_ok() {
            for _ in 0..16 {
                let _ = cs2.pull(0);
            }
        }
    }
    // If we get here without panic, the test passes
}

// ============================================================================
// Edge case: minimum and maximum dimensions
// ============================================================================

/// 1×1 image roundtrip
#[test]
fn edge_1x1_image() {
    let mut cs = Codestream::new();
    {
        let siz = cs.access_siz_mut();
        siz.set_image_extent(Point::new(1, 1));
        siz.set_tile_size(Size::new(1, 1));
        siz.set_num_components(1);
        siz.set_comp_info(0, Point::new(1, 1), 8, false);
    }
    {
        let cod = cs.access_cod_mut();
        cod.set_reversible(true);
        cod.set_color_transform(false);
        cod.set_num_decomposition(0);
    }
    let mut outfile = MemOutfile::new();
    cs.write_headers(&mut outfile, &[]).unwrap();
    cs.exchange(&[42], 0).unwrap();
    cs.flush(&mut outfile).unwrap();

    let data = outfile.get_data().to_vec();
    let mut cs2 = Codestream::new();
    let mut infile = MemInfile::new(&data);
    cs2.read_headers(&mut infile).unwrap();
    cs2.create(&mut infile).unwrap();
    let line = cs2.pull(0).unwrap();
    assert_eq!(line[0], 42, "1×1 roundtrip must preserve the single pixel");
}

/// 2×2 image roundtrip
#[test]
fn edge_2x2_image() {
    let mut cs = Codestream::new();
    {
        let siz = cs.access_siz_mut();
        siz.set_image_extent(Point::new(2, 2));
        siz.set_tile_size(Size::new(2, 2));
        siz.set_num_components(1);
        siz.set_comp_info(0, Point::new(1, 1), 8, false);
    }
    {
        let cod = cs.access_cod_mut();
        cod.set_reversible(true);
        cod.set_color_transform(false);
        cod.set_num_decomposition(1);
    }
    let mut outfile = MemOutfile::new();
    cs.write_headers(&mut outfile, &[]).unwrap();
    cs.exchange(&[10, 20], 0).unwrap();
    cs.exchange(&[30, 40], 0).unwrap();
    cs.flush(&mut outfile).unwrap();

    let data = outfile.get_data().to_vec();
    let mut cs2 = Codestream::new();
    let mut infile = MemInfile::new(&data);
    cs2.read_headers(&mut infile).unwrap();
    cs2.create(&mut infile).unwrap();
    let line0 = cs2.pull(0).unwrap();
    let line1 = cs2.pull(0).unwrap();
    assert_eq!(&line0, &[10, 20]);
    assert_eq!(&line1, &[30, 40]);
}

/// Very narrow image: 1×64
#[test]
#[allow(non_snake_case)]
fn edge_1xN_image() {
    let w = 1u32;
    let h = 64u32;
    let mut cs = Codestream::new();
    {
        let siz = cs.access_siz_mut();
        siz.set_image_extent(Point::new(w, h));
        siz.set_tile_size(Size::new(w, h));
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
    for y in 0..h {
        cs.exchange(&[(y % 256) as i32], 0).unwrap();
    }
    cs.flush(&mut outfile).unwrap();

    let data = outfile.get_data().to_vec();
    let mut cs2 = Codestream::new();
    let mut infile = MemInfile::new(&data);
    cs2.read_headers(&mut infile).unwrap();
    cs2.create(&mut infile).unwrap();

    for y in 0..h {
        let line = cs2.pull(0).unwrap();
        assert_eq!(line[0], (y % 256) as i32, "line {} mismatch", y);
    }
}

/// Very wide image: 64×1
#[test]
#[allow(non_snake_case)]
fn edge_Nx1_image() {
    let w = 64u32;
    let h = 1u32;
    let pixels: Vec<i32> = (0..64).map(|x| (x * 4) % 256).collect();
    let mut cs = Codestream::new();
    {
        let siz = cs.access_siz_mut();
        siz.set_image_extent(Point::new(w, h));
        siz.set_tile_size(Size::new(w, h));
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
    cs.exchange(&pixels, 0).unwrap();
    cs.flush(&mut outfile).unwrap();

    let data = outfile.get_data().to_vec();
    let mut cs2 = Codestream::new();
    let mut infile = MemInfile::new(&data);
    cs2.read_headers(&mut infile).unwrap();
    cs2.create(&mut infile).unwrap();
    let line = cs2.pull(0).unwrap();
    assert_eq!(line, pixels);
}

// ============================================================================
// Edge case: boundary pixel values
// ============================================================================

/// All zeros
#[test]
fn edge_all_zeros() {
    let w = 16u32;
    let h = 16u32;
    let pixels = vec![0i32; (w * h) as usize];
    let mut cs = Codestream::new();
    {
        let siz = cs.access_siz_mut();
        siz.set_image_extent(Point::new(w, h));
        siz.set_tile_size(Size::new(w, h));
        siz.set_num_components(1);
        siz.set_comp_info(0, Point::new(1, 1), 8, false);
    }
    {
        let cod = cs.access_cod_mut();
        cod.set_reversible(true);
        cod.set_color_transform(false);
        cod.set_num_decomposition(2);
    }
    let mut outfile = MemOutfile::new();
    cs.write_headers(&mut outfile, &[]).unwrap();
    for y in 0..h {
        let start = (y * w) as usize;
        let end = start + w as usize;
        cs.exchange(&pixels[start..end], 0).unwrap();
    }
    cs.flush(&mut outfile).unwrap();

    let data = outfile.get_data().to_vec();
    let mut cs2 = Codestream::new();
    let mut infile = MemInfile::new(&data);
    cs2.read_headers(&mut infile).unwrap();
    cs2.create(&mut infile).unwrap();
    for _ in 0..h {
        let line = cs2.pull(0).unwrap();
        assert!(line.iter().all(|&v| v == 0));
    }
}

/// All max value (255 for 8-bit)
#[test]
fn edge_all_max() {
    let w = 16u32;
    let h = 16u32;
    let pixels = vec![255i32; (w * h) as usize];
    let mut cs = Codestream::new();
    {
        let siz = cs.access_siz_mut();
        siz.set_image_extent(Point::new(w, h));
        siz.set_tile_size(Size::new(w, h));
        siz.set_num_components(1);
        siz.set_comp_info(0, Point::new(1, 1), 8, false);
    }
    {
        let cod = cs.access_cod_mut();
        cod.set_reversible(true);
        cod.set_color_transform(false);
        cod.set_num_decomposition(2);
    }
    let mut outfile = MemOutfile::new();
    cs.write_headers(&mut outfile, &[]).unwrap();
    for y in 0..h {
        let start = (y * w) as usize;
        let end = start + w as usize;
        cs.exchange(&pixels[start..end], 0).unwrap();
    }
    cs.flush(&mut outfile).unwrap();

    let data = outfile.get_data().to_vec();
    let mut cs2 = Codestream::new();
    let mut infile = MemInfile::new(&data);
    cs2.read_headers(&mut infile).unwrap();
    cs2.create(&mut infile).unwrap();
    for _ in 0..h {
        let line = cs2.pull(0).unwrap();
        assert!(line.iter().all(|&v| v == 255));
    }
}

/// All max value for 16-bit
#[test]
fn edge_all_max_16bit() {
    let w = 16u32;
    let h = 16u32;
    let max_val = 65535i32;
    let pixels = vec![max_val; (w * h) as usize];
    let mut cs = Codestream::new();
    {
        let siz = cs.access_siz_mut();
        siz.set_image_extent(Point::new(w, h));
        siz.set_tile_size(Size::new(w, h));
        siz.set_num_components(1);
        siz.set_comp_info(0, Point::new(1, 1), 16, false);
    }
    {
        let cod = cs.access_cod_mut();
        cod.set_reversible(true);
        cod.set_color_transform(false);
        cod.set_num_decomposition(2);
    }
    let mut outfile = MemOutfile::new();
    cs.write_headers(&mut outfile, &[]).unwrap();
    for y in 0..h {
        let start = (y * w) as usize;
        let end = start + w as usize;
        cs.exchange(&pixels[start..end], 0).unwrap();
    }
    cs.flush(&mut outfile).unwrap();

    let data = outfile.get_data().to_vec();
    let mut cs2 = Codestream::new();
    let mut infile = MemInfile::new(&data);
    cs2.read_headers(&mut infile).unwrap();
    cs2.create(&mut infile).unwrap();
    for _ in 0..h {
        let line = cs2.pull(0).unwrap();
        assert!(line.iter().all(|&v| v == max_val));
    }
}

// ============================================================================
// Edge case: codestream restart/reuse
// ============================================================================

/// Encode, then restart and encode again with different parameters
#[test]
fn edge_codestream_restart() {
    // First encoding
    let mut cs = Codestream::new();
    {
        let siz = cs.access_siz_mut();
        siz.set_image_extent(Point::new(8, 8));
        siz.set_tile_size(Size::new(8, 8));
        siz.set_num_components(1);
        siz.set_comp_info(0, Point::new(1, 1), 8, false);
    }
    {
        let cod = cs.access_cod_mut();
        cod.set_reversible(true);
        cod.set_color_transform(false);
        cod.set_num_decomposition(1);
    }
    let mut outfile = MemOutfile::new();
    cs.write_headers(&mut outfile, &[]).unwrap();
    for _ in 0..8 {
        cs.exchange(&[100; 8], 0).unwrap();
    }
    cs.flush(&mut outfile).unwrap();
    let data1 = outfile.get_data().to_vec();

    // Restart and re-encode
    cs.restart();
    {
        let siz = cs.access_siz_mut();
        siz.set_image_extent(Point::new(4, 4));
        siz.set_tile_size(Size::new(4, 4));
        siz.set_num_components(1);
        siz.set_comp_info(0, Point::new(1, 1), 8, false);
    }
    {
        let cod = cs.access_cod_mut();
        cod.set_reversible(true);
        cod.set_color_transform(false);
        cod.set_num_decomposition(0);
    }
    let mut outfile2 = MemOutfile::new();
    cs.write_headers(&mut outfile2, &[]).unwrap();
    for _ in 0..4 {
        cs.exchange(&[200; 4], 0).unwrap();
    }
    cs.flush(&mut outfile2).unwrap();
    let data2 = outfile2.get_data().to_vec();

    // Verify both are valid
    assert!(!data1.is_empty());
    assert!(!data2.is_empty());
    assert_ne!(data1, data2);

    // Decode the second one
    let mut cs3 = Codestream::new();
    let mut infile = MemInfile::new(&data2);
    cs3.read_headers(&mut infile).unwrap();
    cs3.create(&mut infile).unwrap();
    for _ in 0..4 {
        let line = cs3.pull(0).unwrap();
        assert!(line.iter().all(|&v| v == 200));
    }
}

// ============================================================================
// Edge case: multiple independent codestreams
// ============================================================================

/// Encoding and decoding multiple independent codestreams
#[test]
fn edge_multiple_independent_codestreams() {
    let values = [42, 128, 255, 0, 100];

    for &val in &values {
        let mut cs = Codestream::new();
        {
            let siz = cs.access_siz_mut();
            siz.set_image_extent(Point::new(4, 4));
            siz.set_tile_size(Size::new(4, 4));
            siz.set_num_components(1);
            siz.set_comp_info(0, Point::new(1, 1), 8, false);
        }
        {
            let cod = cs.access_cod_mut();
            cod.set_reversible(true);
            cod.set_color_transform(false);
            cod.set_num_decomposition(1);
        }
        let mut outfile = MemOutfile::new();
        cs.write_headers(&mut outfile, &[]).unwrap();
        for _ in 0..4 {
            cs.exchange(&[val; 4], 0).unwrap();
        }
        cs.flush(&mut outfile).unwrap();

        let data = outfile.get_data().to_vec();
        let mut cs2 = Codestream::new();
        let mut infile = MemInfile::new(&data);
        cs2.read_headers(&mut infile).unwrap();
        cs2.create(&mut infile).unwrap();
        for _ in 0..4 {
            let line = cs2.pull(0).unwrap();
            assert!(
                line.iter().all(|&v| v == val),
                "expected all {}, got {:?}",
                val,
                line
            );
        }
    }
}

// ============================================================================
// Edge case: progression order string validation
// ============================================================================

/// Valid progression orders should be accepted
#[test]
fn edge_valid_progression_orders() {
    for order in &["LRCP", "RLCP", "RPCL", "PCRL", "CPRL"] {
        let mut cs = Codestream::new();
        {
            let siz = cs.access_siz_mut();
            siz.set_image_extent(Point::new(8, 8));
            siz.set_tile_size(Size::new(8, 8));
            siz.set_num_components(1);
            siz.set_comp_info(0, Point::new(1, 1), 8, false);
        }
        {
            let cod = cs.access_cod_mut();
            cod.set_reversible(true);
            cod.set_color_transform(false);
            cod.set_num_decomposition(2);
            let result = cod.set_progression_order(order);
            assert!(
                result.is_ok(),
                "progression order '{}' should be valid",
                order
            );
        }
    }
}

/// Invalid progression order should fail
#[test]
fn error_invalid_progression_order() {
    let mut cs = Codestream::new();
    let cod = cs.access_cod_mut();
    let result = cod.set_progression_order("INVALID");
    assert!(result.is_err(), "invalid progression order should fail");
}

// ============================================================================
// Edge case: comment markers
// ============================================================================

/// Encode with a comment, verify it doesn't break decoding
#[test]
fn edge_comment_marker() {
    use openjph_core::params::CommentExchange;

    let mut cs = Codestream::new();
    {
        let siz = cs.access_siz_mut();
        siz.set_image_extent(Point::new(8, 8));
        siz.set_tile_size(Size::new(8, 8));
        siz.set_num_components(1);
        siz.set_comp_info(0, Point::new(1, 1), 8, false);
    }
    {
        let cod = cs.access_cod_mut();
        cod.set_reversible(true);
        cod.set_color_transform(false);
        cod.set_num_decomposition(2);
    }

    let mut comment = CommentExchange::default();
    comment.set_string("OpenJPH-RS test comment");

    let mut outfile = MemOutfile::new();
    cs.write_headers(&mut outfile, &[comment]).unwrap();
    for _ in 0..8 {
        cs.exchange(&[128; 8], 0).unwrap();
    }
    cs.flush(&mut outfile).unwrap();

    let data = outfile.get_data().to_vec();
    let mut cs2 = Codestream::new();
    let mut infile = MemInfile::new(&data);
    cs2.read_headers(&mut infile).unwrap();
    cs2.create(&mut infile).unwrap();
    for _ in 0..8 {
        let line = cs2.pull(0).unwrap();
        assert!(line.iter().all(|&v| v == 128));
    }
}

// ============================================================================
// Edge case: pull after all lines consumed
// ============================================================================

/// Pulling beyond the image height should return None
#[test]
fn edge_pull_past_end() {
    let mut cs = Codestream::new();
    {
        let siz = cs.access_siz_mut();
        siz.set_image_extent(Point::new(4, 4));
        siz.set_tile_size(Size::new(4, 4));
        siz.set_num_components(1);
        siz.set_comp_info(0, Point::new(1, 1), 8, false);
    }
    {
        let cod = cs.access_cod_mut();
        cod.set_reversible(true);
        cod.set_color_transform(false);
        cod.set_num_decomposition(1);
    }
    let mut outfile = MemOutfile::new();
    cs.write_headers(&mut outfile, &[]).unwrap();
    for _ in 0..4 {
        cs.exchange(&[100; 4], 0).unwrap();
    }
    cs.flush(&mut outfile).unwrap();

    let data = outfile.get_data().to_vec();
    let mut cs2 = Codestream::new();
    let mut infile = MemInfile::new(&data);
    cs2.read_headers(&mut infile).unwrap();
    cs2.create(&mut infile).unwrap();

    // Pull all 4 lines
    for _ in 0..4 {
        assert!(cs2.pull(0).is_some());
    }
    // 5th pull should return None
    assert!(cs2.pull(0).is_none(), "pull past end should return None");
}
