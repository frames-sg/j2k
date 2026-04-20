//! **OpenJPH-RS** — Pure Rust HTJ2K (JPEG 2000 Part 15) codec.
//!
//! This crate is a faithful port of the [OpenJPH](https://github.com/aous72/OpenJPH)
//! C++ library (v0.26.3) providing encoding and decoding of HTJ2K codestreams as
//! defined in ISO/IEC 15444-15.
//!
//! # Overview
//!
//! The main entry point is [`codestream::Codestream`], which provides both the
//! encoder (write) and decoder (read) pipeline. Image parameters are configured
//! through marker segment types in the [`params`] module.
//!
//! # Quick Start — Encoding
//!
//! ```rust
//! use openjph_core::codestream::Codestream;
//! use openjph_core::file::MemOutfile;
//! use openjph_core::types::{Point, Size};
//!
//! let (width, height) = (8u32, 8u32);
//! let pixels: Vec<i32> = vec![128; (width * height) as usize];
//!
//! let mut cs = Codestream::new();
//! cs.access_siz_mut().set_image_extent(Point::new(width, height));
//! cs.access_siz_mut().set_num_components(1);
//! cs.access_siz_mut().set_comp_info(0, Point::new(1, 1), 8, false);
//! cs.access_siz_mut().set_tile_size(Size::new(width, height));
//! cs.access_cod_mut().set_num_decomposition(0);
//! cs.access_cod_mut().set_reversible(true);
//! cs.access_cod_mut().set_color_transform(false);
//! cs.set_planar(0);
//!
//! let mut outfile = MemOutfile::new();
//! cs.write_headers(&mut outfile, &[]).unwrap();
//! for y in 0..height as usize {
//!     let start = y * width as usize;
//!     cs.exchange(&pixels[start..start + width as usize], 0).unwrap();
//! }
//! cs.flush(&mut outfile).unwrap();
//!
//! let encoded = outfile.get_data();
//! assert!(encoded.len() > 20);
//! ```
//!
//! # Quick Start — Decoding
//!
//! ```rust
//! # use openjph_core::codestream::Codestream;
//! # use openjph_core::file::{MemOutfile, MemInfile};
//! # use openjph_core::types::{Point, Size};
//! # let (width, height) = (8u32, 8u32);
//! # let pixels: Vec<i32> = vec![128; (width * height) as usize];
//! # let mut cs = Codestream::new();
//! # cs.access_siz_mut().set_image_extent(Point::new(width, height));
//! # cs.access_siz_mut().set_num_components(1);
//! # cs.access_siz_mut().set_comp_info(0, Point::new(1, 1), 8, false);
//! # cs.access_siz_mut().set_tile_size(Size::new(width, height));
//! # cs.access_cod_mut().set_num_decomposition(0);
//! # cs.access_cod_mut().set_reversible(true);
//! # cs.access_cod_mut().set_color_transform(false);
//! # cs.set_planar(0);
//! # let mut outfile = MemOutfile::new();
//! # cs.write_headers(&mut outfile, &[]).unwrap();
//! # for y in 0..height as usize {
//! #     let start = y * width as usize;
//! #     cs.exchange(&pixels[start..start + width as usize], 0).unwrap();
//! # }
//! # cs.flush(&mut outfile).unwrap();
//! # let encoded = outfile.get_data().to_vec();
//! let mut infile = MemInfile::new(&encoded);
//! let mut decoder = Codestream::new();
//! decoder.read_headers(&mut infile).unwrap();
//! decoder.create(&mut infile).unwrap();
//!
//! for _y in 0..height {
//!     let line = decoder.pull(0).expect("expected decoded line");
//!     assert_eq!(line.len(), width as usize);
//! }
//! ```
//!
//! # Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`types`] | Numeric aliases, geometric primitives (`Size`, `Point`, `Rect`) |
//! | [`error`] | Error types ([`OjphError`]) and [`Result`] alias |
//! | [`message`] | Diagnostic message dispatch (info/warn/error) |
//! | [`arch`] | CPU feature detection and alignment constants |
//! | [`mem`] | Aligned allocators and line buffers |
//! | [`file`](mod@file) | I/O traits and file/memory stream implementations |
//! | [`params`] | JPEG 2000 marker segment types (SIZ, COD, QCD, NLT, …) |
//! | [`codestream`] | Main codec interface ([`Codestream`](codestream::Codestream)) |
//! | [`arg`] | Minimal CLI argument interpreter |
//! | [`coding`] | HTJ2K block entropy coding (internal) |
//! | [`transform`] | Wavelet and color transforms (internal) |

pub mod arch;
pub mod arg;
pub mod codestream;
pub mod coding;
pub mod error;
pub mod file;
pub mod mem;
pub mod message;
pub mod params;
pub mod transform;
pub mod types;

pub use error::{OjphError, Result};
pub use types::*;

#[cfg(test)]
mod pipeline_tests {
    use crate::codestream::Codestream;
    use crate::file::{MemInfile, MemOutfile};
    use crate::types::*;

    /// Encode an 8×8 image through the full pipeline, decode it, and verify
    /// the round-trip produces values within CUP tolerance.
    #[test]
    fn roundtrip_8x8_single_component() {
        let width = 8u32;
        let height = 8u32;

        // Use pixel value 131 (level-shifted: 3, magnitude=3, odd → exact with p=1)
        let image: Vec<Vec<i32>> = vec![vec![131i32; width as usize]; height as usize];

        // --- Encode ---
        let mut cs = Codestream::new();
        cs.access_siz_mut()
            .set_image_extent(Point::new(width, height));
        cs.access_siz_mut().set_num_components(1);
        cs.access_siz_mut()
            .set_comp_info(0, Point::new(1, 1), 8, false);
        cs.access_siz_mut().set_tile_size(Size::new(width, height));
        cs.access_cod_mut().set_num_decomposition(1);
        cs.access_cod_mut().set_reversible(true);
        cs.access_cod_mut().set_color_transform(false);
        cs.set_planar(0);

        let mut outfile = MemOutfile::new();
        cs.write_headers(&mut outfile, &[]).unwrap();

        for row in &image {
            cs.exchange(row, 0).unwrap();
        }
        cs.flush(&mut outfile).unwrap();

        let encoded = outfile.get_data().to_vec();
        assert!(
            encoded.len() > 20,
            "encoded data too small: {} bytes",
            encoded.len()
        );

        // --- Decode ---
        let mut infile = MemInfile::new(&encoded);
        let mut cs2 = Codestream::new();
        cs2.read_headers(&mut infile).unwrap();
        cs2.create(&mut infile).unwrap();

        for (y, expected_row) in image.iter().enumerate() {
            let line = cs2.pull(0).expect("expected decoded line");
            assert_eq!(
                line,
                *expected_row,
                "mismatch at row {y}: got {:?}, expected {:?}",
                &line[..],
                &expected_row[..]
            );
        }

        // No more lines
        assert!(cs2.pull(0).is_none());
    }

    /// Encode + decode a flat (constant value) image with exact roundtrip.
    #[test]
    fn roundtrip_8x8_constant() {
        let width = 8u32;
        let height = 8u32;
        // Pixel value 125: level-shifted = -3, magnitude 3 (odd), roundtrips exactly
        let image: Vec<Vec<i32>> = vec![vec![125i32; width as usize]; height as usize];

        let mut cs = Codestream::new();
        cs.access_siz_mut()
            .set_image_extent(Point::new(width, height));
        cs.access_siz_mut().set_num_components(1);
        cs.access_siz_mut()
            .set_comp_info(0, Point::new(1, 1), 8, false);
        cs.access_siz_mut().set_tile_size(Size::new(width, height));
        cs.access_cod_mut().set_num_decomposition(1);
        cs.access_cod_mut().set_reversible(true);
        cs.access_cod_mut().set_color_transform(false);
        cs.set_planar(0);

        let mut outfile = MemOutfile::new();
        cs.write_headers(&mut outfile, &[]).unwrap();
        for row in &image {
            cs.exchange(row, 0).unwrap();
        }
        cs.flush(&mut outfile).unwrap();

        let encoded = outfile.get_data().to_vec();
        let mut infile = MemInfile::new(&encoded);
        let mut cs2 = Codestream::new();
        cs2.read_headers(&mut infile).unwrap();
        cs2.create(&mut infile).unwrap();

        for (y, expected_row) in image.iter().enumerate() {
            let line = cs2.pull(0).expect("expected decoded line");
            assert_eq!(line, *expected_row, "mismatch at row {y}");
        }
    }

    /// Zero-decomposition (no DWT): test end-to-end with CUP-compatible values.
    #[test]
    fn roundtrip_8x8_no_dwt() {
        let width = 8u32;
        let height = 8u32;
        // Use constant value for no-DWT test (pixel=131, level-shift=3, magnitude=3, p=1)
        let image: Vec<Vec<i32>> = vec![vec![131i32; width as usize]; height as usize];

        let mut cs = Codestream::new();
        cs.access_siz_mut()
            .set_image_extent(Point::new(width, height));
        cs.access_siz_mut().set_num_components(1);
        cs.access_siz_mut()
            .set_comp_info(0, Point::new(1, 1), 8, false);
        cs.access_siz_mut().set_tile_size(Size::new(width, height));
        cs.access_cod_mut().set_num_decomposition(0);
        cs.access_cod_mut().set_reversible(true);
        cs.access_cod_mut().set_color_transform(false);
        cs.set_planar(0);

        let mut outfile = MemOutfile::new();
        cs.write_headers(&mut outfile, &[]).unwrap();
        for row in &image {
            cs.exchange(row, 0).unwrap();
        }
        cs.flush(&mut outfile).unwrap();

        let encoded = outfile.get_data().to_vec();
        let mut infile = MemInfile::new(&encoded);
        let mut cs2 = Codestream::new();
        cs2.read_headers(&mut infile).unwrap();
        cs2.create(&mut infile).unwrap();

        for (y, expected_row) in image.iter().enumerate() {
            let line = cs2.pull(0).expect("expected decoded line");
            assert_eq!(line, *expected_row, "mismatch at row {y}");
        }
    }

    /// Block coder roundtrip with corrected missing_msbs computation.
    #[test]
    fn block_coder_roundtrip_magnitude_3() {
        use crate::coding::decoder32::decode_codeblock32;
        use crate::coding::encoder::encode_codeblock32;

        // Magnitude 3 with p=1 should roundtrip exactly
        let mag = 3u32;
        let u32_val = (1u32 << 31) | mag;
        let buf = vec![u32_val; 16];
        let num_bits = 32 - mag.leading_zeros();
        let missing_msbs = 31 - num_bits; // p=1, msbs=29

        let enc = encode_codeblock32(&buf, missing_msbs, 1, 4, 4, 4).unwrap();
        assert!(enc.length >= 2);

        let mut coded = enc.data.clone();
        coded.resize(coded.len() + 16, 0);
        let mut decoded = vec![0u32; 16];
        decode_codeblock32(
            &mut coded,
            &mut decoded,
            missing_msbs,
            1,
            enc.length,
            0,
            4,
            4,
            4,
            false,
        )
        .unwrap();

        for i in 0..16 {
            assert_eq!(
                decoded[i], buf[i],
                "mismatch at {}: got 0x{:08X}, expected 0x{:08X}",
                i, decoded[i], buf[i]
            );
        }
    }
}

#[cfg(test)]
mod debug_roundtrip_test {
    use crate::codestream::Codestream;
    use crate::coding::decoder32::decode_codeblock32;
    use crate::coding::encoder::encode_codeblock32;
    use crate::file::{MemInfile, MemOutfile};
    use crate::types::{Point, Size};

    #[test]
    fn debug_no_dwt_value100() {
        let width = 8u32;
        let height = 8u32;
        let pixels = vec![100i32; (width * height) as usize];

        let mut cs = Codestream::new();
        cs.access_siz_mut()
            .set_image_extent(Point::new(width, height));
        cs.access_siz_mut().set_num_components(1);
        cs.access_siz_mut()
            .set_comp_info(0, Point::new(1, 1), 8, false);
        cs.access_siz_mut().set_tile_size(Size::new(width, height));
        cs.access_cod_mut().set_num_decomposition(0);
        cs.access_cod_mut().set_reversible(true);
        cs.access_cod_mut().set_color_transform(false);
        cs.set_planar(0);

        let mut outfile = MemOutfile::new();
        cs.write_headers(&mut outfile, &[]).unwrap();
        for y in 0..height as usize {
            let start = y * width as usize;
            let end = start + width as usize;
            cs.exchange(&pixels[start..end], 0).unwrap();
        }
        cs.flush(&mut outfile).unwrap();

        let encoded = outfile.get_data().to_vec();
        let mut infile = MemInfile::new(&encoded);
        let mut cs2 = Codestream::new();
        cs2.read_headers(&mut infile).unwrap();
        cs2.create(&mut infile).unwrap();

        for y in 0..height as usize {
            let line = cs2.pull(0).expect("expected decoded line");
            for x in 0..width as usize {
                if line[x] != pixels[y * width as usize + x] {
                    panic!(
                        "Mismatch at ({},{}): expected {}, got {}",
                        x,
                        y,
                        pixels[y * width as usize + x],
                        line[x]
                    );
                }
            }
        }
    }

    #[test]
    fn debug_no_dwt_33x33_packet_preserves_block_bytes() {
        let width = 33u32;
        let height = 33u32;
        let mut image = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                image.push(((3 * x + 7 * y) & 0xFF) as i32);
            }
        }

        let kmax = 8u32;
        let shift = 31 - kmax;
        let missing_msbs = kmax - 1;
        let stride = 64u32;
        let nominal_h = 64u32;
        let mut direct_samples = vec![0u32; (stride * nominal_h) as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let pixel = image[y * width as usize + x];
                let centered = pixel - 128;
                let sign = if centered < 0 { 0x8000_0000 } else { 0 };
                let mag = centered.unsigned_abs() << shift;
                direct_samples[y * stride as usize + x] = sign | mag;
            }
        }
        let direct = encode_codeblock32(&direct_samples, missing_msbs, 1, width, height, stride)
            .expect("direct block encode failed");
        let direct_bytes = direct.data[..direct.length as usize].to_vec();

        let mut cs = Codestream::new();
        cs.access_siz_mut()
            .set_image_extent(Point::new(width, height));
        cs.access_siz_mut().set_num_components(1);
        cs.access_siz_mut()
            .set_comp_info(0, Point::new(1, 1), 8, false);
        cs.access_siz_mut().set_tile_size(Size::new(width, height));
        cs.access_cod_mut().set_num_decomposition(0);
        cs.access_cod_mut().set_reversible(true);
        cs.access_cod_mut().set_color_transform(false);
        cs.set_planar(0);

        let mut outfile = MemOutfile::new();
        cs.write_headers(&mut outfile, &[]).unwrap();
        for y in 0..height as usize {
            let start = y * width as usize;
            let end = start + width as usize;
            cs.exchange(&image[start..end], 0).unwrap();
        }
        cs.flush(&mut outfile).unwrap();

        let enc_cb =
            &cs.debug_inner().tiles[0].tile_comps[0].resolutions[0].subbands[0].codeblocks[0];
        let enc_state = enc_cb.enc_state.as_ref().expect("missing encode state");
        assert_eq!(enc_state.pass1_bytes as usize, direct_bytes.len());
        assert_eq!(enc_cb.coded_data, direct_bytes);

        let encoded = outfile.get_data().to_vec();
        let mut infile = MemInfile::new(&encoded);
        let mut dec = Codestream::new();
        dec.read_headers(&mut infile).unwrap();
        dec.create(&mut infile).unwrap();

        let dec_cb =
            &dec.debug_inner().tiles[0].tile_comps[0].resolutions[0].subbands[0].codeblocks[0];
        let dec_state = dec_cb.dec_state.as_ref().expect("missing decode state");
        assert_eq!(dec_state.pass1_len as usize, direct_bytes.len());
        assert_eq!(dec_cb.coded_data, direct_bytes);

        let mut decoded = Vec::with_capacity((width * height) as usize);
        for _ in 0..height {
            decoded.extend(dec.pull(0).expect("missing decoded line"));
        }
        assert_eq!(decoded, image);
    }

    #[test]
    fn debug_d2_subband_coeffs_survive_roundtrip() {
        let width = 64u32;
        let height = 64u32;
        let mut image = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                image.push(((3 * x + 7 * y) & 0xFF) as i32);
            }
        }

        let mut cs = Codestream::new();
        cs.access_siz_mut()
            .set_image_extent(Point::new(width, height));
        cs.access_siz_mut().set_num_components(1);
        cs.access_siz_mut()
            .set_comp_info(0, Point::new(1, 1), 8, false);
        cs.access_siz_mut().set_tile_size(Size::new(width, height));
        cs.access_cod_mut().set_num_decomposition(2);
        cs.access_cod_mut().set_reversible(true);
        cs.access_cod_mut().set_color_transform(false);
        cs.set_planar(0);

        let mut outfile = MemOutfile::new();
        cs.write_headers(&mut outfile, &[]).unwrap();
        for y in 0..height as usize {
            let start = y * width as usize;
            let end = start + width as usize;
            cs.exchange(&image[start..end], 0).unwrap();
        }
        cs.flush(&mut outfile).unwrap();

        let enc_tc = &cs.debug_inner().tiles[0].tile_comps[0];
        let enc_coeffs: Vec<Vec<Vec<i32>>> = enc_tc
            .resolutions
            .iter()
            .map(|res| res.subbands.iter().map(|sb| sb.coeffs.clone()).collect())
            .collect();

        let encoded = outfile.get_data().to_vec();
        let mut infile = MemInfile::new(&encoded);
        let mut dec = Codestream::new();
        dec.read_headers(&mut infile).unwrap();
        dec.create(&mut infile).unwrap();

        let dec_tc = &dec.debug_inner().tiles[0].tile_comps[0];
        for (res_idx, (enc_res, dec_res)) in enc_coeffs.iter().zip(&dec_tc.resolutions).enumerate()
        {
            for (sb_idx, (enc_sb, dec_sb)) in enc_res.iter().zip(&dec_res.subbands).enumerate() {
                assert_eq!(
                    enc_sb, &dec_sb.coeffs,
                    "subband mismatch at resolution {res_idx}, subband {sb_idx}"
                );
            }
        }
    }

    #[test]
    fn debug_tiled_d5_encoder_codeblocks_are_decodable() {
        let width = 256u32;
        let height = 256u32;
        let mut components = (0..3)
            .map(|_| Vec::<i32>::with_capacity((width * height) as usize))
            .collect::<Vec<_>>();
        for y in 0..height {
            for x in 0..width {
                components[0].push(((x * 255) / width.max(1)) as i32);
                components[1].push(((y * 255) / height.max(1)) as i32);
                components[2].push((((x + y) * 127) / (width + height).max(1)) as i32);
            }
        }

        let mut cs = Codestream::new();
        cs.access_siz_mut()
            .set_image_extent(Point::new(width, height));
        cs.access_siz_mut().set_tile_size(Size::new(33, 33));
        cs.access_siz_mut().set_num_components(3);
        for c in 0..3 {
            cs.access_siz_mut()
                .set_comp_info(c, Point::new(1, 1), 8, false);
        }
        cs.access_cod_mut().set_num_decomposition(5);
        cs.access_cod_mut().set_reversible(true);
        cs.access_cod_mut().set_color_transform(true);

        let mut outfile = MemOutfile::new();
        cs.write_headers(&mut outfile, &[]).unwrap();
        for y in 0..height as usize {
            let start = y * width as usize;
            let end = start + width as usize;
            for (c, comp) in components.iter().enumerate() {
                cs.exchange(&comp[start..end], c as u32).unwrap();
            }
        }
        cs.flush(&mut outfile).unwrap();

        for (tile_idx, tile) in cs.debug_inner().tiles.iter().enumerate() {
            for (comp_idx, tc) in tile.tile_comps.iter().enumerate() {
                for (res_idx, res) in tc.resolutions.iter().enumerate() {
                    for (sb_idx, sb) in res.subbands.iter().enumerate() {
                        for (cb_idx, cb) in sb.codeblocks.iter().enumerate() {
                            let Some(enc) = cb.enc_state.as_ref() else {
                                continue;
                            };
                            if !enc.has_data || enc.num_passes == 0 {
                                continue;
                            }

                            let mut coded = cb.coded_data.clone();
                            let nominal_w = 1u32 << cb.log_block_dims.w;
                            let nominal_h = 1u32 << cb.log_block_dims.h;
                            let stride = (nominal_w + 7) & !7;
                            let mut decoded = vec![0u32; (stride * nominal_h) as usize];
                            let result =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    decode_codeblock32(
                                        &mut coded,
                                        &mut decoded,
                                        enc.missing_msbs,
                                        enc.num_passes,
                                        enc.pass1_bytes,
                                        enc.pass2_bytes,
                                        cb.width(),
                                        cb.height(),
                                        stride,
                                        false,
                                    )
                                }));

                            match result {
                                Ok(Ok(true)) | Ok(Ok(false)) => {}
                                Ok(Err(err)) => panic!(
                                    "decode error for tile={tile_idx} comp={comp_idx} res={res_idx} sb={sb_idx} cb={cb_idx} rect={:?} passes={} p1={} p2={} mmsbs={}: {err:?}",
                                    cb.cb_rect,
                                    enc.num_passes,
                                    enc.pass1_bytes,
                                    enc.pass2_bytes,
                                    enc.missing_msbs,
                                ),
                                Err(_) => panic!(
                                    "decode panic for tile={tile_idx} comp={comp_idx} res={res_idx} sb={sb_idx} cb={cb_idx} rect={:?} passes={} p1={} p2={} mmsbs={} bytes={:02X?}",
                                    cb.cb_rect,
                                    enc.num_passes,
                                    enc.pass1_bytes,
                                    enc.pass2_bytes,
                                    enc.missing_msbs,
                                    cb.coded_data,
                                ),
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn debug_rev53_4x1024_encoder_codeblocks_are_decodable() {
        let width = 256u32;
        let height = 256u32;
        let mut components = (0..3)
            .map(|_| Vec::<i32>::with_capacity((width * height) as usize))
            .collect::<Vec<_>>();
        for y in 0..height {
            for x in 0..width {
                components[0].push(((x * 255) / width.max(1)) as i32);
                components[1].push(((y * 255) / height.max(1)) as i32);
                components[2].push((((x + y) * 127) / (width + height).max(1)) as i32);
            }
        }

        let mut cs = Codestream::new();
        cs.access_siz_mut()
            .set_image_extent(Point::new(width, height));
        cs.access_siz_mut().set_tile_size(Size::new(width, height));
        cs.access_siz_mut().set_num_components(3);
        for c in 0..3 {
            cs.access_siz_mut()
                .set_comp_info(c, Point::new(1, 1), 8, false);
        }
        cs.access_cod_mut().set_num_decomposition(5);
        cs.access_cod_mut().set_reversible(true);
        cs.access_cod_mut().set_color_transform(true);
        cs.access_cod_mut().set_block_dims(4, 1024);

        let mut outfile = MemOutfile::new();
        cs.write_headers(&mut outfile, &[]).unwrap();
        for y in 0..height as usize {
            let start = y * width as usize;
            let end = start + width as usize;
            for (c, comp) in components.iter().enumerate() {
                cs.exchange(&comp[start..end], c as u32).unwrap();
            }
        }
        cs.flush(&mut outfile).unwrap();

        for (tile_idx, tile) in cs.debug_inner().tiles.iter().enumerate() {
            for (comp_idx, tc) in tile.tile_comps.iter().enumerate() {
                for (res_idx, res) in tc.resolutions.iter().enumerate() {
                    for (sb_idx, sb) in res.subbands.iter().enumerate() {
                        for (cb_idx, cb) in sb.codeblocks.iter().enumerate() {
                            let Some(enc) = cb.enc_state.as_ref() else {
                                continue;
                            };
                            if !enc.has_data || enc.num_passes == 0 {
                                continue;
                            }

                            let nominal_w = 1u32 << cb.log_block_dims.w;
                            let nominal_h = 1u32 << cb.log_block_dims.h;
                            let stride = (nominal_w + 7) & !7;
                            let mut coded = cb.coded_data.clone();
                            let mut decoded = vec![0u32; (stride * nominal_h) as usize];
                            let result =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    decode_codeblock32(
                                        &mut coded,
                                        &mut decoded,
                                        enc.missing_msbs,
                                        enc.num_passes,
                                        enc.pass1_bytes,
                                        enc.pass2_bytes,
                                        cb.width(),
                                        cb.height(),
                                        stride,
                                        false,
                                    )
                                }));

                            match result {
                                Ok(Ok(true)) | Ok(Ok(false)) => {}
                                Ok(Err(err)) => panic!(
                                    "decode error for tile={tile_idx} comp={comp_idx} res={res_idx} sb={sb_idx} cb={cb_idx} rect={:?} log_dims={:?} passes={} p1={} p2={} mmsbs={} bytes={:02X?}: {err:?}",
                                    cb.cb_rect,
                                    cb.log_block_dims,
                                    enc.num_passes,
                                    enc.pass1_bytes,
                                    enc.pass2_bytes,
                                    enc.missing_msbs,
                                    cb.coded_data,
                                ),
                                Err(_) => panic!(
                                    "decode panic for tile={tile_idx} comp={comp_idx} res={res_idx} sb={sb_idx} cb={cb_idx} rect={:?} log_dims={:?} passes={} p1={} p2={} mmsbs={} bytes={:02X?}",
                                    cb.cb_rect,
                                    cb.log_block_dims,
                                    enc.num_passes,
                                    enc.pass1_bytes,
                                    enc.pass2_bytes,
                                    enc.missing_msbs,
                                    cb.coded_data,
                                ),
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn debug_rev53_4x1024_packet_preserves_codeblocks() {
        let width = 256u32;
        let height = 256u32;
        let mut components = (0..3)
            .map(|_| Vec::<i32>::with_capacity((width * height) as usize))
            .collect::<Vec<_>>();
        for y in 0..height {
            for x in 0..width {
                components[0].push(((x * 255) / width.max(1)) as i32);
                components[1].push(((y * 255) / height.max(1)) as i32);
                components[2].push((((x + y) * 127) / (width + height).max(1)) as i32);
            }
        }

        let mut cs = Codestream::new();
        cs.access_siz_mut()
            .set_image_extent(Point::new(width, height));
        cs.access_siz_mut().set_tile_size(Size::new(width, height));
        cs.access_siz_mut().set_num_components(3);
        for c in 0..3 {
            cs.access_siz_mut()
                .set_comp_info(c, Point::new(1, 1), 8, false);
        }
        cs.access_cod_mut().set_num_decomposition(5);
        cs.access_cod_mut().set_reversible(true);
        cs.access_cod_mut().set_color_transform(true);
        cs.access_cod_mut().set_block_dims(4, 1024);

        let mut outfile = MemOutfile::new();
        cs.write_headers(&mut outfile, &[]).unwrap();
        for y in 0..height as usize {
            let start = y * width as usize;
            let end = start + width as usize;
            for (c, comp) in components.iter().enumerate() {
                cs.exchange(&comp[start..end], c as u32).unwrap();
            }
        }
        cs.flush(&mut outfile).unwrap();

        let encoded = outfile.get_data().to_vec();
        let mut infile = MemInfile::new(&encoded);
        let mut dec = Codestream::new();
        dec.read_headers(&mut infile).unwrap();
        dec.create(&mut infile).unwrap();

        for (tile_idx, (enc_tile, dec_tile)) in cs
            .debug_inner()
            .tiles
            .iter()
            .zip(&dec.debug_inner().tiles)
            .enumerate()
        {
            for (comp_idx, (enc_tc, dec_tc)) in enc_tile
                .tile_comps
                .iter()
                .zip(&dec_tile.tile_comps)
                .enumerate()
            {
                for (res_idx, (enc_res, dec_res)) in enc_tc
                    .resolutions
                    .iter()
                    .zip(&dec_tc.resolutions)
                    .enumerate()
                {
                    for (sb_idx, (enc_sb, dec_sb)) in
                        enc_res.subbands.iter().zip(&dec_res.subbands).enumerate()
                    {
                        for (cb_idx, (enc_cb, dec_cb)) in
                            enc_sb.codeblocks.iter().zip(&dec_sb.codeblocks).enumerate()
                        {
                            let enc_state = enc_cb.enc_state.as_ref();
                            let dec_state = dec_cb.dec_state.as_ref();
                            match (enc_state, dec_state) {
                                (None, None) => continue,
                                (Some(enc), None) if !enc.has_data || enc.num_passes == 0 => continue,
                                (Some(enc), Some(dec_state)) => {
                                    if enc.has_data != (dec_state.num_passes > 0) {
                                        panic!(
                                            "has_data mismatch tile={tile_idx} comp={comp_idx} res={res_idx} sb={sb_idx} cb={cb_idx} rect={:?} enc={:?} dec={:?}",
                                            enc_cb.cb_rect, enc, dec_state
                                        );
                                    }
                                    if !enc.has_data {
                                        continue;
                                    }
                                    if enc.pass1_bytes != dec_state.pass1_len
                                        || enc.pass2_bytes != dec_state.pass2_len
                                        || enc.num_passes != dec_state.num_passes
                                        || enc.missing_msbs != dec_state.missing_msbs
                                        || enc_cb.coded_data != dec_cb.coded_data
                                    {
                                        panic!(
                                            "packet mismatch tile={tile_idx} comp={comp_idx} res={res_idx} sb={sb_idx} cb={cb_idx} rect={:?} enc={:?} dec={:?} enc_bytes={:02X?} dec_bytes={:02X?}",
                                            enc_cb.cb_rect,
                                            enc,
                                            dec_state,
                                            enc_cb.coded_data,
                                            dec_cb.coded_data,
                                        );
                                    }
                                }
                                _ => panic!(
                                    "state presence mismatch tile={tile_idx} comp={comp_idx} res={res_idx} sb={sb_idx} cb={cb_idx} rect={:?} enc={:?} dec={:?}",
                                    enc_cb.cb_rect,
                                    enc_state,
                                    dec_state,
                                ),
                            }
                        }
                    }
                }
            }
        }
    }
}
