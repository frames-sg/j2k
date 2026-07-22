#![cfg(target_os = "macos")]

use std::sync::Arc;

use j2k::{
    BatchCodecRoute, BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples,
    DecodeRequest, EncodedImage, J2kContext, NativeSampleType, PreparationDepth,
};
use j2k_core::{
    BackendKind, BackendRequest, CodecError, DeviceSubmission, DeviceSurface, Downscale,
    ImageDecode, ImageDecodeDevice, PixelFormat, Rect, TileBatchDecodeDevice,
    TileBatchDecodeManyDevice, TileBatchDecodeSubmit, TileRegionScaledDeviceDecodeRequest,
};
use j2k_metal::{
    Codec, DecodeOperation, Error, J2kDecoder, J2kScratchPool, MetalBackendSession,
    MetalBatchDecoder, MetalDecodeRequest, MetalImageDestination, MetalImageLayout, MetalSession,
    MetalTileBatch, Surface, SurfaceResidency,
};
use j2k_native::{encode, encode_htj2k, EncodeOptions};

const UNSUPPORTED_RGBA16_REASON: &str = "J2K Metal does not support PixelFormat::Rgba16";
const AUTO_DECODE_CPU_FALLBACK_REASON: &str =
    "J2K Metal Auto decode stays on CPU until decode benchmark evidence justifies Metal routing";

macro_rules! submit_tile_region_scaled_to_device {
    ($ctx:expr, $session:expr, $pool:expr, $input:expr, $fmt:expr, $roi:expr, $scale:expr, $backend:expr $(,)?) => {
        Codec::submit_tile_region_scaled_to_device(
            $ctx,
            $session,
            $pool,
            TileRegionScaledDeviceDecodeRequest {
                input: $input,
                fmt: $fmt,
                roi: $roi,
                scale: $scale,
                backend: $backend,
            },
        )
    };
}

fn should_run_metal_runtime() -> bool {
    j2k_test_support::metal_runtime_gate(module_path!())
}

fn unsupported_classic_roi_rgb() -> Arc<[u8]> {
    let pixels = (0..4_u8)
        .flat_map(|index| [index * 17, index * 29 + 3, index * 41 + 5])
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        roi_component_shifts: vec![3, 0, 0],
        ..EncodeOptions::default()
    };
    Arc::from(
        encode(&pixels, 2, 2, 3, 8, false, &options)
            .expect("encode classic RGB8 with unsupported RGN maxshift"),
    )
}

fn completed_surface_metal_buffer(surface: &Surface) -> Option<(&metal::Buffer, usize)> {
    // SAFETY: Every surface passed by these tests has completed its decode, and
    // the tests never submit a writer or mutate a returned handle.
    unsafe { surface.metal_buffer() }
}

fn completed_resident_batch_bytes(group: &j2k_metal::MetalBatchGroup) -> Vec<u8> {
    let resident = group
        .resident_batch()
        .expect("completed group has resident Metal storage");
    // SAFETY: the group is returned only after codec completion and this test
    // performs a readback without submitting a writer or retaining the handle.
    unsafe {
        j2k_metal_support::checked_buffer_read_vec::<u8>(
            resident.metal_buffer(),
            resident.byte_offset(),
            resident.byte_len(),
        )
        .expect("read completed dense resident batch")
    }
}

fn assert_unsupported_rgba16_report(result: Result<j2k_metal::DecodeSurfaceWithReport, Error>) {
    match result {
        Err(Error::UnsupportedMetalRequest { reason }) => {
            assert_eq!(reason, UNSUPPORTED_RGBA16_REASON);
        }
        Err(other) => panic!("unexpected explicit Metal error: {other:?}"),
        Ok(surface) => panic!(
            "explicit Metal must not silently fall back; got {:?}",
            surface.report.selected_backend
        ),
    }
}

fn fixture_rgb8() -> Vec<u8> {
    let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 2, 2, 3, 8, false, &options).expect("encode rgb8")
}

fn fixture_gray8() -> Vec<u8> {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode gray8")
}

fn fixture_gray8_sized(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x + y) & 0xFF) as u8);
        }
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 3,
        guard_bits: 2,
        ..EncodeOptions::default()
    };
    encode(&pixels, width, height, 1, 8, false, &options).expect("encode sized gray8")
}

fn fixture_ht_gray8_sized(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 3 + y * 5) & 0xFF) as u8);
        }
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 3,
        guard_bits: 2,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, width, height, 1, 8, false, &options).expect("encode sized ht gray8")
}

fn fixture_rgb8_sized(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 3 + y * 5) & 0xFF) as u8);
            pixels.push(((x * 7 + y * 11 + 13) & 0xFF) as u8);
            pixels.push(((x * 17 + y * 19 + 29) & 0xFF) as u8);
        }
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 3,
        guard_bits: 2,
        ..EncodeOptions::default()
    };
    encode(&pixels, width, height, 3, 8, false, &options).expect("encode sized rgb8")
}

fn fixture_classic_multitile_rgb8() -> (Vec<u8>, Vec<u8>) {
    let width = 19_u32;
    let height = 13_u32;
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 17 + y * 13 + 3) & 0xFF) as u8);
            pixels.push(((x * 7 + y * 29 + 41) & 0xFF) as u8);
            pixels.push(((x * 31 + y * 5 + 97) & 0xFF) as u8);
        }
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        tile_size: Some((11, 7)),
        ..EncodeOptions::default()
    };
    let encoded = encode(&pixels, width, height, 3, 8, false, &options)
        .expect("encode odd-edge multi-tile classic RGB8");
    (encoded, pixels)
}

fn fixture_ht_gray8_unsupported_direct_width() -> Vec<u8> {
    let width = 512u32;
    let height = 8u32;
    let mut pixels = Vec::with_capacity(width as usize * height as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 7 + y * 11 + x / 3) & 0xFF) as u8);
        }
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 0,
        code_block_width_exp: 7,
        code_block_height_exp: 1,
        guard_bits: 2,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, width, height, 1, 8, false, &options).expect("encode wide ht gray8")
}

fn fixture_gray8_reversed() -> Vec<u8> {
    let pixels: Vec<u8> = (0..16).rev().collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode reversed gray8")
}

fn fixture_gray12() -> Vec<u8> {
    let mut pixels = Vec::with_capacity(8);
    for sample in [0u16, 257, 1023, 4095] {
        pixels.extend_from_slice(&sample.to_le_bytes());
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 2, 2, 1, 12, false, &options).expect("encode gray12")
}

fn fixture_ht_gray12_offset(offset: u16) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(4 * 4 * 2);
    for y in 0..4_u16 {
        for x in 0..4_u16 {
            let sample = (offset + x * 193 + y * 257) & 0x0FFF;
            pixels.extend_from_slice(&sample.to_le_bytes());
        }
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, 4, 4, 1, 12, false, &options).expect("encode ht gray12")
}

fn fixture_ht_signed_gray12(offset: i16) -> (Vec<u8>, Vec<i16>) {
    let samples = (0..16_i16)
        .map(|index| -1_700 + offset + index * 197)
        .collect::<Vec<_>>();
    assert!(samples
        .iter()
        .all(|sample| (-2_048..=2_047).contains(sample)));
    let pixels = samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    (
        encode_htj2k(&pixels, 4, 4, 1, 12, true, &options).expect("encode signed ht gray12"),
        samples,
    )
}

fn fixture_classic_signed_gray12() -> (Vec<u8>, Vec<i16>) {
    let samples = (0..16_i16)
        .map(|index| -1_700 + index * 197)
        .collect::<Vec<_>>();
    let pixels = samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    (
        encode(&pixels, 4, 4, 1, 12, true, &options).expect("encode signed classic gray12"),
        samples,
    )
}

fn fixture_gray8_irreversible() -> Vec<u8> {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: false,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode gray8 irreversible")
}

fn fixture_rgb12() -> Vec<u8> {
    let mut pixels = Vec::with_capacity(12);
    for sample in [0u16, 1023, 2047, 3071, 4095, 17] {
        pixels.extend_from_slice(&sample.to_le_bytes());
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 2, 1, 3, 12, false, &options).expect("encode rgb12")
}

fn fixture_ht_gray8() -> Vec<u8> {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, 4, 4, 1, 8, false, &options).expect("encode ht gray8")
}

fn fixture_ht_gray8_reversed() -> Vec<u8> {
    let pixels: Vec<u8> = (0..16).rev().collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, 4, 4, 1, 8, false, &options).expect("encode reversed ht gray8")
}

fn fixture_ht_gray8_offset_sized(width: u32, height: u32, offset: u8) -> Vec<u8> {
    let pixels = (0..width * height)
        .map(|index| offset.wrapping_add(index.to_le_bytes()[0].wrapping_mul(17)))
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, width, height, 1, 8, false, &options).expect("encode offset HT gray8")
}

fn fixture_ht_rgb_u8_sized(width: u32, height: u32, offset: u8) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            for pattern in [x * 3 + y * 5, x * 7 + y * 11 + 13, x * 17 + y * 19 + 29] {
                let pattern = u8::try_from(pattern & 0x3f).expect("six-bit RGB fixture pattern");
                pixels.push(offset.wrapping_add(pattern) & 0x3f);
            }
        }
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        guard_bits: 2,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, width, height, 3, 6, false, &options)
        .expect("encode sub-native HT RGB U8")
}

fn fixture_ht_rgb_u16_sized(width: u32, height: u32, offset: u16) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3 * 2);
    for y in 0..height {
        for x in 0..width {
            for pattern in [
                x * 193 + y * 257,
                x * 313 + y * 97 + 31,
                x * 71 + y * 401 + 63,
            ] {
                let pattern =
                    u16::try_from(pattern & 0x07ff).expect("twelve-bit RGB fixture pattern");
                let sample = offset.wrapping_add(pattern);
                pixels.extend_from_slice(&(sample & 0x0fff).to_le_bytes());
            }
        }
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        guard_bits: 2,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, width, height, 3, 12, false, &options)
        .expect("encode sub-native HT RGB U16")
}

fn fixture_direct_rgb8() -> Vec<u8> {
    fixture_direct_rgb8_offset(0)
}

fn fixture_direct_rgb8_offset(offset: u8) -> Vec<u8> {
    let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let pixels = pixels.map(|sample: u8| sample.saturating_add(offset));
    let options = EncodeOptions {
        reversible: false,
        guard_bits: 4,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 2, 2, 3, 8, false, &options).expect("encode direct rgb8")
}

fn fixture_direct_rgb8_variant(seed: u8) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(8 * 8 * 3);
    for y in 0..8u8 {
        for x in 0..8u8 {
            pixels.push(seed.wrapping_add(x.wrapping_mul(17)).wrapping_add(y));
            pixels.push(seed.wrapping_add(x).wrapping_add(y.wrapping_mul(19)));
            pixels.push(
                seed.wrapping_add(x.wrapping_mul(7))
                    .wrapping_add(y.wrapping_mul(11)),
            );
        }
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 8, 8, 3, 8, false, &options).expect("encode direct rgb8 variant")
}

#[path = "device/auto_tile_batch.rs"]
mod auto_tile_batch;
#[path = "device/batch_sessions.rs"]
mod batch_sessions;
#[path = "device/color_batch.rs"]
mod color_batch;
#[path = "device/color_mct_group.rs"]
mod color_mct_group;
#[path = "device/decode.rs"]
mod decode;
#[path = "device/direct_gray_requests.rs"]
mod direct_gray_requests;
#[path = "device/direct_repeated.rs"]
mod direct_repeated;
#[path = "device/direct_rgb_requests.rs"]
mod direct_rgb_requests;
#[path = "device/external_batch.rs"]
mod external_batch;
#[path = "device/grayscale_external.rs"]
mod grayscale_external;
#[path = "device/legacy_batch.rs"]
mod legacy_batch;
#[path = "device/multitile_color.rs"]
mod multitile_color;
#[path = "device/resident_batch.rs"]
mod resident_batch;
#[path = "device/tile_batch.rs"]
mod tile_batch;
