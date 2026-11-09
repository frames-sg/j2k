// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared JPEG inputs and 9/7 option variants for transcode equality tests.
//!
//! The CPU pins and the Metal and CUDA equality tests iterate the same matrix:
//! grayscale, 4:4:4, 4:2:2 and 4:2:0 sampling, odd and single-block-row
//! sizes, restart intervals, one axis above the 1024-sample staged GPU limit,
//! and decomposition-level, code-block and quality-layer variants.

use j2k_jpeg::{
    encode_jpeg_baseline, JpegBackend, JpegEncodeOptions, JpegSamples, JpegSubsampling,
};
use j2k_transcode::JpegToHtj2kOptions;

/// One 8-bit baseline JPEG input.
pub struct MatrixJpeg {
    /// Stable fixture name used in pins and failure messages.
    pub name: &'static str,
    /// Encoded JPEG bytes.
    pub bytes: Vec<u8>,
}

/// One irreversible 9/7 option variant applied on top of
/// [`JpegToHtj2kOptions::lossy_97`].
#[derive(Clone, Copy, Debug)]
pub struct MatrixVariant {
    /// Stable variant name used in pins and failure messages.
    pub name: &'static str,
    /// Requested decomposition levels (clamped to the component geometry).
    pub levels: u8,
    /// Code-block width and height exponents minus two.
    pub code_block_exp: (u8, u8),
    /// Global irreversible quantization scale.
    pub scale: f32,
    /// Quality layers; more than one selects the native precomputed route.
    pub layers: u8,
}

/// The option variants every backend must reproduce.
pub const MATRIX_VARIANTS: [MatrixVariant; 4] = [
    MatrixVariant {
        name: "default",
        levels: 1,
        code_block_exp: (4, 4),
        scale: 1.9,
        layers: 1,
    },
    MatrixVariant {
        name: "levels3_cb32x16",
        levels: 3,
        code_block_exp: (3, 2),
        scale: 1.9,
        layers: 1,
    },
    MatrixVariant {
        name: "levels5_scale1",
        levels: 5,
        code_block_exp: (4, 4),
        scale: 1.0,
        layers: 1,
    },
    MatrixVariant {
        name: "levels2_layers2",
        levels: 2,
        code_block_exp: (4, 4),
        scale: 1.9,
        layers: 2,
    },
];

/// [`JpegToHtj2kOptions::lossy_97`] with `variant` applied; the precision is
/// left at its default.
#[must_use]
pub fn matrix_options(variant: MatrixVariant) -> JpegToHtj2kOptions {
    let mut options = JpegToHtj2kOptions::lossy_97();
    options.encode_options.num_decomposition_levels = variant.levels;
    options.encode_options.code_block_width_exp = variant.code_block_exp.0;
    options.encode_options.code_block_height_exp = variant.code_block_exp.1;
    options.encode_options.irreversible_quantization_scale = variant.scale;
    options.encode_options.num_layers = variant.layers;
    options
}

/// The JPEG inputs every backend must reproduce.
#[must_use]
pub fn matrix_jpegs() -> Vec<MatrixJpeg> {
    let mut jpegs = vec![
        MatrixJpeg {
            name: "conformance_grayscale_8x8",
            bytes: j2k_test_support::JPEG_GRAYSCALE_8X8.to_vec(),
        },
        MatrixJpeg {
            name: "conformance_444_8x8",
            bytes: j2k_test_support::JPEG_BASELINE_444_8X8.to_vec(),
        },
        MatrixJpeg {
            name: "conformance_422_16x8",
            bytes: j2k_test_support::JPEG_BASELINE_422_16X8.to_vec(),
        },
        MatrixJpeg {
            name: "conformance_420_16x16",
            bytes: j2k_test_support::JPEG_BASELINE_420_16X16.to_vec(),
        },
        MatrixJpeg {
            name: "conformance_420_restart_32x16",
            bytes: j2k_test_support::JPEG_BASELINE_420_RESTART_32X16.to_vec(),
        },
        MatrixJpeg {
            name: "conformance_444_restart_67x45",
            bytes: j2k_test_support::JPEG_BASELINE_444_RESTART_67X45.to_vec(),
        },
    ];
    for spec in SYNTHETIC {
        jpegs.push(MatrixJpeg {
            name: spec.name,
            bytes: synthetic_jpeg(spec),
        });
    }
    jpegs
}

#[derive(Clone, Copy)]
struct SyntheticSpec {
    name: &'static str,
    width: usize,
    height: usize,
    subsampling: JpegSubsampling,
    restart_interval: Option<u16>,
}

const SYNTHETIC: [SyntheticSpec; 9] = [
    SyntheticSpec {
        name: "synthetic_444_224_restart",
        width: 224,
        height: 224,
        subsampling: JpegSubsampling::Ybr444,
        restart_interval: Some(28),
    },
    SyntheticSpec {
        name: "synthetic_420_224_restart",
        width: 224,
        height: 224,
        subsampling: JpegSubsampling::Ybr420,
        restart_interval: Some(28),
    },
    SyntheticSpec {
        name: "synthetic_420_512_restart",
        width: 512,
        height: 512,
        subsampling: JpegSubsampling::Ybr420,
        restart_interval: Some(64),
    },
    SyntheticSpec {
        name: "synthetic_gray_100x37_restart",
        width: 100,
        height: 37,
        subsampling: JpegSubsampling::Gray,
        restart_interval: Some(3),
    },
    SyntheticSpec {
        name: "synthetic_422_131x77",
        width: 131,
        height: 77,
        subsampling: JpegSubsampling::Ybr422,
        restart_interval: None,
    },
    SyntheticSpec {
        name: "synthetic_420_97x61_restart",
        width: 97,
        height: 61,
        subsampling: JpegSubsampling::Ybr420,
        restart_interval: Some(5),
    },
    SyntheticSpec {
        name: "synthetic_444_45x67_restart",
        width: 45,
        height: 67,
        subsampling: JpegSubsampling::Ybr444,
        restart_interval: Some(1),
    },
    SyntheticSpec {
        name: "synthetic_gray_1031x9",
        width: 1031,
        height: 9,
        subsampling: JpegSubsampling::Gray,
        restart_interval: None,
    },
    SyntheticSpec {
        name: "synthetic_420_1100x1030_restart",
        width: 1100,
        height: 1030,
        subsampling: JpegSubsampling::Ybr420,
        restart_interval: Some(69),
    },
];

fn synthetic_jpeg(spec: SyntheticSpec) -> Vec<u8> {
    let width = u32::try_from(spec.width).expect("fixture width fits u32");
    let height = u32::try_from(spec.height).expect("fixture height fits u32");
    let gray;
    let rgb;
    let samples = if matches!(spec.subsampling, JpegSubsampling::Gray) {
        gray = (0..spec.width * spec.height)
            .map(|index| pattern_sample(index % spec.width, index / spec.width, 0))
            .collect::<Vec<_>>();
        JpegSamples::Gray8 {
            data: &gray,
            width,
            height,
        }
    } else {
        rgb = (0..spec.width * spec.height * 3)
            .map(|index| {
                let pixel = index / 3;
                pattern_sample(pixel % spec.width, pixel / spec.width, index % 3)
            })
            .collect::<Vec<_>>();
        JpegSamples::Rgb8 {
            data: &rgb,
            width,
            height,
        }
    };
    encode_jpeg_baseline(
        samples,
        JpegEncodeOptions {
            quality: 90,
            subsampling: spec.subsampling,
            restart_interval: spec.restart_interval,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("synthetic matrix JPEG encodes")
    .data
}

/// Smooth gradients plus hard wrap-around edges and a checkerboard so every
/// subband carries energy.
fn pattern_sample(x: usize, y: usize, channel: usize) -> u8 {
    let ramp = match channel {
        0 => x * 5 + y * 3 + 17,
        1 => (x ^ y) * 2 + 41,
        _ => x * 11 + y * 13 + 73,
    };
    let checker = if (x / 3 + y / 5).is_multiple_of(2) {
        0
    } else {
        37
    };
    u8::try_from((ramp + checker) & 0xff).expect("masked sample fits u8")
}

/// FNV-1a 64-bit digest used to pin codestream bytes in tests.
#[must_use]
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}
