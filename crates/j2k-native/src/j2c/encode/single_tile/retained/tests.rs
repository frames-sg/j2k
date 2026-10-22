// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;

use super::*;
use crate::{
    encode_with_roi_regions, irreversible_quantization_profile_scale_ceiling,
    irreversible_quantization_scale_bounds, EncodeProgressionOrder,
};

struct Case {
    name: &'static str,
    size: (u32, u32),
    components: u16,
    bit_depth: u8,
    signed: bool,
    options: EncodeOptions,
    roi: Vec<EncodeRoiRegion>,
    /// Whether successful calls keep the transform.
    retains: bool,
    /// Probed scales; `None` runs a rate-search-like sequence.
    scales: Option<Vec<f32>>,
}

fn case(
    name: &'static str,
    size: (u32, u32),
    components: u16,
    bit_depth: u8,
    options: EncodeOptions,
) -> Case {
    Case {
        name,
        size,
        components,
        bit_depth,
        signed: false,
        options,
        roi: Vec::new(),
        retains: true,
        scales: None,
    }
}

impl Case {
    fn pixels(&self) -> Vec<u8> {
        let (width, height) = self.size;
        let bytes_per_sample = usize::from(self.bit_depth).div_ceil(8);
        let count = width as usize * height as usize * usize::from(self.components);
        let mask = (1_u64 << self.bit_depth) - 1;
        let mut state = u64::from(width) * 31 + u64::from(self.components);
        let mut pixels = Vec::with_capacity(count * bytes_per_sample);
        // A gradient plus noise, so coarse scales still leave coefficients.
        for index in 0..count as u64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let gradient = index.wrapping_mul(mask / 97 + 1);
            let value = (gradient ^ (state >> 37) & (mask >> 2)) & mask;
            pixels.extend_from_slice(&value.to_le_bytes()[..bytes_per_sample]);
        }
        pixels
    }

    fn encoder<'a>(&'a self, pixels: &'a [u8]) -> IrreversibleScaleEncoder<'a> {
        let (width, height) = self.size;
        IrreversibleScaleEncoder::new(
            pixels,
            width,
            height,
            self.components,
            self.bit_depth,
            self.signed,
            self.options.clone(),
            &self.roi,
        )
    }

    fn options_at(&self, scale: f32) -> EncodeOptions {
        EncodeOptions {
            irreversible_quantization_scale: scale,
            ..self.options.clone()
        }
    }

    fn reference(&self, pixels: &[u8], scale: f32) -> EncodeResult<Vec<u8>> {
        let (width, height) = self.size;
        encode_with_roi_regions(
            pixels,
            width,
            height,
            self.components,
            self.bit_depth,
            self.signed,
            &self.options_at(scale),
            &self.roi,
        )
    }

    fn reference_with_cap(&self, pixels: &[u8], scale: f32, cap: usize) -> EncodeResult<Vec<u8>> {
        let (width, height) = self.size;
        let options = self.options_at(scale);
        let session = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), cap)?;
        super::super::encode_impl(
            pixels,
            width,
            height,
            self.components,
            self.bit_depth,
            self.signed,
            &options,
            block_coding_mode(&options),
            &self.roi,
            &[],
            &session,
            &mut CpuOnlyJ2kEncodeStageAccelerator,
        )
        .map_err(NativeEncodePipelineError::into_encode_error)
    }

    /// Floor, doublings, bisection, repeats, and the admissible ceiling.
    fn scales(&self) -> Vec<f32> {
        self.scales.clone().unwrap_or_else(|| {
            let (lower, upper) =
                irreversible_quantization_scale_bounds(self.bit_depth, &self.options);
            let floor = lower.max(1.0);
            [1.0, 2.0, 4.0, 3.0, 3.5, 3.0, 1.0]
                .map(|factor| floor * factor)
                .into_iter()
                .chain([upper])
                .collect()
        })
    }
}

fn lossy(levels: u8, ht: bool) -> EncodeOptions {
    EncodeOptions {
        reversible: false,
        num_decomposition_levels: levels,
        use_ht_block_coding: ht,
        validate_high_throughput_codestream: false,
        ..EncodeOptions::default()
    }
}

fn assert_same(actual: &EncodeResult<Vec<u8>>, expected: &EncodeResult<Vec<u8>>, what: &str) {
    assert!(
        actual == expected,
        "{what}: retained {:?} != reference {:?}",
        actual.as_ref().map(Vec::len),
        expected.as_ref().map(Vec::len)
    );
}

fn profiled(components: u8, levels: u8) -> Vec<Vec<f32>> {
    let bands = 1 + 3 * levels;
    (0..components)
        .map(|component| {
            (0..bands)
                .map(|band| 1.0 + 0.25 * f32::from(component + band % 3))
                .collect()
        })
        .collect()
}

/// Geometry, sample types, options, ROI, routes that must not retain, and
/// errors before, between, and after retained candidates.
#[expect(
    clippy::too_many_lines,
    reason = "one table of identity rows reads better than several helpers"
)]
fn matrix() -> Vec<Case> {
    let profile_errors = EncodeOptions {
        irreversible_quantization_profile: Some(vec![vec![1.0, 4.0, 4.0, 64.0]; 3]),
        ..lossy(1, true)
    };
    let ceiling = irreversible_quantization_profile_scale_ceiling(8, &profile_errors)
        .expect("non-unity profile has a ceiling");
    let roi = |component, (x, y), (width, height), shift| EncodeRoiRegion {
        component,
        x,
        y,
        width,
        height,
        shift,
    };
    let reference_only = |name, options| Case {
        retains: false,
        scales: Some(vec![1.0, 2.0, 1.5]),
        ..case(name, (40, 24), 3, 8, options)
    };
    vec![
        case("1x1 gray ht", (1, 1), 1, 8, lossy(0, true)),
        case("1x37 gray classic", (1, 37), 1, 8, lossy(3, false)),
        // Plain single-layer HT rows are eligible for the whole-tile hook,
        // which the CPU-only accelerator declines.
        case("37x1 gray ht", (37, 1), 1, 8, lossy(3, true)),
        case("7x9 rgb ht", (7, 9), 3, 8, lossy(1, true)),
        Case {
            signed: true,
            ..case("17x13 signed rgb classic", (17, 13), 3, 8, lossy(3, false))
        },
        case("64x48 rgb ht capped levels", (64, 48), 3, 8, lossy(6, true)),
        case(
            "65x33 12-bit cmyk classic layers",
            (65, 33),
            4,
            12,
            EncodeOptions {
                num_layers: 3,
                quality_layer_byte_targets: vec![96, 256, 900],
                ..lossy(3, false)
            },
        ),
        case(
            "33x17 u16 five-component ht layers",
            (33, 17),
            5,
            16,
            EncodeOptions {
                num_layers: 3,
                quality_layer_byte_targets: vec![128, 300, 1_200],
                ..lossy(2, true)
            },
        ),
        case("9x7 29-bit classic", (9, 7), 1, 29, lossy(2, false)),
        case(
            "40x24 rgb ht markers",
            (40, 24),
            3,
            8,
            EncodeOptions {
                write_sop: true,
                write_eph: true,
                write_plt: true,
                write_tlm: true,
                tile_part_packet_limit: Some(2),
                code_block_width_exp: 2,
                code_block_height_exp: 2,
                precinct_exponents: vec![(4, 4), (5, 5), (5, 5)],
                progression_order: EncodeProgressionOrder::Rpcl,
                ..lossy(2, true)
            },
        ),
        case(
            "40x24 rgb classic ppm",
            (40, 24),
            3,
            8,
            EncodeOptions {
                write_ppm: true,
                progression_order: EncodeProgressionOrder::Pcrl,
                ..lossy(2, false)
            },
        ),
        case(
            "31x29 rgb ht profile",
            (31, 29),
            3,
            8,
            EncodeOptions {
                irreversible_quantization_profile: Some(profiled(3, 2)),
                ..lossy(2, true)
            },
        ),
        Case {
            signed: true,
            ..case(
                "31x29 signed 10-bit classic profile",
                (31, 29),
                3,
                10,
                EncodeOptions {
                    irreversible_quantization_profile: Some(profiled(3, 3)),
                    ..lossy(3, false)
                },
            )
        },
        Case {
            roi: vec![roi(1, (5, 7), (17, 11), 14)],
            ..case("48x40 rgb ht roi", (48, 40), 3, 8, lossy(3, true))
        },
        Case {
            roi: vec![roi(0, (0, 0), (48, 40), 20)],
            ..case(
                "48x40 gray classic whole roi",
                (48, 40),
                2,
                8,
                lossy(2, false),
            )
        },
        reference_only(
            "multi-tile",
            EncodeOptions {
                tile_size: Some((16, 16)),
                ..lossy(2, true)
            },
        ),
        reference_only(
            "reversible",
            EncodeOptions {
                reversible: true,
                ..lossy(2, false)
            },
        ),
        reference_only(
            "self-validated ht",
            EncodeOptions {
                validate_high_throughput_codestream: true,
                ..lossy(2, true)
            },
        ),
        Case {
            scales: Some(vec![
                f32::NAN,
                1.0,
                ceiling * 4.0,
                2.0,
                0.0,
                -1.0,
                f32::INFINITY,
                3.0,
            ]),
            ..case("profile errors", (24, 16), 3, 8, profile_errors)
        },
    ]
}

#[test]
fn scale_encoder_matches_independent_encode_at_every_scale() {
    for case in matrix() {
        let pixels = case.pixels();
        let samples = pixels.len() / usize::from(case.bit_depth).div_ceil(8);
        let mut encoder = case.encoder(&pixels);
        // A failure while retaining switches retention off for good.
        let mut retaining = case.retains;
        let mut successes = 0;
        for scale in case.scales() {
            let held = encoder.transform.is_some();
            let expected = case.reference(&pixels, scale);
            let actual = encoder.encode_at_scale(scale);
            let what = format!("{} at scale {scale}", case.name);
            assert_same(&actual, &expected, &what);
            if expected.is_err() {
                retaining &= !held;
                continue;
            }
            successes += 1;
            // Falling back would hide a retained-path defect behind correct
            // bytes, so a retaining case must still hold its transform.
            assert_eq!(encoder.transform.is_some(), retaining, "{what}");
            if retaining {
                assert!(encoder.retained_bytes() >= 4 * samples, "{what}");
            }
        }
        assert!(successes > 0, "{} never encoded", case.name);
    }
}

/// Smallest cap at which `succeeds` holds; every check here is monotonic.
fn minimal_cap(succeeds: impl Fn(usize) -> bool) -> usize {
    let (mut failing, mut passing) = (0, DEFAULT_MAX_CODEC_BYTES);
    assert!(succeeds(passing) && !succeeds(failing));
    while passing - failing > 1 {
        let middle = failing + (passing - failing) / 2;
        if succeeds(middle) {
            passing = middle;
        } else {
            failing = middle;
        }
    }
    passing
}

#[test]
fn scale_encoder_budget_matches_reference() {
    // The forced fallbacks need a fine scale to cost more than a coarse one.
    // Classic's floors differ about twofold, HT's by a few hundred bytes of
    // coded data that platform scratch sizes can hide, so only classic runs
    // them.
    for (case, fallbacks) in [
        (case("ht", (48, 40), 3, 8, lossy(3, true)), false),
        (case("classic", (48, 40), 3, 8, lossy(2, false)), true),
    ] {
        let pixels = case.pixels();
        let capped = |cap| {
            let mut encoder = case.encoder(&pixels);
            encoder.cap = cap;
            encoder
        };
        let retains = |cap, scales: &[f32]| {
            let mut encoder = capped(cap);
            for &scale in scales {
                assert!(encoder.encode_at_scale(scale).is_ok(), "{}", case.name);
            }
            encoder.transform.is_some()
        };
        // Under- or over-charged retained bytes would move these edges.
        let [fine, coarse] = [1.0_f32, 64.0].map(|scale| {
            let floor = minimal_cap(|cap| case.reference_with_cap(&pixels, scale, cap).is_ok());
            let first = minimal_cap(|cap| retains(cap, &[scale]));
            let later = minimal_cap(|cap| retains(cap, &[1e6, scale]));
            assert_eq!((first, later), (floor, floor), "{} at {scale}", case.name);
            floor
        });
        if !fallbacks {
            continue;
        }
        assert!(fine > coarse, "{}: fine scales need more memory", case.name);

        // Forced fallbacks return the ordinary encode and drop the transform:
        // the first candidate below its floor, then a fine later candidate.
        let mut encoder = capped(fine - 1);
        let actual = encoder.encode_at_scale(1.0);
        assert_same(&actual, &case.reference(&pixels, 1.0), case.name);
        assert!(encoder.transform.is_none(), "{}", case.name);
        let mut encoder = capped(coarse);
        for (call, scale) in [64.0_f32, 1.0, 64.0].into_iter().enumerate() {
            let actual = encoder.encode_at_scale(scale);
            assert_same(&actual, &case.reference(&pixels, scale), case.name);
            assert_eq!(encoder.transform.is_some(), call == 0, "{}", case.name);
        }
    }
}
