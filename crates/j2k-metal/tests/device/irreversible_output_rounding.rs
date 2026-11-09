// SPDX-License-Identifier: MIT OR Apache-2.0

//! Irreversible (9/7) integer output must match the CPU byte for byte. The CPU
//! rounds each centered sample ties-to-even and only then adds the unsigned
//! level shift (`round_ties_even_then_add`). Adding the shift first costs one
//! bit of f32 precision, so `k + 0.5 - ulp` becomes a tie and rounds up: Metal
//! was one code value high on roughly ten samples per million. Integer output
//! of wider samples, including their scaling to `U8`, must match it too.

use super::*;

const SIZE: u32 = 512;
const ROI: Rect = Rect {
    x: SIZE / 4,
    y: SIZE / 4,
    w: SIZE / 2,
    h: SIZE / 2,
};
const BATCH: usize = 3;

#[derive(Clone, Copy, Debug)]
struct LossyCase {
    components: u16,
    ht: bool,
    use_mct: bool,
}

const CASES: [LossyCase; 5] = [
    LossyCase {
        components: 3,
        ht: true,
        use_mct: true,
    },
    LossyCase {
        components: 3,
        ht: false,
        use_mct: true,
    },
    LossyCase {
        components: 3,
        ht: true,
        use_mct: false,
    },
    LossyCase {
        components: 1,
        ht: true,
        use_mct: false,
    },
    LossyCase {
        components: 1,
        ht: false,
        use_mct: false,
    },
];

fn lossy_fixture(case: LossyCase) -> Vec<u8> {
    let components = u32::from(case.components);
    let mut state = 0x1234_5678_u32;
    let pixels: Vec<u8> = (0..SIZE * SIZE * components)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let (pixel, component) = (index / components, index % components);
            let (x, y) = (pixel % SIZE, pixel / SIZE);
            let base = (x * 3 + y * 5 + component * 40) / 9 % 200 + 20;
            u8::try_from(base + state % 17).expect("sample fits u8")
        })
        .collect();
    let options = EncodeOptions {
        reversible: false,
        use_mct: case.use_mct,
        irreversible_quantization_scale: 4.0,
        ..EncodeOptions::default()
    };
    if case.ht {
        encode_htj2k(&pixels, SIZE, SIZE, case.components, 8, false, &options)
    } else {
        encode(&pixels, SIZE, SIZE, case.components, 8, false, &options)
    }
    .expect("encode lossy 9/7 fixture")
}

/// Samples whose unrounded CPU output lands exactly on `k + 0.5` after the
/// level shift: the inputs on which shift-then-round disagrees with the CPU.
#[expect(
    clippy::float_cmp,
    reason = "a tie is an exact property of the f32 value"
)]
fn shifted_tie_count(bytes: &[u8]) -> usize {
    let image = j2k_native::Image::new(bytes, &j2k_native::DecodeSettings::default())
        .expect("inspect lossy fixture");
    let mut context = j2k_native::DecoderContext::default();
    let components = image
        .decode_components_with_context(&mut context)
        .expect("CPU float component decode");
    components
        .planes()
        .iter()
        .flat_map(j2k_native::ComponentPlane::samples)
        .filter(|sample| {
            let doubled = **sample * 2.0;
            doubled == doubled.floor() && doubled.rem_euclid(2.0) == 1.0
        })
        .count()
}

fn decode_single(
    bytes: &[u8],
    request: MetalDecodeRequest,
    session: &MetalBackendSession,
) -> Vec<u8> {
    J2kDecoder::new(bytes)
        .expect("lossy decoder")
        .decode_request_to_device_with_session(request, session)
        .expect("lossy decode")
        .as_bytes()
        .expect("lossy readback")
        .into_owned()
}

fn decode_batch(bytes: &Arc<[u8]>, fmt: PixelFormat, backend: BackendRequest) -> Vec<u8> {
    let mut batch = MetalTileBatch::with_capacity(BATCH);
    for _ in 0..BATCH {
        batch
            .push_shared_tile_request(Arc::clone(bytes), MetalDecodeRequest::full(fmt, backend))
            .expect("queue lossy batch item");
    }
    let surfaces = batch.decode_all().expect("lossy batch decode");
    assert_eq!(surfaces.len(), BATCH);
    surfaces
        .iter()
        .flat_map(|surface| {
            surface
                .as_bytes()
                .expect("lossy batch readback")
                .into_owned()
        })
        .collect()
}

fn compare(label: &str, cpu: &[u8], metal: &[u8], failures: &mut Vec<String>) {
    if cpu.len() != metal.len() {
        failures.push(format!("{label}: lengths {} / {}", cpu.len(), metal.len()));
        return;
    }
    let mismatches: Vec<usize> = (0..cpu.len()).filter(|&i| cpu[i] != metal[i]).collect();
    if let Some(&first) = mismatches.first() {
        failures.push(format!(
            "{label}: {} of {} bytes differ; first at byte {first} (CPU {}, Metal {})",
            mismatches.len(),
            cpu.len(),
            cpu[first],
            metal[first],
        ));
    }
}

/// Compare the CPU and explicit Metal output of `bytes` for every pixel
/// format of its component count, through full, region and half-resolution
/// single decodes and a full-image tile batch.
fn compare_requests(
    label: &str,
    bytes: &[u8],
    components: u16,
    roi: Rect,
    session: &MetalBackendSession,
    failures: &mut Vec<String>,
) {
    let shared: Arc<[u8]> = Arc::from(bytes);
    let formats: &[PixelFormat] = if components == 1 {
        &[PixelFormat::Gray8, PixelFormat::Gray16]
    } else {
        &[PixelFormat::Rgb8, PixelFormat::Rgba8, PixelFormat::Rgb16]
    };
    for &fmt in formats {
        let requests = [
            ("full", MetalDecodeRequest::full(fmt, BackendRequest::Cpu)),
            (
                "roi",
                MetalDecodeRequest::region(fmt, roi, BackendRequest::Cpu),
            ),
            (
                "half",
                MetalDecodeRequest::scaled(fmt, Downscale::Half, BackendRequest::Cpu),
            ),
        ];
        for (operation, request) in requests {
            let cpu = decode_single(bytes, request, session);
            let metal = MetalDecodeRequest {
                backend: BackendRequest::Metal,
                ..request
            };
            let metal = decode_single(bytes, metal, session);
            compare(
                &format!("{label} {fmt:?} {operation}"),
                &cpu,
                &metal,
                failures,
            );
        }
        let cpu = decode_batch(&shared, fmt, BackendRequest::Cpu);
        let metal = decode_batch(&shared, fmt, BackendRequest::Metal);
        compare(&format!("{label} {fmt:?} batch"), &cpu, &metal, failures);
    }
}

#[test]
fn explicit_metal_irreversible_integer_output_matches_cpu_rounding() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal session");
    let mut failures = Vec::new();
    for case in CASES {
        let bytes = lossy_fixture(case);
        assert!(
            shifted_tie_count(&bytes) > 0,
            "{case:?}: fixture has no shifted ties, so it cannot detect the rounding order"
        );
        compare_requests(
            &format!("{case:?}"),
            &bytes,
            case.components,
            ROI,
            &session,
            &mut failures,
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

const WIDE_SIZE: u32 = 128;

/// A 9/7 image of `bit_depth` (9 to 16) bit samples: ramps with noise, every
/// sample in two native-endian bytes.
fn lossy_wide_fixture(components: u16, bit_depth: u8, ht: bool) -> Vec<u8> {
    let span = 1_u32 << bit_depth;
    let channels = u32::from(components);
    let mut state = 0x9e37_79b9_u32;
    let pixels: Vec<u8> = (0..WIDE_SIZE * WIDE_SIZE * channels)
        .flat_map(|index| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let (pixel, component) = (index / channels, index % channels);
            let (x, y) = (pixel % WIDE_SIZE, pixel / WIDE_SIZE);
            let ramp = (x * 7 + y * 11 + component * 29) * (span / 256);
            let sample = (ramp + state % (span / 8)) & (span - 1);
            u16::try_from(sample)
                .expect("sample fits u16")
                .to_ne_bytes()
        })
        .collect();
    let options = EncodeOptions {
        reversible: false,
        use_mct: components == 3,
        irreversible_quantization_scale: 4.0,
        ..EncodeOptions::default()
    };
    if ht {
        encode_htj2k(
            &pixels, WIDE_SIZE, WIDE_SIZE, components, bit_depth, false, &options,
        )
    } else {
        encode(
            &pixels, WIDE_SIZE, WIDE_SIZE, components, bit_depth, false, &options,
        )
    }
    .unwrap_or_else(|error| panic!("encode {bit_depth}-bit 9/7 fixture: {error}"))
}

/// `U8` output of a 9- to 16-bit sample `x` is `round(x / max * 255)`, which
/// lands within `1 / (2 * max)` of a half for some codes; f32 rounding there
/// must agree with the CPU. A lossless image holding every code value once,
/// in each component, checks all of them.
#[test]
fn metal_u8_output_scales_every_code_value_like_the_cpu() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal session");
    let mut failures = Vec::new();
    for bit_depth in 9_u8..=16 {
        let codes = 1_u32 << bit_depth;
        let (width, height) = (256, codes / 256);
        for components in [1_u16, 3] {
            let pixels: Vec<u8> = (0..codes)
                .flat_map(|code| {
                    // Each component visits every code in its own order.
                    (0..u32::from(components)).flat_map(move |component| {
                        let value = (code + component * (codes / 3 + 1)) % codes;
                        u16::try_from(value).expect("code fits u16").to_ne_bytes()
                    })
                })
                .collect();
            let options = EncodeOptions {
                reversible: true,
                num_decomposition_levels: 2,
                ..EncodeOptions::default()
            };
            let bytes = encode(
                &pixels, width, height, components, bit_depth, false, &options,
            )
            .expect("encode every-code fixture");
            let formats: &[PixelFormat] = if components == 1 {
                &[PixelFormat::Gray8]
            } else {
                &[PixelFormat::Rgb8, PixelFormat::Rgba8]
            };
            for &fmt in formats {
                let cpu = decode_single(
                    &bytes,
                    MetalDecodeRequest::full(fmt, BackendRequest::Cpu),
                    &session,
                );
                let metal = decode_single(
                    &bytes,
                    MetalDecodeRequest::full(fmt, BackendRequest::Metal),
                    &session,
                );
                compare(
                    &format!("{bit_depth}-bit {components}-component every code {fmt:?}"),
                    &cpu,
                    &metal,
                    &mut failures,
                );
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Irreversible 9- to 16-bit output scales to `U8` and stores at native
/// precision in `U16`; both must match the CPU byte for byte.
#[test]
fn explicit_metal_irreversible_wide_output_matches_cpu() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal session");
    let roi = Rect {
        x: WIDE_SIZE / 4,
        y: WIDE_SIZE / 4,
        w: WIDE_SIZE / 2,
        h: WIDE_SIZE / 2,
    };
    let mut failures = Vec::new();
    for bit_depth in 9_u8..=16 {
        for (components, ht) in [(3_u16, true), (3, false), (1, true), (1, false)] {
            let bytes = lossy_wide_fixture(components, bit_depth, ht);
            compare_requests(
                &format!("{bit_depth}-bit {components}-component ht={ht}"),
                &bytes,
                components,
                roi,
                &session,
                &mut failures,
            );
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
