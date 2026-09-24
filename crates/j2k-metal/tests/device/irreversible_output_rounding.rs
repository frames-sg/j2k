// SPDX-License-Identifier: MIT OR Apache-2.0

//! Irreversible (9/7) integer output must match the CPU byte for byte. The CPU
//! rounds each centered sample ties-to-even and only then adds the unsigned
//! level shift (`round_ties_even_then_add`). Adding the shift first costs one
//! bit of f32 precision, so `k + 0.5 - ulp` becomes a tie and rounds up: Metal
//! was one code value high on roughly ten samples per million.

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

type RequestBuilder = fn(PixelFormat, BackendRequest) -> MetalDecodeRequest;

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
        let shared: Arc<[u8]> = Arc::from(bytes.as_slice());
        let formats: &[PixelFormat] = if case.components == 1 {
            &[PixelFormat::Gray8, PixelFormat::Gray16]
        } else {
            &[PixelFormat::Rgb8, PixelFormat::Rgba8, PixelFormat::Rgb16]
        };
        for &fmt in formats {
            let requests: [(&str, RequestBuilder); 3] = [
                ("full", MetalDecodeRequest::full),
                ("roi", |fmt, backend| {
                    MetalDecodeRequest::region(fmt, ROI, backend)
                }),
                ("half", |fmt, backend| {
                    MetalDecodeRequest::scaled(fmt, Downscale::Half, backend)
                }),
            ];
            for (operation, request) in requests {
                let cpu = decode_single(&bytes, request(fmt, BackendRequest::Cpu), &session);
                let metal = decode_single(&bytes, request(fmt, BackendRequest::Metal), &session);
                compare(
                    &format!("{case:?} {fmt:?} {operation}"),
                    &cpu,
                    &metal,
                    &mut failures,
                );
            }
            let cpu = decode_batch(&shared, fmt, BackendRequest::Cpu);
            let metal = decode_batch(&shared, fmt, BackendRequest::Metal);
            compare(
                &format!("{case:?} {fmt:?} batch"),
                &cpu,
                &metal,
                &mut failures,
            );
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
