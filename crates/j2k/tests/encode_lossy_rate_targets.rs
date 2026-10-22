// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    encode_j2k_lossy, EncodeBackendPreference, J2kLossyEncodeOptions, J2kLossySamples,
    J2kRateTarget,
};
use j2k_native::{DecodeSettings, Image};

fn fixture() -> Vec<u8> {
    (0..32 * 32)
        .map(|index| u8::try_from(((index * 29) ^ (index / 7)) & 0xff).expect("masked sample"))
        .collect()
}

fn options(target: J2kRateTarget) -> J2kLossyEncodeOptions {
    let mut options = J2kLossyEncodeOptions::default()
        .with_backend(EncodeBackendPreference::CpuOnly)
        .with_rate_target(Some(target));
    options.psnr_iteration_budget = 2;
    options.psnr_tolerance_db = 0.5;
    options
}

fn assert_strictly_decodes(codestream: &[u8]) {
    let decoded = Image::new(codestream, &DecodeSettings::strict())
        .expect("rate-target codestream parses strictly")
        .decode_native()
        .expect("rate-target codestream decodes");
    assert_eq!((decoded.width, decoded.height), (32, 32));
    assert_eq!(decoded.num_components, 1);
}

#[test]
fn psnr_target_returns_a_validated_candidate_at_the_selected_scale() {
    let pixels = fixture();
    let samples = J2kLossySamples::new(&pixels, 32, 32, 1, 8, false).expect("fixture samples");
    let target = 24.0;
    let options = options(J2kRateTarget::PsnrDb(target));

    let encoded = encode_j2k_lossy(samples, &options).expect("PSNR-target encode");
    let psnr = encoded.report.psnr_db.expect("CPU validation reports PSNR");

    assert!(psnr + options.psnr_tolerance_db >= target);
    assert!(encoded.report.quantization_scale.is_finite());
    assert!(encoded.report.quantization_scale > 0.0);
    assert_strictly_decodes(&encoded.codestream);
}

#[test]
fn byte_target_reencoding_is_deterministic() {
    let pixels = fixture();
    let samples = J2kLossySamples::new(&pixels, 32, 32, 1, 8, false).expect("fixture samples");
    let target = 512_u64;
    let options = options(J2kRateTarget::Bytes(target));

    let first = encode_j2k_lossy(samples, &options).expect("first byte-target encode");
    let second = encode_j2k_lossy(samples, &options).expect("second byte-target encode");

    assert_eq!(first.codestream, second.codestream);
    assert_eq!(
        first.report.quantization_scale.to_bits(),
        second.report.quantization_scale.to_bits()
    );
    assert_eq!(first.report.actual_bytes, first.codestream.len() as u64);
    assert!(first.report.actual_bytes <= target + 512);
    assert_strictly_decodes(&first.codestream);
}

fn textured_rgb(width: u32, height: u32) -> Vec<u8> {
    let mut state = 0x2545_f491_u32;
    (0..width * height * 3)
        .map(|index| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            u8::try_from(((index % (width * 3)) / 3 + (state >> 26)) & 0xff).expect("masked sample")
        })
        .collect()
}

/// The CPU facade shares one colour transform and DWT across rate-search
/// candidates; the accelerator facade with CPU-only hooks encodes each one
/// independently. The `lossy::candidates` unit tests are the full oracle.
#[test]
fn cpu_rate_search_matches_accelerator_facade_route() {
    use j2k::{
        encode_j2k_lossy_with_accelerator, BackendKind, CpuOnlyJ2kEncodeStageAccelerator,
        J2kBlockCodingMode, J2kEncodeValidation,
    };

    let (width, height) = (96, 72);
    let pixels = textured_rgb(width, height);
    let samples = J2kLossySamples::new(&pixels, width, height, 3, 8, false).expect("samples");
    for mode in [
        J2kBlockCodingMode::HighThroughput,
        J2kBlockCodingMode::Classic,
    ] {
        for target in [
            J2kRateTarget::BitsPerPixel(0.5),
            J2kRateTarget::Bytes(3_000),
            J2kRateTarget::PsnrDb(32.0),
        ] {
            let mut options = options(target)
                .with_block_coding_mode(mode)
                .with_validation(J2kEncodeValidation::CpuRoundTrip);
            options.psnr_iteration_budget = 6;
            let cpu = encode_j2k_lossy(samples, &options).expect("CPU rate search");
            let accelerated = encode_j2k_lossy_with_accelerator(
                samples,
                &options.clone().with_backend(EncodeBackendPreference::Auto),
                BackendKind::Cpu,
                &mut CpuOnlyJ2kEncodeStageAccelerator,
            )
            .expect("accelerator-facade rate search");
            let outcome = |encoded: j2k::EncodedLossyJ2k| {
                let report = encoded.report;
                let psnr = report.psnr_db.map(f64::to_bits);
                (
                    encoded.codestream,
                    report.quantization_scale.to_bits(),
                    psnr,
                )
            };
            assert_eq!(outcome(cpu), outcome(accelerated), "{mode:?} {target:?}");
        }
    }
}
