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
