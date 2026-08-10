// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn benchmark_route_uses_resident_ht_tier1_with_cpu_packetization() {
    if !should_run_metal_runtime() {
        return;
    }

    assert_resident_ht_host_output_matches_cpu(
        512,
        512,
        1,
        &j2k_test_support::patterned_gray8(512, 512),
        MetalEncodeStageAccelerator::for_host_output_benchmark(),
    );
}

#[test]
fn auto_routes_the_verified_rgb8_1024_cell_through_resident_ht_tier1() {
    if !should_run_metal_runtime() {
        return;
    }

    assert_resident_ht_host_output_matches_cpu(
        1024,
        1024,
        3,
        &j2k_test_support::patterned_rgb8(1024, 1024),
        MetalEncodeStageAccelerator::for_auto_host_output(),
    );
}

fn assert_resident_ht_host_output_matches_cpu(
    width: u32,
    height: u32,
    components: u16,
    pixels: &[u8],
    mut accelerator: MetalEncodeStageAccelerator,
) {
    let cpu_options = lossless_options! {
        backend: EncodeBackendPreference::CpuOnly,
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        max_decomposition_levels: Some(3),
        validation: J2kEncodeValidation::External,
    };
    let expected = encode_j2k_lossless(
        J2kLosslessSamples::new(pixels, width, height, components, 8, false)
            .expect("valid CPU samples"),
        &cpu_options,
    )
    .expect("CPU HTJ2K lossless encode");
    let actual = encode_j2k_lossless_with_accelerator(
        J2kLosslessSamples::new(pixels, width, height, components, 8, false)
            .expect("valid hybrid samples"),
        &cpu_options.with_backend(EncodeBackendPreference::Auto),
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("resident HTJ2K host-output encode");

    assert!(actual.dispatch_report.ht_code_block > 0);
    assert_eq!(actual.dispatch_report.tier1_code_block, 0);
    assert_eq!(actual.dispatch_report.packetization, 0);
    let decoded = Image::new(&actual.codestream, &DecodeSettings::default())
        .expect("decode resident HTJ2K host-output codestream")
        .decode_native()
        .expect("decode resident HTJ2K host-output pixels");
    assert_decoded_bytes_match(&decoded.data, pixels);
    assert_codestream_matches(&actual.codestream, &expected.codestream);
}

fn assert_codestream_matches(actual: &[u8], expected: &[u8]) {
    if actual == expected {
        return;
    }
    let differing_bytes = actual
        .iter()
        .zip(expected)
        .filter(|(actual, expected)| actual != expected)
        .count();
    let first_difference = actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| actual != expected)
        .map(|index| (index, actual[index], expected[index]));
    panic!(
        "resident HTJ2K codestream differs from CPU: actual_len={}, expected_len={}, differing_bytes={differing_bytes}, first_difference={first_difference:?}",
        actual.len(),
        expected.len()
    );
}
