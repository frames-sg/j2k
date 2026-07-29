// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn parse_fallback_requires_an_actual_lenient_recovery() {
    let error = J2kError::Unsupported(j2k_core::Unsupported {
        what: "test metadata rejection",
    });

    assert!(!should_fallback_to_backend_after_parse_error(&error, false));
    assert!(should_fallback_to_backend_after_parse_error(&error, true));
}

#[test]
fn scaled_decode_native_context_preserves_configured_parallelism() {
    let mut decoder = J2kDecoder {
        bytes: &[],
        info: Info {
            dimensions: (1, 1),
            components: 1,
            colorspace: j2k_core::Colorspace::SGray,
            bit_depth: 8,
            tile_layout: None,
            coded_unit_layout: None,
            restart_interval: None,
            resolution_levels: 1,
        },
        image: None,
        settings: DecodeSettings::strict(),
        native_context: j2k_native::DecoderContext::default(),
    };
    decoder.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);

    let native_context = decoder.scaled_decode_native_context();

    assert_eq!(
        native_context.cpu_decode_parallelism(),
        CpuDecodeParallelism::Serial.to_native()
    );
}
