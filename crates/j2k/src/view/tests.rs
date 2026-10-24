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
fn fresh_native_context_preserves_configured_parallelism() {
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

    let native_context = decoder.fresh_native_context();

    assert_eq!(
        native_context.cpu_decode_parallelism(),
        CpuDecodeParallelism::Serial.to_native()
    );
}

#[test]
fn region_decode_leaves_no_decode_buffers_in_the_decoder() {
    let pixels: Vec<u8> = (0..32 * 32 * 3)
        .map(|index| u8::try_from((index * 7 + 3) & 0xFF).expect("masked fixture byte"))
        .collect();
    let options = j2k_native::EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        ..j2k_native::EncodeOptions::default()
    };
    let codestream = j2k_native::encode(&pixels, 32, 32, 3, 8, false, &options).expect("encode");
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let roi = Rect {
        x: 5,
        y: 3,
        w: 17,
        h: 20,
    };
    let stride = roi.w as usize * 3;
    let mut out = vec![0_u8; stride * roi.h as usize];

    decoder
        .decode_region_into(
            &mut J2kScratchPool::new(),
            &mut out,
            stride,
            PixelFormat::Rgb8,
            roi,
        )
        .expect("region decode");

    let stats = decoder.native_context.workspace_stats();
    assert_eq!(stats.retained_component_bytes(), 0);
    assert_eq!(stats.retained_scratch_bytes(), 0);
}
