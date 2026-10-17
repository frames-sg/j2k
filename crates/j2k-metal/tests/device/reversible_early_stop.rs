// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reversible code blocks whose coding passes stop early take the T.800
//! Equation E-8 midpoint (r = 1/2) on the Metal classic Tier-1 kernels, as on
//! the CPU, whichever route an image takes.

use super::*;
use j2k_test_support::{
    auto_routing_sha256, REVERSIBLE_EARLY_STOP_LOSSLESS, REVERSIBLE_EARLY_STOP_LOSSLESS_U16LE,
    REVERSIBLE_EARLY_STOP_RATE_LIMITED, REVERSIBLE_EARLY_STOP_RATE_LIMITED_RGB8,
    REVERSIBLE_EARLY_STOP_RATE_LIMITED_RGB8_SHA256, REVERSIBLE_EARLY_STOP_RATE_LIMITED_U16LE,
};

#[test]
fn explicit_metal_decode_of_early_stopped_reversible_blocks_matches_the_fixtures() {
    if !should_run_metal_runtime() {
        return;
    }

    for (label, codestream, expected) in [
        (
            "lossless",
            REVERSIBLE_EARLY_STOP_LOSSLESS,
            REVERSIBLE_EARLY_STOP_LOSSLESS_U16LE,
        ),
        (
            "rate-limited",
            REVERSIBLE_EARLY_STOP_RATE_LIMITED,
            REVERSIBLE_EARLY_STOP_RATE_LIMITED_U16LE,
        ),
    ] {
        let surface = J2kDecoder::new(codestream)
            .unwrap_or_else(|error| panic!("{label} decoder: {error}"))
            .decode_to_device(PixelFormat::Gray16, BackendRequest::Metal)
            .unwrap_or_else(|error| panic!("{label} explicit Metal decode: {error}"));
        assert_eq!(surface.backend_kind(), BackendKind::Metal, "{label}");
        assert_eq!(
            surface.as_bytes().expect("surface byte access").as_ref(),
            expected,
            "{label}"
        );
    }

    let surface = J2kDecoder::new(REVERSIBLE_EARLY_STOP_RATE_LIMITED_RGB8)
        .expect("RGB8 decoder")
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Metal)
        .expect("RGB8 explicit Metal decode");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(
        auto_routing_sha256(&surface.as_bytes().expect("surface byte access")),
        REVERSIBLE_EARLY_STOP_RATE_LIMITED_RGB8_SHA256
    );
}

#[test]
fn cooperative_batch_of_early_stopped_reversible_blocks_matches_on_cpu_and_gpu() {
    if !should_run_metal_runtime() {
        return;
    }

    // Distinct allocations, so the batch does not treat them as one repeated plan.
    let inputs = (0..8)
        .map(|_| EncodedImage::full(Arc::from(REVERSIBLE_EARLY_STOP_RATE_LIMITED_RGB8)))
        .collect::<Vec<_>>();
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let cpu = CpuBatchDecoder::new(options)
        .decode(inputs.clone())
        .expect("CPU batch decode");
    let CpuBatchSamples::U8(expected) = cpu.groups()[0].samples() else {
        panic!("RGB8 batch storage");
    };
    for image in expected.chunks_exact(expected.len() / inputs.len()) {
        assert_eq!(
            auto_routing_sha256(image),
            REVERSIBLE_EARLY_STOP_RATE_LIMITED_RGB8_SHA256
        );
    }

    let mut decoder = MetalBatchDecoder::system_default_with_options(options).unwrap();
    let prepared = decoder.prepare(inputs).unwrap();

    // Every image's Tier-1 on the GPU.
    let strict = decoder
        .decode_prepared(&prepared)
        .expect("GPU Tier-1 decode");
    assert_eq!(strict.errors(), []);
    let group = &strict.groups()[0];
    assert_eq!(group.dispatch_report().cpu_tier1_images, 0);
    assert!(group.dispatch_report().classic_tier1 > 0);
    assert_eq!(&completed_resident_batch_bytes(group), expected);

    // Some images' Tier-1 on the CPU, the rest on the GPU.
    let cooperative = decoder
        .decode_prepared_cooperative(&prepared, std::num::NonZeroUsize::new(12).unwrap())
        .expect("cooperative decode");
    assert_eq!(cooperative.errors(), []);
    assert!(cooperative.group_errors().is_empty());
    let group = &cooperative.groups()[0];
    assert!(group.dispatch_report().cpu_tier1_images > 0);
    assert!(group.dispatch_report().classic_tier1 > 0);
    assert_eq!(&completed_resident_batch_bytes(group), expected);
}
