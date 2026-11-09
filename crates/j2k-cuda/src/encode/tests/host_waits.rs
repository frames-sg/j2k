// SPDX-License-Identifier: MIT OR Apache-2.0

//! A whole-tile CUDA encode queues each stage's kernels back to back and
//! waits on the host once per engine stage and for the readbacks it needs,
//! not after every kernel.

#[cfg(feature = "cuda-runtime")]
use super::{
    encode_j2k_lossy_with_accelerator, BackendKind, CudaEncodeStageAccelerator,
    EncodeBackendPreference, J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions,
    J2kLosslessSamples, J2kLossyEncodeOptions, J2kLossySamples,
};

#[cfg(feature = "cuda-runtime")]
fn encode(
    lossless: bool,
    pixels: &[u8],
    size: u32,
    components: u16,
    accelerator: &mut CudaEncodeStageAccelerator,
) -> Vec<u8> {
    if lossless {
        let samples = J2kLosslessSamples::new(pixels, size, size, components, 8, false)
            .expect("valid lossless samples");
        let options = J2kLosslessEncodeOptions::default()
            .with_backend(EncodeBackendPreference::RequireDevice)
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_validation(J2kEncodeValidation::External);
        j2k::encode_j2k_lossless_with_accelerator(samples, &options, BackendKind::Cuda, accelerator)
            .expect("CUDA lossless encode")
            .codestream
    } else {
        let samples = J2kLossySamples::new(pixels, size, size, components, 8, false)
            .expect("valid lossy samples");
        let options = J2kLossyEncodeOptions::default()
            .with_backend(EncodeBackendPreference::RequireDevice)
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_validation(J2kEncodeValidation::External);
        encode_j2k_lossy_with_accelerator(samples, &options, BackendKind::Cuda, accelerator)
            .expect("CUDA lossy encode")
            .codestream
    }
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_whole_tile_encode_waits_once_per_stage_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    // Context waits: deinterleave, forward MCT (three components), and per
    // component the plane copy, the DWT levels and the quantization.
    // Event waits: the code-block status and payload readbacks. The
    // packetization status readback blocks without a separate wait.
    let size = 512u32;
    for (lossless, components, max_context_waits) in [
        (true, 1u16, 4u64),
        (true, 3, 11),
        (false, 1, 4),
        (false, 3, 11),
    ] {
        let pixels = if components == 1 {
            j2k_test_support::patterned_gray8(size, size)
        } else {
            j2k_test_support::patterned_rgb8(size, size)
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();
        let warm = encode(lossless, &pixels, size, components, &mut accelerator);
        let context = accelerator
            .cuda_context()
            .expect("CUDA context")
            .expect("CUDA device");
        let before = context.diagnostics().expect("diagnostics before");
        let measured = encode(lossless, &pixels, size, components, &mut accelerator);
        let after = context.diagnostics().expect("diagnostics after");

        let label = format!("lossless={lossless}, {components} components");
        assert_eq!(accelerator.htj2k_tile_dispatches(), 2, "{label}");
        assert!(measured == warm, "{label}: repeated encode differs");
        let context_waits =
            after.context_host_synchronizations - before.context_host_synchronizations;
        let event_waits = after.event_host_synchronizations - before.event_host_synchronizations;
        assert!(
            context_waits <= max_context_waits && event_waits <= 2,
            "{label}: {context_waits} context and {event_waits} event waits per encode"
        );
    }
}
