// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reversible code blocks whose coding passes stop early take the T.800
//! Equation E-8 midpoint (r = 1/2) on the CUDA classic Tier-1 kernel, as on
//! the CPU.

#![cfg(feature = "cuda-runtime")]

use j2k_core::{BackendRequest, DeviceSurface, ImageDecodeDevice, PixelFormat};
use j2k_cuda::{J2kDecoder, SurfaceResidency};
use j2k_test_support::{
    cuda_runtime_and_strict_oxide_gate, REVERSIBLE_EARLY_STOP_LOSSLESS,
    REVERSIBLE_EARLY_STOP_LOSSLESS_U16LE, REVERSIBLE_EARLY_STOP_RATE_LIMITED_RGB8,
    REVERSIBLE_EARLY_STOP_RATE_LIMITED_RGB8_PIXELS, REVERSIBLE_EARLY_STOP_ROI,
    REVERSIBLE_EARLY_STOP_ROI_GRAY8,
};

fn decode_on_cuda(codestream: &[u8], format: PixelFormat) -> Vec<u8> {
    let surface = J2kDecoder::new(codestream)
        .expect("decoder")
        .decode_to_device(format, BackendRequest::Cuda)
        .expect("explicit CUDA decode");
    assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
    let mut downloaded = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut downloaded, surface.pitch_bytes())
        .expect("download CUDA surface");
    downloaded
}

#[test]
fn explicit_cuda_decode_of_early_stopped_reversible_blocks_matches_the_fixtures() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    // The 12-bit rate-limited fixture is checked through the batch decoder,
    // which keeps native samples; see `batch_decoder_api/classic_native.rs`.
    assert_eq!(
        decode_on_cuda(REVERSIBLE_EARLY_STOP_LOSSLESS, PixelFormat::Gray16),
        REVERSIBLE_EARLY_STOP_LOSSLESS_U16LE
    );
    assert_eq!(
        decode_on_cuda(REVERSIBLE_EARLY_STOP_RATE_LIMITED_RGB8, PixelFormat::Rgb8),
        REVERSIBLE_EARLY_STOP_RATE_LIMITED_RGB8_PIXELS
    );
    assert_eq!(
        decode_on_cuda(REVERSIBLE_EARLY_STOP_ROI, PixelFormat::Gray8),
        REVERSIBLE_EARLY_STOP_ROI_GRAY8
    );
}
