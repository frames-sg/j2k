// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA encodes of high-precision unsigned samples whose container padding
//! bits are set must produce the CPU encoder's bytes: both ignore the bits
//! above the declared precision.

#[cfg(feature = "cuda-runtime")]
use super::{
    encode_j2k_lossy_with_accelerator, BackendKind, CudaEncodeStageAccelerator,
    EncodeBackendPreference, J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions,
    J2kLosslessSamples, J2kLossyEncodeOptions, J2kLossySamples,
};

/// Wider than one 64x64 code block in each direction, with ragged edges.
#[cfg(feature = "cuda-runtime")]
const WIDTH: u32 = 197;
#[cfg(feature = "cuda-runtime")]
const HEIGHT: u32 = 141;

/// Little-endian u16 samples of `bit_depth` bits: a smooth ramp plus texture,
/// with the padding bits above the precision set, wholly or partly, in most
/// words.
#[cfg(feature = "cuda-runtime")]
fn padded_samples(components: u16, bit_depth: u8) -> Vec<u8> {
    let mask = (1u32 << bit_depth) - 1;
    let padding = 0xffff & !mask;
    let mut bytes = Vec::new();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            for component in 0..u32::from(components) {
                let ramp = (x * 7 + y * 5 + component * 311) << (bit_depth - 8);
                let texture = (x ^ y).wrapping_mul(2_654_435_761 + component) >> 21;
                let value = (ramp + texture) & mask;
                let high = match (x + 3 * y + component) % 4 {
                    0 => padding,
                    1 => 0xf800 & padding,
                    2 => 0,
                    _ => padding & (padding << 1),
                };
                let word = u16::try_from(value | high).expect("sample fits u16");
                bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
    }
    bytes
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_lossless_ignores_unsigned_padding_bits_like_cpu_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let options = J2kLosslessEncodeOptions::default()
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_validation(J2kEncodeValidation::External);
    for bit_depth in [10u8, 12, 14] {
        for components in [1u16, 3] {
            let pixels = padded_samples(components, bit_depth);
            let samples = || {
                J2kLosslessSamples::new(&pixels, WIDTH, HEIGHT, components, bit_depth, false)
                    .expect("valid padded samples")
            };
            let cpu = j2k::encode_j2k_lossless(samples(), &options.with_cpu_only_backend())
                .expect("CPU lossless encode");
            let mut accelerator = CudaEncodeStageAccelerator::default();
            let cuda = j2k::encode_j2k_lossless_with_accelerator(
                samples(),
                &options.with_backend(EncodeBackendPreference::RequireDevice),
                BackendKind::Cuda,
                &mut accelerator,
            )
            .expect("CUDA lossless encode");

            let label = format!("{bit_depth}-bit, {components} components");
            assert_eq!(cuda.backend, BackendKind::Cuda, "{label}");
            assert_eq!(accelerator.htj2k_tile_dispatches(), 1, "{label}");
            assert!(accelerator.ht_code_block_attempts() > 1, "{label}");
            assert!(
                cuda.codestream == cpu.codestream,
                "{label}: CUDA bytes differ from CPU"
            );
        }
    }
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_lossy_ignores_unsigned_padding_bits_like_cpu_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let options = J2kLossyEncodeOptions::default()
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_validation(J2kEncodeValidation::External);
    for bit_depth in [10u8, 12, 14] {
        for components in [1u16, 3] {
            let pixels = padded_samples(components, bit_depth);
            let samples = || {
                J2kLossySamples::new(&pixels, WIDTH, HEIGHT, components, bit_depth, false)
                    .expect("valid padded samples")
            };
            let cpu = j2k::encode_j2k_lossy(samples(), &options.clone().with_cpu_only_backend())
                .expect("CPU lossy encode");
            let mut accelerator = CudaEncodeStageAccelerator::default();
            let cuda = encode_j2k_lossy_with_accelerator(
                samples(),
                &options
                    .clone()
                    .with_backend(EncodeBackendPreference::RequireDevice),
                BackendKind::Cuda,
                &mut accelerator,
            )
            .expect("CUDA lossy encode");

            let label = format!("{bit_depth}-bit, {components} components");
            assert_eq!(cuda.backend, BackendKind::Cuda, "{label}");
            assert_eq!(accelerator.htj2k_tile_dispatches(), 1, "{label}");
            assert!(accelerator.ht_code_block_attempts() > 1, "{label}");
            assert!(
                cuda.codestream == cpu.codestream,
                "{label}: CUDA bytes differ from CPU"
            );
        }
    }
}
