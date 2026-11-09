// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA packet headers and tag trees must reproduce the CPU encoder's bytes on
//! tiles with several precincts per resolution and several code blocks per
//! precinct.

#[cfg(feature = "cuda-runtime")]
use super::{
    encode_with_cuda_test_accelerator, CudaEncodeStageAccelerator, CudaTestEncodeRequest,
    EncodeOptions,
};
#[cfg(feature = "cuda-runtime")]
use j2k_native::EncodeProgressionOrder;

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_packet_headers_match_cpu_across_precincts_and_code_blocks_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let (width, height) = (300u32, 220u32);
    for (components, reversible, progression_order) in [
        (1u8, true, EncodeProgressionOrder::Lrcp),
        (3, true, EncodeProgressionOrder::Rpcl),
        (3, true, EncodeProgressionOrder::Rlcp),
        (1, false, EncodeProgressionOrder::Pcrl),
        (3, false, EncodeProgressionOrder::Cprl),
    ] {
        let pixels: Vec<u8> = (0..width * height * u32::from(components))
            .map(|index| {
                let pixel = index / u32::from(components);
                let (x, y) = (pixel % width, pixel / width);
                // A ramp, texture and flat patches, so some code blocks and
                // precincts code nothing while others hold every bitplane.
                let value = if (x / 40 + y / 40) % 3 == 0 {
                    128
                } else {
                    (x * 3 + y * 2 + (x ^ y).wrapping_mul(2_654_435_761) / (1 << 26)) & 0xff
                };
                u8::try_from(value).expect("masked sample fits u8")
            })
            .collect();
        // 16x16 code blocks; 32x32 precincts at resolution 0 and 64x64 to
        // 128x128 above it give several precincts per resolution and several
        // code blocks per precinct band.
        let options = EncodeOptions {
            reversible,
            use_ht_block_coding: true,
            use_mct: components == 3,
            num_decomposition_levels: 3,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            progression_order,
            precinct_exponents: vec![(5, 5), (6, 6), (6, 6), (7, 7)],
            ..EncodeOptions::default()
        };
        let expected = j2k_native::encode_htj2k(
            &pixels,
            width,
            height,
            u16::from(components),
            8,
            false,
            &options,
        )
        .expect("CPU encode with explicit precincts");
        let mut accelerator = CudaEncodeStageAccelerator::default();
        let actual = encode_with_cuda_test_accelerator(CudaTestEncodeRequest {
            pixels: &pixels,
            width,
            height,
            components,
            bit_depth: 8,
            signed: false,
            options: &options,
            accelerator: &mut accelerator,
        })
        .expect("CUDA encode with explicit precincts");

        let label =
            format!("{components} components, reversible={reversible}, {progression_order:?}");
        assert!(accelerator.ht_code_block_dispatches() > 0, "{label}");
        assert!(
            accelerator.packetization_dispatches() > 0,
            "{label}: CUDA packetization did not run"
        );
        assert!(actual == expected, "{label}: CUDA bytes differ from CPU");
    }
}
