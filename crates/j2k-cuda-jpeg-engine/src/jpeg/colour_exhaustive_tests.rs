// SPDX-License-Identifier: MIT OR Apache-2.0

//! The encode kernel converts every 8-bit RGB colour, all 2^24, to the YCbCr
//! of the CPU encoder's scalar transform, `j2k_codec_math::jpeg::ycbcr`.
//!
//! The precompute stage codes one 4:4:4 frame per red value, whose 8x8 block
//! `(green, blue)` is flat `(red, green, blue)`. Under quantizers of one a
//! flat block's only coefficient is the exact DC `8 * (sample - 128)`, so the
//! DCs read back are the conversion of every colour.

use j2k_codec_math::jpeg::ycbcr;

use super::encode_staging::checked_staged_encode_plan;
use super::encode_staging_launch::CudaJpegBaselinePrecomputeLaunch;
use crate::bytes::cuda_jpeg_baseline_encode_params_as_bytes;
use crate::{CudaContext, CudaJpegBaselineEncodeParams, JpegCudaEngine};

const SIDE: u32 = 256 * 8;

fn clamp_u8(value: i32) -> i32 {
    value.clamp(0, 255)
}

#[test]
fn precompute_converts_every_colour_as_the_cpu_does() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let context = CudaContext::system_default().expect("CUDA context");
    let engine = JpegCudaEngine::new(&context);
    let params = CudaJpegBaselineEncodeParams {
        input_width: SIDE,
        input_height: SIDE,
        output_width: SIDE,
        output_height: SIDE,
        pitch_bytes: SIDE * 3,
        mcus_per_row: SIDE / 8,
        mcu_rows: SIDE / 8,
        format: 1,
        components: 3,
        max_h: 1,
        max_v: 1,
        h0: 1,
        v0: 1,
        h1: 1,
        v1: 1,
        h2: 1,
        v2: 1,
        entropy_capacity: 1,
        ..CudaJpegBaselineEncodeParams::default()
    };
    let plan = checked_staged_encode_plan(&[params]).expect("staged plan");
    let coefficients = engine
        .allocate(plan.coefficient_bytes)
        .expect("coefficients");
    let params_buffer = engine
        .upload(cuda_jpeg_baseline_encode_params_as_bytes(&[params]))
        .expect("params");
    let quantizers = engine.upload(&[1u8; 64]).expect("quantizers");
    let mut read_back = vec![0u8; plan.coefficient_bytes];
    for red in 0..=255u8 {
        let mut rgb = Vec::with_capacity(SIDE as usize * SIDE as usize * 3);
        for y in 0..SIDE {
            let green = u8::try_from(y / 8).expect("block row fits u8");
            for x in 0..SIDE {
                let blue = u8::try_from(x / 8).expect("block column fits u8");
                rgb.extend([red, green, blue]);
            }
        }
        let input = engine.upload(&rgb).expect("colour frame");
        engine
            .launch_jpeg_encode_baseline_precompute_batch(&CudaJpegBaselinePrecomputeLaunch {
                input_ptr: input.device_ptr(),
                coefficients: &coefficients,
                params: &params_buffer,
                q_luma: &quantizers,
                q_chroma: &quantizers,
                tile_count: 1,
                total_mcus: plan.total_mcus,
                geometry: plan.precompute_geometry,
            })
            .expect("precompute launch");
        coefficients
            .copy_to_host(&mut read_back)
            .expect("read coefficients");
        let blocks = read_back.as_chunks::<{ 3 * 64 * 4 }>().0;
        for (mcu, block) in blocks.iter().enumerate() {
            let green = i32::try_from(mcu / 256).expect("block row fits i32");
            let blue = i32::try_from(mcu % 256).expect("block column fits i32");
            let red = i32::from(red);
            let expected = [
                clamp_u8(ycbcr::rgb_to_y(red, green, blue)),
                clamp_u8(ycbcr::rgb_to_cb(red, green, blue, 128)),
                clamp_u8(ycbcr::rgb_to_cr(red, green, blue, 128)),
            ];
            for (component, sample) in expected.into_iter().enumerate() {
                let offset = component * 64 * 4;
                let dc = i32::from_ne_bytes(
                    block[offset..offset + 4]
                        .try_into()
                        .expect("four coefficient bytes"),
                );
                assert_eq!(
                    dc,
                    8 * (sample - 128),
                    "RGB ({red}, {green}, {blue}) component {component}"
                );
            }
        }
    }
}
