// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    append_color_payload_to_shared, build_cuda_classic_color_plans_from_referenced_with_profile,
    build_cuda_color_plan_from_bytes_for_device_plan_with_profile,
    build_cuda_htj2k_color_plans_from_referenced_with_profile, host_owners,
    CudaHtj2kColorDecodePlans, Error, NativeColorBatchInput, NativeDecoderContext, PixelFormat,
};

pub(super) fn prepare_native_color_input<'a>(
    input: &NativeColorBatchInput<'a>,
    fmt: PixelFormat,
    native_context: &mut NativeDecoderContext<'a>,
    colors: &Vec<CudaHtj2kColorDecodePlans>,
    shared_payload: &mut Vec<u8>,
) -> Result<Vec<CudaHtj2kColorDecodePlans>, Error> {
    match (input.referenced_plan, input.referenced_classic_plan) {
        (Some(referenced_plan), None) => {
            let mut budget = host_owners::color_batch_budget(
                colors,
                shared_payload,
                None,
                "j2k CUDA referenced exact RGB batch plan owners",
            )?;
            build_cuda_htj2k_color_plans_from_referenced_with_profile(
                input.bytes,
                referenced_plan,
                fmt,
                input.device_plan,
                shared_payload,
                &mut budget,
            )
        }
        (None, Some(referenced_plan)) => {
            let mut budget = host_owners::color_batch_budget(
                colors,
                shared_payload,
                None,
                "j2k CUDA referenced classic color batch plan owners",
            )?;
            build_cuda_classic_color_plans_from_referenced_with_profile(
                input.bytes,
                referenced_plan,
                fmt,
                input.device_plan,
                shared_payload,
                &mut budget,
            )
        }
        (None, None) => {
            prepare_native_color_from_bytes(input, fmt, native_context, colors, shared_payload)
        }
        (Some(_), Some(_)) => Err(Error::UnsupportedCudaRequest {
            reason: "prepared CUDA color input contains conflicting codec plans",
        }),
    }
}

fn prepare_native_color_from_bytes<'a>(
    input: &NativeColorBatchInput<'a>,
    fmt: PixelFormat,
    native_context: &mut NativeDecoderContext<'a>,
    colors: &Vec<CudaHtj2kColorDecodePlans>,
    shared_payload: &mut Vec<u8>,
) -> Result<Vec<CudaHtj2kColorDecodePlans>, Error> {
    let mut color = build_cuda_color_plan_from_bytes_for_device_plan_with_profile(
        input.bytes,
        fmt,
        input.device_plan,
        input.settings,
        native_context,
    )?;
    let mut budget = host_owners::color_batch_budget(
        colors,
        shared_payload,
        Some(&color),
        "j2k CUDA exact classic RGB batch plan owners",
    )?;
    append_color_payload_to_shared(&mut color, shared_payload, &mut budget)?;
    let mut one = budget.try_vec_with_capacity(1)?;
    one.push(color);
    Ok(one)
}
