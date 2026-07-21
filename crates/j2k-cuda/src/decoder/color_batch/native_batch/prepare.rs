// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    append_color_payload_to_shared, build_cuda_classic_color_plans_from_referenced_with_profile,
    build_cuda_color_plan_from_bytes_for_device_plan_with_profile,
    build_cuda_htj2k_color_plans_from_referenced_with_profile, host_owners,
    CudaHtj2kColorDecodePlans, Error, HostPhaseBudget, NativeColorBatchInput, NativeDecoderContext,
    PixelFormat, CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
};

pub(super) struct PreparedNativeColorBatch {
    pub(super) colors: Vec<CudaHtj2kColorDecodePlans>,
    pub(super) shared_payload: Vec<u8>,
    pub(super) source_indices: Vec<usize>,
}

#[expect(
    clippy::too_many_lines,
    reason = "one preparation boundary keeps tile plans, shared payload ownership, and source identities aligned"
)]
pub(super) fn prepare_native_color_batch(
    inputs: &[NativeColorBatchInput<'_>],
    fmt: PixelFormat,
) -> Result<PreparedNativeColorBatch, Error> {
    if !matches!(
        fmt,
        PixelFormat::Rgb8
            | PixelFormat::Rgb16
            | PixelFormat::RgbI16
            | PixelFormat::Rgba8
            | PixelFormat::Rgba16
            | PixelFormat::RgbaI16
    ) {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        });
    }
    let mut initial_budget = HostPhaseBudget::new("j2k CUDA exact RGB batch plan owners");
    let mut colors = initial_budget.try_vec_with_capacity(inputs.len())?;
    let mut source_indices = initial_budget.try_vec_with_capacity(inputs.len())?;
    let mut shared_payload = Vec::new();
    let mut native_context = NativeDecoderContext::default();
    for (output_index, input) in inputs.iter().enumerate() {
        let (mut input_colors, payload_is_shared) =
            match (input.referenced_plan, input.referenced_classic_plan) {
                (Some(referenced_plan), None) => {
                    let mut budget = host_owners::color_batch_budget(
                        &colors,
                        &shared_payload,
                        None,
                        "j2k CUDA referenced exact RGB batch plan owners",
                    )?;
                    (
                        build_cuda_htj2k_color_plans_from_referenced_with_profile(
                            input.bytes,
                            referenced_plan,
                            fmt,
                            input.device_plan,
                            &mut shared_payload,
                            &mut budget,
                        )?,
                        true,
                    )
                }
                (None, Some(referenced_plan)) => {
                    let mut budget = host_owners::color_batch_budget(
                        &colors,
                        &shared_payload,
                        None,
                        "j2k CUDA referenced classic color batch plan owners",
                    )?;
                    (
                        build_cuda_classic_color_plans_from_referenced_with_profile(
                            input.bytes,
                            referenced_plan,
                            fmt,
                            input.device_plan,
                            &mut shared_payload,
                            &mut budget,
                        )?,
                        true,
                    )
                }
                (None, None) => {
                    let mut color = build_cuda_color_plan_from_bytes_for_device_plan_with_profile(
                        input.bytes,
                        fmt,
                        input.device_plan,
                        input.settings,
                        &mut native_context,
                    )?;
                    let mut budget = host_owners::color_batch_budget(
                        &colors,
                        &shared_payload,
                        Some(&color),
                        "j2k CUDA exact classic RGB batch plan owners",
                    )?;
                    append_color_payload_to_shared(&mut color, &mut shared_payload, &mut budget)?;
                    let mut one = budget.try_vec_with_capacity(1)?;
                    one.push(color);
                    (one, true)
                }
                (Some(_), Some(_)) => {
                    return Err(Error::UnsupportedCudaRequest {
                        reason: "prepared CUDA color input contains conflicting codec plans",
                    });
                }
            };
        if !payload_is_shared {
            return Err(Error::UnsupportedCudaRequest {
                reason: "CUDA exact color plan payload ownership is inconsistent",
            });
        }
        let mut append_budget = host_owners::color_batch_budget(
            &colors,
            &shared_payload,
            None,
            "j2k CUDA exact color tile owner append",
        )?;
        append_budget.account_vec(&source_indices)?;
        append_budget.try_vec_reserve(&mut colors, input_colors.len())?;
        append_budget.try_vec_reserve(&mut source_indices, input_colors.len())?;
        for color in &mut input_colors {
            color.output_index = output_index;
        }
        source_indices.extend(std::iter::repeat_n(input.source_index, input_colors.len()));
        colors.append(&mut input_colors);
    }
    Ok(PreparedNativeColorBatch {
        colors,
        shared_payload,
        source_indices,
    })
}
