// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    append_color_payload_to_shared, build_cuda_htj2k_color_plans_from_bytes_with_profile,
    host_owners, CudaHtj2kColorDecodePlans, Error, HostPhaseBudget, NativeDecoderContext,
    PixelFormat, CUDA_HTJ2K_KERNELS_NOT_READY,
};

pub(super) fn prepare_color_cuda_resident_batch(
    inputs: &[&[u8]],
    fmt: PixelFormat,
) -> Result<(Vec<CudaHtj2kColorDecodePlans>, Vec<u8>), Error> {
    let mut initial_budget = HostPhaseBudget::new("j2k CUDA color batch plan owners");
    let mut colors = initial_budget.try_vec_with_capacity(inputs.len())?;
    let mut shared_payload = Vec::new();
    let mut native_context = NativeDecoderContext::default();
    for input in inputs {
        let mut color =
            build_cuda_htj2k_color_plans_from_bytes_with_profile(input, fmt, &mut native_context)?;
        if color.components.len() != 3 {
            return Err(Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_KERNELS_NOT_READY,
            });
        }
        let mut append_budget = host_owners::color_batch_budget(
            &colors,
            &shared_payload,
            Some(&color),
            "j2k CUDA color batch plan owners",
        )?;
        append_color_payload_to_shared(&mut color, &mut shared_payload, &mut append_budget)?;
        colors.push(color);
        host_owners::color_batch_budget(
            &colors,
            &shared_payload,
            None,
            "j2k CUDA color batch plan owners",
        )?;
    }
    Ok((colors, shared_payload))
}
