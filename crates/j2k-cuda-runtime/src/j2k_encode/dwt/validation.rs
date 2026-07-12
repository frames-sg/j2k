// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_codec_math::dwt::max_decomposition_levels;

use crate::{error::CudaError, kernels::j2k_dwt53_launch_geometry};

pub(super) const FORWARD_DWT_LEVELS_EXCEED_GEOMETRY: &str =
    "forward DWT decomposition levels exceed image geometry";
pub(super) const FORWARD_DWT_SAMPLES_EXCEED_INDEX_ABI: &str =
    "forward DWT sample count exceeds the CUDA u32 indexing ABI";
pub(super) const FORWARD_DWT_GEOMETRY_EXCEEDS_LAUNCH_LIMITS: &str =
    "forward DWT geometry exceeds static CUDA launch limits";

pub(super) fn validate_forward_dwt_request(
    width: u32,
    height: u32,
    num_levels: u8,
) -> Result<(), CudaError> {
    if num_levels != 0 {
        let samples = u64::from(width) * u64::from(height);
        if samples.saturating_sub(1) > u64::from(u32::MAX) {
            return Err(CudaError::InvalidArgument {
                message: format!(
                    "{FORWARD_DWT_SAMPLES_EXCEED_INDEX_ABI}: {samples} samples for {width}x{height}"
                ),
            });
        }
    }

    let max_levels = max_decomposition_levels(width, height);
    if num_levels > max_levels {
        return Err(CudaError::InvalidArgument {
            message: format!(
                "{FORWARD_DWT_LEVELS_EXCEED_GEOMETRY}: requested {num_levels}, maximum {max_levels} for {width}x{height}"
            ),
        });
    }
    if num_levels != 0 && j2k_dwt53_launch_geometry(width, height).is_none() {
        return Err(CudaError::InvalidArgument {
            message: format!("{FORWARD_DWT_GEOMETRY_EXCEEDS_LAUNCH_LIMITS}: {width}x{height}"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
