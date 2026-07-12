// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_jpeg::{Decoder, Info, SofKind};

use super::{JpegToHtj2kCoefficientPath, JpegToHtj2kError, JpegToHtj2kOptions};
use crate::allocation::{
    checked_add_allocation_bytes, checked_allocation_bytes, ensure_allocation_bytes,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct JpegTranscodeWorkspace {
    peak_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct JpegWorkspaceGeometry {
    total_blocks: usize,
    total_samples: usize,
    max_component_blocks: usize,
    max_component_samples: usize,
}

impl JpegTranscodeWorkspace {
    pub(super) const fn peak_bytes(self) -> usize {
        self.peak_bytes
    }
}

pub(super) fn validate_jpeg_transcode_workspace(
    bytes: &[u8],
    options: &JpegToHtj2kOptions,
) -> Result<JpegTranscodeWorkspace, JpegToHtj2kError> {
    let info = Decoder::inspect(bytes)?;
    workspace_from_info(&info, options)
}

fn workspace_from_info(
    info: &Info,
    options: &JpegToHtj2kOptions,
) -> Result<JpegTranscodeWorkspace, JpegToHtj2kError> {
    let JpegWorkspaceGeometry {
        total_blocks,
        total_samples,
        max_component_blocks,
        max_component_samples,
    } = workspace_geometry(info)?;

    // Transcode extraction retains only the dequantized i16 plane. Progressive
    // extraction also owns an i32 accumulator while assembling that plane.
    let progressive = matches!(
        info.sof_kind,
        SofKind::Progressive8 | SofKind::Progressive12
    );
    let retained_dct_bytes = checked_allocation_bytes::<[i16; 64]>(total_blocks)?;
    let extraction_peak_bytes = if progressive {
        let i32_bytes = checked_allocation_bytes::<[i32; 64]>(total_blocks)?;
        checked_add_allocation_bytes(i32_bytes, retained_dct_bytes)?
    } else {
        retained_dct_bytes
    };

    // Precomputed native-encoder bands retain one f32 coefficient per logical
    // sample. Validation additionally retains actual and expected i32 vectors.
    let precomputed_bytes = checked_allocation_bytes::<f32>(total_samples)?;
    let validation_pair_bytes =
        checked_allocation_bytes::<i32>(total_samples.checked_mul(2).ok_or_else(cap_overflow)?)?;
    let mut persistent_transform_bytes =
        checked_add_allocation_bytes(retained_dct_bytes, precomputed_bytes)?;
    if options.validate_against_float_reference {
        persistent_transform_bytes =
            checked_add_allocation_bytes(persistent_transform_bytes, validation_pair_bytes)?;
    }
    if options.validate_against_integer_reference {
        persistent_transform_bytes =
            checked_add_allocation_bytes(persistent_transform_bytes, validation_pair_bytes)?;
    }

    let integer_cache_bytes = checked_allocation_bytes::<Option<[i32; 64]>>(max_component_blocks)?;
    let integer_plane_bytes = checked_allocation_bytes::<i32>(
        max_component_samples
            .checked_mul(3)
            .ok_or_else(cap_overflow)?,
    )?;
    let float_block_bytes = checked_allocation_bytes::<[[f64; 8]; 8]>(max_component_blocks)?;
    let float_plane_bytes = checked_allocation_bytes::<f64>(
        max_component_samples
            .checked_mul(4)
            .ok_or_else(cap_overflow)?,
    )?;

    let mut transient_bytes = match options.coefficient_path {
        JpegToHtj2kCoefficientPath::IntegerDirect53 => {
            checked_add_allocation_bytes(integer_cache_bytes, integer_plane_bytes)?
        }
        JpegToHtj2kCoefficientPath::FloatDirectLinear53
        | JpegToHtj2kCoefficientPath::FloatDirectLinear97 => {
            checked_add_allocation_bytes(float_block_bytes, float_plane_bytes)?
        }
    };
    if options.validate_against_float_reference
        && matches!(
            options.coefficient_path,
            JpegToHtj2kCoefficientPath::IntegerDirect53
        )
    {
        transient_bytes = checked_add_allocation_bytes(transient_bytes, float_block_bytes)?;
        transient_bytes = checked_add_allocation_bytes(transient_bytes, float_plane_bytes)?;
    }
    if options.validate_against_integer_reference
        && !matches!(
            options.coefficient_path,
            JpegToHtj2kCoefficientPath::IntegerDirect53
        )
    {
        transient_bytes = checked_add_allocation_bytes(transient_bytes, integer_plane_bytes)?;
    }

    let transform_peak_bytes =
        checked_add_allocation_bytes(persistent_transform_bytes, transient_bytes)?;
    let peak_bytes = extraction_peak_bytes.max(transform_peak_bytes);
    ensure_allocation_bytes(peak_bytes)?;
    Ok(JpegTranscodeWorkspace { peak_bytes })
}

fn workspace_geometry(info: &Info) -> Result<JpegWorkspaceGeometry, JpegToHtj2kError> {
    let width = usize::try_from(info.dimensions.0).map_err(|_| {
        JpegToHtj2kError::Validation("JPEG width does not fit the host address space")
    })?;
    let height = usize::try_from(info.dimensions.1).map_err(|_| {
        JpegToHtj2kError::Validation("JPEG height does not fit the host address space")
    })?;
    let max_h = usize::from(info.sampling.max_h);
    let max_v = usize::from(info.sampling.max_v);
    let mcu_cols = usize::try_from(info.mcu_geometry.columns).map_err(|_| {
        JpegToHtj2kError::Validation("JPEG MCU columns do not fit the host address space")
    })?;
    let mcu_rows = usize::try_from(info.mcu_geometry.rows).map_err(|_| {
        JpegToHtj2kError::Validation("JPEG MCU rows do not fit the host address space")
    })?;

    let mut total_blocks = 0usize;
    let mut total_samples = 0usize;
    let mut max_component_blocks = 0usize;
    let mut max_component_samples = 0usize;
    for &(h_samp, v_samp) in info.sampling.components() {
        let h_samp = usize::from(h_samp);
        let v_samp = usize::from(v_samp);
        let block_cols = mcu_cols.checked_mul(h_samp).ok_or_else(cap_overflow)?;
        let block_rows = mcu_rows.checked_mul(v_samp).ok_or_else(cap_overflow)?;
        let block_count = block_cols
            .checked_mul(block_rows)
            .ok_or_else(cap_overflow)?;

        let component_width = width
            .checked_mul(h_samp)
            .ok_or_else(cap_overflow)?
            .div_ceil(max_h);
        let component_height = height
            .checked_mul(v_samp)
            .ok_or_else(cap_overflow)?
            .div_ceil(max_v);
        let sample_count = component_width
            .checked_mul(component_height)
            .ok_or_else(cap_overflow)?;

        total_blocks = total_blocks
            .checked_add(block_count)
            .ok_or_else(cap_overflow)?;
        total_samples = total_samples
            .checked_add(sample_count)
            .ok_or_else(cap_overflow)?;
        max_component_blocks = max_component_blocks.max(block_count);
        max_component_samples = max_component_samples.max(sample_count);
    }

    Ok(JpegWorkspaceGeometry {
        total_blocks,
        total_samples,
        max_component_blocks,
        max_component_samples,
    })
}

fn cap_overflow() -> JpegToHtj2kError {
    JpegToHtj2kError::MemoryCapExceeded {
        requested: usize::MAX,
        cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    }
}

#[cfg(test)]
mod tests {
    use super::validate_jpeg_transcode_workspace;
    use crate::{JpegToHtj2kError, JpegToHtj2kOptions};
    use j2k_jpeg::rewrite_sof_dimensions;
    use j2k_test_support::JPEG_GRAYSCALE_8X8;

    #[test]
    fn huge_sof_geometry_is_rejected_before_dct_extraction() {
        let jpeg = rewrite_sof_dimensions(JPEG_GRAYSCALE_8X8, (65_500, 65_500))
            .expect("fixture contains a valid SOF marker");
        let result = validate_jpeg_transcode_workspace(&jpeg, &JpegToHtj2kOptions::lossless_53());
        assert!(
            matches!(
                &result,
                Err(JpegToHtj2kError::MemoryCapExceeded { requested, cap })
                    if requested > cap
            ),
            "unexpected workspace result: {result:?}"
        );
    }
}
