// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{JpegDctComponent, JpegToHtj2kCoefficientPath, JpegToHtj2kError, JpegToHtj2kOptions};

pub(super) fn validate_transcode_options(
    options: &JpegToHtj2kOptions,
) -> Result<(), JpegToHtj2kError> {
    if !options.encode_options.use_ht_block_coding {
        return Err(JpegToHtj2kError::Unsupported(
            "jpeg_to_htj2k requires HT block coding",
        ));
    }
    if options.encode_options.use_mct {
        return Err(JpegToHtj2kError::Unsupported(
            "jpeg_to_htj2k requires use_mct=false because JPEG components stay in native color space",
        ));
    }

    match (options.coefficient_path, options.encode_options.reversible) {
        (
            JpegToHtj2kCoefficientPath::IntegerDirect53
            | JpegToHtj2kCoefficientPath::FloatDirectLinear53,
            true,
        )
        | (JpegToHtj2kCoefficientPath::FloatDirectLinear97, false) => Ok(()),
        (
            JpegToHtj2kCoefficientPath::IntegerDirect53
            | JpegToHtj2kCoefficientPath::FloatDirectLinear53,
            false,
        ) => Err(JpegToHtj2kError::Unsupported(
            "5/3 coefficient path requires reversible HTJ2K encode",
        )),
        (JpegToHtj2kCoefficientPath::FloatDirectLinear97, true) => {
            Err(JpegToHtj2kError::Unsupported(
                "9/7 coefficient path requires irreversible HTJ2K encode",
            ))
        }
    }
}

pub(super) fn validate_component_block_grid(
    component: &JpegDctComponent,
) -> Result<(), JpegToHtj2kError> {
    let block_cols = component.block_cols as usize;
    let block_rows = component.block_rows as usize;
    let expected_blocks =
        block_cols
            .checked_mul(block_rows)
            .ok_or(JpegToHtj2kError::Validation(
                "component block grid overflow",
            ))?;
    if component.dequantized_blocks.len() != expected_blocks {
        return Err(JpegToHtj2kError::Validation(
            "component block count does not match block grid",
        ));
    }

    Ok(())
}

pub(super) fn decomposition_levels_for_components(
    components: &[JpegDctComponent],
    requested_levels: u8,
) -> Result<u8, JpegToHtj2kError> {
    if requested_levels == 0 {
        return Err(JpegToHtj2kError::Unsupported(
            "jpeg_to_htj2k requires at least one decomposition level",
        ));
    }

    let available_levels = components
        .iter()
        .map(|component| available_decomposition_levels(component.width, component.height))
        .min()
        .ok_or(JpegToHtj2kError::Unsupported("missing JPEG components"))?;
    let decomposition_levels = requested_levels.min(available_levels);
    if decomposition_levels == 0 {
        return Err(JpegToHtj2kError::Unsupported(
            "component dimensions are too small for a DWT decomposition",
        ));
    }

    Ok(decomposition_levels)
}

pub(super) fn available_decomposition_levels(width: u32, height: u32) -> u8 {
    let min_dim = width.min(height);
    if min_dim <= 1 {
        0
    } else {
        min_dim.ilog2() as u8
    }
}

pub(super) fn component_sampling_for_jpeg(
    components: &[JpegDctComponent],
    reference_width: u32,
    reference_height: u32,
) -> Result<Vec<(u8, u8)>, JpegToHtj2kError> {
    let max_h = components
        .iter()
        .map(|component| component.h_samp)
        .max()
        .ok_or(JpegToHtj2kError::Unsupported("missing JPEG components"))?;
    let max_v = components
        .iter()
        .map(|component| component.v_samp)
        .max()
        .ok_or(JpegToHtj2kError::Unsupported("missing JPEG components"))?;

    components
        .iter()
        .map(|component| {
            if component.h_samp == 0 || component.v_samp == 0 {
                return Err(JpegToHtj2kError::Unsupported(
                    "JPEG component sampling factors must be non-zero",
                ));
            }
            if max_h % component.h_samp != 0 || max_v % component.v_samp != 0 {
                return Err(JpegToHtj2kError::Unsupported(
                    "fractional JPEG component sampling is not supported",
                ));
            }

            let x_rsiz = max_h / component.h_samp;
            let y_rsiz = max_v / component.v_samp;
            let expected_width = reference_width.div_ceil(u32::from(x_rsiz));
            let expected_height = reference_height.div_ceil(u32::from(y_rsiz));
            if component.width != expected_width || component.height != expected_height {
                return Err(JpegToHtj2kError::Unsupported(
                    "JPEG component dimensions do not match derived SIZ sampling",
                ));
            }

            Ok((x_rsiz, y_rsiz))
        })
        .collect()
}
