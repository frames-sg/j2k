// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use crate::{
    j2c, DecodingError, J2kForwardDwt97Output, J2kIdwtBand, J2kRect, J2kSingleDecompositionIdwtJob,
    J2kWaveletTransform, Result, ValidationError,
};

fn band(values: &[f32], width: u32, height: u32) -> Result<J2kIdwtBand<'_>> {
    let count = u64::from(width) * u64::from(height);
    if count != values.len() as u64 || values.iter().any(|v| !v.is_finite()) {
        return Err(ValidationError::InvalidDimensions.into());
    }
    Ok(J2kIdwtBand {
        rect: J2kRect {
            x0: 0,
            y0: 0,
            x1: width,
            y1: height,
        },
        coefficients: values,
    })
}

/// Reconstruct origin-zero forward 9/7 coefficients without rounding or clipping.
#[doc(hidden)]
pub fn inverse_dwt97_reference(tree: &J2kForwardDwt97Output) -> Result<Vec<f32>> {
    band(&tree.ll, tree.ll_width, tree.ll_height)?;
    if tree.ll_width == 0 || tree.ll_height == 0 {
        return Err(ValidationError::InvalidDimensions.into());
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(tree.ll.len())
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    output.extend_from_slice(&tree.ll);
    let (mut width, mut height) = (tree.ll_width, tree.ll_height);
    for level in &tree.levels {
        if level.width == 0
            || level.height == 0
            || (width, height) != (level.low_width, level.low_height)
            || level.low_width != level.width.div_ceil(2)
            || level.low_height != level.height.div_ceil(2)
            || level.high_width != level.width / 2
            || level.high_height != level.height / 2
        {
            return Err(ValidationError::InvalidDimensions.into());
        }
        let job = J2kSingleDecompositionIdwtJob {
            rect: J2kRect {
                x0: 0,
                y0: 0,
                x1: level.width,
                y1: level.height,
            },
            transform: J2kWaveletTransform::Irreversible97,
            ll: band(&output, width, height)?,
            hl: band(&level.hl, level.high_width, level.low_height)?,
            lh: band(&level.lh, level.low_width, level.high_height)?,
            hh: band(&level.hh, level.high_width, level.high_height)?,
        };
        let mut next = Vec::new();
        j2c::idwt::apply_single_decomposition_idwt_job(job, &mut next)?;
        output = next;
        (width, height) = (level.width, level.height);
    }
    Ok(output)
}

/// Invert centered ICT planes without level restoration, rounding, or clipping.
#[doc(hidden)]
pub fn inverse_ict_reference(mut planes: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>> {
    if planes.len() != 3
        || planes.iter().any(|p| p.len() != planes[0].len())
        || planes.iter().flatten().any(|v| !v.is_finite())
    {
        return Err(ValidationError::InvalidComponentMetadata.into());
    }
    let [y, cb, cr] = planes.as_mut_slice() else {
        unreachable!()
    };
    j2c::mct::apply_inner(
        j2c::codestream::WaveletTransform::Irreversible97,
        y,
        cb,
        cr,
        [0.0; 3],
    );
    Ok(planes)
}
