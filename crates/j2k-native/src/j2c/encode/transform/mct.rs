// SPDX-License-Identifier: MIT OR Apache-2.0

//! Accelerator handoff for reversible and irreversible color transforms.

use super::super::{J2kEncodeStageAccelerator, J2kForwardIctJob, J2kForwardRctJob, Vec};

pub(in crate::j2c::encode) fn try_encode_forward_rct(
    components: &mut [Vec<f32>],
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> crate::J2kEncodeStageResult<bool> {
    debug_assert!(components.len() >= 3);
    let (plane0, rest) = components.split_at_mut(1);
    let (plane1, plane2) = rest.split_at_mut(1);
    accelerator.encode_forward_rct(J2kForwardRctJob {
        plane0: &mut plane0[0],
        plane1: &mut plane1[0],
        plane2: &mut plane2[0],
    })
}

pub(in crate::j2c::encode) fn try_encode_forward_ict(
    components: &mut [Vec<f32>],
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> crate::J2kEncodeStageResult<bool> {
    debug_assert!(components.len() >= 3);
    let (plane0, rest) = components.split_at_mut(1);
    let (plane1, plane2) = rest.split_at_mut(1);
    accelerator.encode_forward_ict(J2kForwardIctJob {
        plane0: &mut plane0[0],
        plane1: &mut plane1[0],
        plane2: &mut plane2[0],
    })
}
