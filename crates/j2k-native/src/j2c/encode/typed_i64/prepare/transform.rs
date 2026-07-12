// SPDX-License-Identifier: MIT OR Apache-2.0

//! In-place exact-i64 forward transform with retained scratch ownership.

use alloc::vec::Vec;

use super::I64ComponentPrepareRequest;
use crate::j2c::encode::allocation::{
    checked_add_bytes, checked_element_bytes, host_allocation_failed,
};
use crate::j2c::encode::{NativeEncodePipelineError, NativeEncodePipelineResult};
use crate::j2c::fdwt::{try_forward_dwt_packed_i64, PackedDwtGeometry};

pub(super) struct PackedI64Transform {
    pub(super) samples: Vec<i64>,
    pub(super) geometry: PackedDwtGeometry,
    pub(super) line_scratch: Vec<i64>,
    pub(super) retained_source_bytes: usize,
}

pub(super) fn try_transform_component(
    mut samples: Vec<i64>,
    request: &I64ComponentPrepareRequest<'_, '_>,
) -> NativeEncodePipelineResult<PackedI64Transform> {
    let sample_bytes =
        checked_element_bytes::<i64>(samples.capacity(), "exact i64 component samples")?;
    let scratch_count = usize::try_from(request.width.max(request.height)).map_err(|_| {
        NativeEncodePipelineError::arithmetic_overflow("exact i64 DWT scratch length exceeds usize")
    })?;
    let requested_scratch =
        checked_element_bytes::<i64>(scratch_count, "exact i64 DWT line scratch")?;
    let source_base = checked_add_bytes(
        request.retained_base_bytes,
        sample_bytes,
        "exact i64 transform source",
    )?;
    request.session.checked_phase(
        checked_add_bytes(source_base, requested_scratch, "exact i64 DWT line scratch")?,
        "exact i64 DWT line scratch",
    )?;
    let mut line_scratch = Vec::new();
    line_scratch
        .try_reserve_exact(scratch_count)
        .map_err(|_| host_allocation_failed("exact i64 DWT line scratch", requested_scratch))?;
    line_scratch.resize(scratch_count, 0);
    let scratch_bytes =
        checked_element_bytes::<i64>(line_scratch.capacity(), "exact i64 DWT line scratch")?;
    let source_and_scratch = checked_add_bytes(
        source_base,
        scratch_bytes,
        "exact i64 DWT source and scratch",
    )?;
    request
        .session
        .checked_phase(source_and_scratch, "exact i64 DWT source and scratch")?;
    let shape = try_forward_dwt_packed_i64(
        &mut samples,
        request.width,
        request.height,
        request.num_levels,
        &mut line_scratch,
    )?;
    let geometry = PackedDwtGeometry::try_new(request.width, request.height, samples.len(), shape)?;
    if geometry.num_levels() != request.num_levels {
        return Err(NativeEncodePipelineError::internal_invariant(
            "exact i64 DWT level count differs from the marker plan",
        ));
    }
    Ok(PackedI64Transform {
        samples,
        geometry,
        line_scratch,
        retained_source_bytes: source_and_scratch,
    })
}
