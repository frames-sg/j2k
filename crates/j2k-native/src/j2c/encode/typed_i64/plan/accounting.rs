// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retained-capacity accounting for typed high-bit plans.

use alloc::vec::Vec;

use super::{TypedI64ExecutionPlan, TypedI64HighBitPlan};
use crate::j2c::encode::allocation::{checked_add_bytes, checked_element_bytes};
use crate::j2c::encode::single_tile::ownership::encode_params_retained_bytes;
use crate::j2c::encode::{EncodeComponentSampleInfo, NativeEncodePipelineResult};

impl TypedI64HighBitPlan {
    pub(in crate::j2c::encode::typed_i64) fn retained_bytes(
        &self,
    ) -> NativeEncodePipelineResult<usize> {
        let mut bytes = checked_element_bytes::<(u16, u16)>(
            self.quant_params.capacity(),
            "typed i64 quantization",
        )?;
        bytes = add_nested_capacity(
            bytes,
            &self.component_step_sizes,
            "typed i64 component step owners",
            "typed i64 component steps",
        )?;
        bytes = checked_add_bytes(
            bytes,
            checked_element_bytes::<EncodeComponentSampleInfo>(
                self.component_sample_info.capacity(),
                "typed i64 component metadata",
            )?,
            "typed i64 plan",
        )?;
        bytes = add_nested_capacity(
            bytes,
            &self.component_quantization_step_sizes,
            "typed i64 component quantization owners",
            "typed i64 component quantization",
        )?;
        checked_add_bytes(
            bytes,
            checked_element_bytes::<(u8, u8)>(
                self.component_sampling.capacity(),
                "typed i64 component sampling",
            )?,
            "typed i64 plan",
        )
        .map_err(Into::into)
    }
}

impl TypedI64ExecutionPlan {
    pub(in crate::j2c::encode::typed_i64) fn retained_bytes(
        &self,
    ) -> NativeEncodePipelineResult<usize> {
        let mut bytes = encode_params_retained_bytes(&self.params)?;
        bytes = checked_add_bytes(
            bytes,
            checked_element_bytes::<(u16, u16)>(
                self.quant_params.capacity(),
                "typed i64 quantization",
            )?,
            "typed i64 execution plan",
        )?;
        add_nested_capacity(
            bytes,
            &self.component_step_sizes,
            "typed i64 component step owners",
            "typed i64 component steps",
        )
    }
}

fn add_nested_capacity<T>(
    mut bytes: usize,
    values: &Vec<Vec<T>>,
    outer_what: &'static str,
    inner_what: &'static str,
) -> NativeEncodePipelineResult<usize> {
    bytes = checked_add_bytes(
        bytes,
        checked_element_bytes::<Vec<T>>(values.capacity(), outer_what)?,
        outer_what,
    )?;
    for value in values {
        bytes = checked_add_bytes(
            bytes,
            checked_element_bytes::<T>(value.capacity(), inner_what)?,
            inner_what,
        )?;
    }
    Ok(bytes)
}
