// SPDX-License-Identifier: MIT OR Apache-2.0

//! Capacity-accounted reuse of decoded component owners across image calls.

use alloc::vec::Vec;
use core::mem::size_of;

use super::{ComponentData, DecoderContext, Header, Tile, TileDecodeContext};
use crate::error::{DecodeError, DecodingError, Result, ValidationError};
use crate::math::{SimdBuffer, SIMD_WIDTH};
use crate::{
    checked_decode_sample_count, try_reserve_decode_elements, try_resize_decode_elements,
    DEFAULT_MAX_DECODE_BYTES,
};

const CONTEXT_ALLOCATION_WHAT: &str = "native decoder context retained components";

#[derive(Clone, Copy)]
pub(super) struct ReusedDecodeBaseline {
    pub(super) parser_live: usize,
    pub(super) channel_capacity: usize,
    pub(super) scratch_capacity: usize,
}

impl DecoderContext<'_> {
    pub(super) fn prepare_reused_decode_baseline(
        &mut self,
        retained_image_bytes: usize,
    ) -> Result<ReusedDecodeBaseline> {
        self.prepare_reused_decode_baseline_with_cap(retained_image_bytes, DEFAULT_MAX_DECODE_BYTES)
    }

    fn prepare_reused_decode_baseline_with_cap(
        &mut self,
        retained_image_bytes: usize,
        cap: usize,
    ) -> Result<ReusedDecodeBaseline> {
        // Parsed graphs borrow the previous input and are never reused.
        // Component, Tier-1, and IDWT owners are lifetime-free and their exact
        // capacities can be carried explicitly across unrelated inputs.
        self.storage.release_all_allocations();

        let retained_channel_bytes = match self.tile_decode_context.retained_channel_bytes() {
            Ok(bytes) => bytes,
            Err(error) if is_capacity_error(&error) => {
                self.tile_decode_context.release_channel_allocations();
                0
            }
            Err(error) => return Err(error),
        };
        let retained_scratch_bytes = self.tile_decode_context.retained_scratch_bytes()?;
        let retained_workspace_bytes =
            checked_combined_context_bytes(retained_channel_bytes, retained_scratch_bytes, cap)?;
        match checked_combined_context_bytes(retained_image_bytes, retained_workspace_bytes, cap) {
            Ok(parser_live) => Ok(ReusedDecodeBaseline {
                parser_live,
                channel_capacity: retained_channel_bytes,
                scratch_capacity: retained_scratch_bytes,
            }),
            Err(error) if retained_scratch_bytes != 0 => Err(error),
            Err(_) if retained_channel_bytes != 0 => {
                // Reuse is optional. A stale cache must never make a decode
                // fail when the same request fits with a fresh context.
                self.tile_decode_context.release_channel_allocations();
                Ok(ReusedDecodeBaseline {
                    parser_live: checked_combined_context_bytes(retained_image_bytes, 0, cap)?,
                    channel_capacity: 0,
                    scratch_capacity: 0,
                })
            }
            Err(error) => Err(error),
        }
    }

    pub(super) fn retry_without_retained_scratch_on_capacity<T>(
        &mut self,
        retained_scratch_bytes: usize,
        first_result: Result<T>,
        retry: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        if retained_scratch_bytes == 0 {
            return first_result;
        }
        match first_result {
            Err(error) if is_capacity_error(&error) => {
                // A retained cache is never allowed to turn a request that
                // fits a fresh decoder into a resource-cap failure.
                self.storage.release_all_allocations();
                self.tile_decode_context.release_tile_scratch_allocations();
                self.record_scratch_capacity_retry();
                retry(self)
            }
            result => result,
        }
    }

    pub(super) fn discard_reused_channels(&mut self) {
        self.tile_decode_context.release_channel_allocations();
    }

    pub(super) fn reset(
        &mut self,
        header: &Header<'_>,
        initial_tile: &Tile<'_>,
        retained_baseline_bytes: usize,
        retained_channel_bytes: usize,
    ) -> Result<usize> {
        let retained_decode_baseline = self.tile_decode_context.reset(
            header,
            initial_tile,
            retained_baseline_bytes,
            retained_channel_bytes,
        )?;
        self.storage.reset_for_next_tile();
        Ok(retained_decode_baseline)
    }
}

impl TileDecodeContext {
    pub(super) fn retained_channel_bytes(&self) -> Result<usize> {
        retained_channel_bytes_with_cap(&self.channel_data, DEFAULT_MAX_DECODE_BYTES)
    }

    pub(super) fn release_channel_allocations(&mut self) {
        self.channel_data = Vec::new();
    }

    fn reset(
        &mut self,
        header: &Header<'_>,
        initial_tile: &Tile<'_>,
        retained_baseline_bytes: usize,
        retained_channel_bytes: usize,
    ) -> Result<usize> {
        self.debug_counters = super::DecodeDebugCounters::default();

        let (output_width, output_height) = self.output_region.map_or(
            (
                header.size_data.image_width(),
                header.size_data.image_height(),
            ),
            super::OutputRegion::dimensions,
        );
        let sample_count = checked_decode_sample_count(output_width, output_height)?;
        let exact_integer_decode = initial_tile
            .component_infos
            .iter()
            .any(super::ComponentInfo::requires_exact_integer_decode);

        let actual_retained = self.retained_channel_bytes()?;
        if actual_retained != retained_channel_bytes {
            return Err(DecodingError::CodeBlockDecodeFailure.into());
        }
        let fresh_baseline = retained_baseline_bytes
            .checked_sub(retained_channel_bytes)
            .ok_or_else(allocation_overflow)?;
        let mut budget = ContextCapacityBudget::from_live_bytes(retained_baseline_bytes)?;
        let reused = reset_channel_data(
            &mut self.channel_data,
            &initial_tile.component_infos,
            sample_count,
            exact_integer_decode,
            &mut budget,
        );
        match reused {
            Ok(()) => Ok(budget.bytes()),
            Err(error) if retained_channel_bytes != 0 && is_capacity_error(&error) => {
                self.release_channel_allocations();
                let mut fresh_budget = ContextCapacityBudget::from_live_bytes(fresh_baseline)?;
                reset_channel_data(
                    &mut self.channel_data,
                    &initial_tile.component_infos,
                    sample_count,
                    exact_integer_decode,
                    &mut fresh_budget,
                )?;
                Ok(fresh_budget.bytes())
            }
            Err(error) => Err(error),
        }
    }
}

fn reset_channel_data(
    components: &mut Vec<ComponentData>,
    component_infos: &[super::ComponentInfo],
    sample_count: usize,
    exact_integer_decode: bool,
    budget: &mut ContextCapacityBudget,
) -> Result<()> {
    let component_count = component_infos.len();
    if components.capacity() < component_count {
        let released = retained_channel_bytes_with_cap(components, DEFAULT_MAX_DECODE_BYTES)?;
        *components = Vec::new();
        budget.release_bytes(released)?;
        budget.include_elements::<ComponentData>(component_count)?;
        try_reserve_decode_elements(components, component_count)?;
        if let Err(error) =
            budget.include_capacity_overage::<ComponentData>(component_count, components.capacity())
        {
            *components = Vec::new();
            return Err(error);
        }
    } else if components.len() > component_count {
        for component in &components[component_count..] {
            budget.release_bytes(component_nested_bytes(component)?)?;
        }
        components.truncate(component_count);
    }

    while components.len() < component_count {
        let info = &component_infos[components.len()];
        components.push(ComponentData {
            container: SimdBuffer::empty(),
            integer_container: None,
            bit_depth: info.size_info.precision,
            signed: info.size_info.signed,
        });
    }

    if !exact_integer_decode {
        // Release every now-unneeded exact sidecar before growing any SIMD
        // owner. A high-bit-depth image followed by a larger low-bit-depth
        // image should not discard otherwise reusable component allocations
        // merely because soon-to-be-released i64 capacity occupied the cap.
        for component in components.iter_mut() {
            if let Some(values) = component.integer_container.take() {
                budget.release_elements::<i64>(values.capacity())?;
            }
        }
    }

    for (component, info) in components.iter_mut().zip(component_infos) {
        reset_simd_samples(component, sample_count, budget)?;
        if exact_integer_decode {
            reset_integer_samples(component, sample_count, budget)?;
        }
        component.bit_depth = info.size_info.precision;
        component.signed = info.size_info.signed;
    }
    Ok(())
}

fn reset_simd_samples(
    component: &mut ComponentData,
    sample_count: usize,
    budget: &mut ContextCapacityBudget,
) -> Result<()> {
    let planned_capacity =
        SimdBuffer::<SIMD_WIDTH>::padded_len(sample_count).ok_or(ValidationError::ImageTooLarge)?;
    if component.container.capacity() >= planned_capacity {
        component
            .container
            .try_reset_zeros(sample_count)
            .map_err(|_| DecodingError::HostAllocationFailed)?;
        return Ok(());
    }

    let released_capacity = component.container.capacity();
    component.container = SimdBuffer::empty();
    budget.release_elements::<f32>(released_capacity)?;
    budget.include_elements::<f32>(planned_capacity)?;
    let prepared =
        SimdBuffer::try_zeros(sample_count).map_err(|_| DecodingError::HostAllocationFailed)?;
    budget.include_capacity_overage::<f32>(planned_capacity, prepared.capacity())?;
    component.container = prepared;
    Ok(())
}

fn reset_integer_samples(
    component: &mut ComponentData,
    sample_count: usize,
    budget: &mut ContextCapacityBudget,
) -> Result<()> {
    if let Some(values) = component.integer_container.as_mut() {
        if values.capacity() >= sample_count {
            try_resize_decode_elements(values, sample_count, 0_i64)?;
            values.fill(0);
            return Ok(());
        }
    }

    if let Some(values) = component.integer_container.take() {
        budget.release_elements::<i64>(values.capacity())?;
    }
    budget.include_elements::<i64>(sample_count)?;
    let mut values = Vec::new();
    try_resize_decode_elements(&mut values, sample_count, 0_i64)?;
    budget.include_capacity_overage::<i64>(sample_count, values.capacity())?;
    component.integer_container = Some(values);
    Ok(())
}

fn retained_channel_bytes_with_cap(components: &Vec<ComponentData>, cap: usize) -> Result<usize> {
    let mut budget = ContextCapacityBudget::with_cap(0, cap)?;
    budget.include_elements::<ComponentData>(components.capacity())?;
    for component in components {
        budget.include_bytes(component_nested_bytes(component)?)?;
    }
    Ok(budget.bytes())
}

fn component_nested_bytes(component: &ComponentData) -> Result<usize> {
    let float_bytes = component
        .container
        .capacity()
        .checked_mul(size_of::<f32>())
        .ok_or_else(allocation_overflow)?;
    let integer_bytes = component
        .integer_container
        .as_ref()
        .map_or(Ok(0), |values| {
            values
                .capacity()
                .checked_mul(size_of::<i64>())
                .ok_or_else(allocation_overflow)
        })?;
    float_bytes
        .checked_add(integer_bytes)
        .ok_or_else(allocation_overflow)
}

fn checked_combined_context_bytes(left: usize, right: usize, cap: usize) -> Result<usize> {
    let requested = left.checked_add(right).ok_or_else(allocation_overflow)?;
    if requested > cap {
        return Err(DecodeError::AllocationTooLarge {
            what: CONTEXT_ALLOCATION_WHAT,
            requested,
            cap,
        });
    }
    Ok(requested)
}

pub(super) fn is_capacity_error(error: &DecodeError) -> bool {
    matches!(
        error,
        DecodeError::AllocationTooLarge { .. }
            | DecodeError::Validation(ValidationError::ImageTooLarge)
    )
}

fn allocation_overflow() -> DecodeError {
    DecodeError::AllocationTooLarge {
        what: CONTEXT_ALLOCATION_WHAT,
        requested: usize::MAX,
        cap: DEFAULT_MAX_DECODE_BYTES,
    }
}

struct ContextCapacityBudget {
    bytes: usize,
    cap: usize,
}

impl ContextCapacityBudget {
    fn from_live_bytes(bytes: usize) -> Result<Self> {
        Self::with_cap(bytes, DEFAULT_MAX_DECODE_BYTES)
    }

    const fn with_cap(bytes: usize, cap: usize) -> Result<Self> {
        if bytes > cap {
            return Err(DecodeError::AllocationTooLarge {
                what: CONTEXT_ALLOCATION_WHAT,
                requested: bytes,
                cap,
            });
        }
        Ok(Self { bytes, cap })
    }

    fn include_elements<T>(&mut self, count: usize) -> Result<()> {
        let bytes = count
            .checked_mul(size_of::<T>())
            .ok_or_else(allocation_overflow)?;
        self.include_bytes(bytes)
    }

    fn include_capacity_overage<T>(
        &mut self,
        planned_count: usize,
        actual_capacity: usize,
    ) -> Result<()> {
        if actual_capacity > planned_count {
            self.include_elements::<T>(actual_capacity - planned_count)?;
        }
        Ok(())
    }

    fn include_bytes(&mut self, additional: usize) -> Result<()> {
        let requested = self
            .bytes
            .checked_add(additional)
            .ok_or_else(allocation_overflow)?;
        if requested > self.cap {
            return Err(DecodeError::AllocationTooLarge {
                what: CONTEXT_ALLOCATION_WHAT,
                requested,
                cap: self.cap,
            });
        }
        self.bytes = requested;
        Ok(())
    }

    fn release_elements<T>(&mut self, count: usize) -> Result<()> {
        let bytes = count
            .checked_mul(size_of::<T>())
            .ok_or_else(allocation_overflow)?;
        self.release_bytes(bytes)
    }

    fn release_bytes(&mut self, released: usize) -> Result<()> {
        self.bytes = self
            .bytes
            .checked_sub(released)
            .ok_or_else(allocation_overflow)?;
        Ok(())
    }

    const fn bytes(&self) -> usize {
        self.bytes
    }
}

#[cfg(test)]
mod ownership_tests;

#[cfg(test)]
mod tests {
    use super::{
        checked_combined_context_bytes, reset_integer_samples, reset_simd_samples,
        ContextCapacityBudget,
    };
    use crate::error::DecodeError;
    use crate::j2c::ComponentData;
    use crate::math::{SimdBuffer, SIMD_WIDTH};
    use alloc::{vec, vec::Vec};
    use core::mem::size_of;

    #[test]
    fn retained_context_baseline_accepts_exact_cap_and_rejects_one_over() {
        assert_eq!(
            checked_combined_context_bytes(5, 3, 8).expect("exact context boundary"),
            8
        );
        assert!(matches!(
            checked_combined_context_bytes(5, 4, 8),
            Err(DecodeError::AllocationTooLarge {
                requested: 9,
                cap: 8,
                ..
            })
        ));
    }

    #[test]
    fn retained_scratch_exact_cap_and_scratch_free_retry_are_deterministic() {
        let mut context = super::DecoderContext::default();
        context
            .tile_decode_context
            .idwt_scratch_buffer
            .try_reserve_exact(8)
            .expect("reserve deterministic retained scratch");
        let scratch_bytes = context
            .tile_decode_context
            .retained_scratch_bytes()
            .expect("scratch capacity bytes");
        assert!(scratch_bytes > 0);

        let exact = context
            .prepare_reused_decode_baseline_with_cap(0, scratch_bytes)
            .expect("retained scratch fits exact cap");
        assert_eq!(exact.parser_live, scratch_bytes);
        assert_eq!(exact.scratch_capacity, scratch_bytes);
        let first_result = context.prepare_reused_decode_baseline_with_cap(0, scratch_bytes - 1);
        assert!(matches!(
            first_result,
            Err(DecodeError::AllocationTooLarge { .. })
        ));

        let retried = context
            .retry_without_retained_scratch_on_capacity(scratch_bytes, first_result, |context| {
                context.prepare_reused_decode_baseline_with_cap(0, 0)
            })
            .expect("scratch-free retry fits");
        assert_eq!(retried.parser_live, 0);
        assert_eq!(retried.scratch_capacity, 0);
        assert_eq!(
            context.tile_decode_context.idwt_capacity_bytes().unwrap(),
            0
        );
        assert_eq!(context.workspace_stats().scratch_capacity_retries(), 1);
    }

    #[test]
    fn context_budget_release_replaces_old_owner_without_double_counting() {
        let mut budget = ContextCapacityBudget::with_cap(8, 8).expect("full old owner");
        budget.release_bytes(5).expect("release old capacity");
        budget.include_bytes(5).expect("replacement fits exact cap");
        assert_eq!(budget.bytes(), 8);
        assert!(matches!(
            budget.include_bytes(1),
            Err(DecodeError::AllocationTooLarge {
                requested: 9,
                cap: 8,
                ..
            })
        ));
    }

    #[test]
    fn reused_sample_owners_keep_addresses_and_clear_stale_values() {
        let mut integers = Vec::with_capacity(24);
        integers.resize(17, 19_i64);
        let mut component = ComponentData {
            container: SimdBuffer::<SIMD_WIDTH>::new(vec![7.0; 17]),
            integer_container: Some(integers),
            bit_depth: 29,
            signed: false,
        };
        let sample_ptr = component.container.as_ptr();
        let sample_capacity = component.container.capacity();
        let (integer_ptr, integer_capacity) = component
            .integer_container
            .as_ref()
            .map(|values| (values.as_ptr(), values.capacity()))
            .expect("integer owner");
        let owner_bytes = sample_capacity * size_of::<f32>() + integer_capacity * size_of::<i64>();
        let mut budget = ContextCapacityBudget::with_cap(owner_bytes, owner_bytes)
            .expect("existing owners fit exact test cap");

        reset_simd_samples(&mut component, 3, &mut budget).expect("reuse SIMD owner");
        reset_integer_samples(&mut component, 3, &mut budget).expect("reuse integer owner");

        assert_eq!(component.container.as_ptr(), sample_ptr);
        assert_eq!(component.container.capacity(), sample_capacity);
        assert_eq!(component.container.truncated(), [0.0; 3]);
        let integers = component.integer_container.as_ref().expect("integer owner");
        assert_eq!(integers.as_ptr(), integer_ptr);
        assert_eq!(integers.capacity(), integer_capacity);
        assert_eq!(integers, &[0_i64; 3]);
        assert_eq!(budget.bytes(), owner_bytes);

        component.container.fill(23.0);
        component
            .integer_container
            .as_mut()
            .expect("integer owner")
            .fill(29);
        reset_simd_samples(&mut component, 17, &mut budget).expect("regrow SIMD owner");
        reset_integer_samples(&mut component, 17, &mut budget).expect("regrow integer owner");

        assert_eq!(component.container.as_ptr(), sample_ptr);
        assert!(component.container.iter().all(|sample| *sample == 0.0));
        let integers = component.integer_container.as_ref().expect("integer owner");
        assert_eq!(integers.as_ptr(), integer_ptr);
        assert_eq!(integers, &[0_i64; 17]);
        assert_eq!(budget.bytes(), owner_bytes);
    }
}
