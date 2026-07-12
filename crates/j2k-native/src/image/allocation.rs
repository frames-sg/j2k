// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retained allocation accounting for the parsed image owner.

use core::mem::size_of;

use crate::color::ColorSpace;
use crate::error::{DecodeError, Result};
use crate::j2c::codestream::allocation::retained_header_bytes;
use crate::j2c::{ComponentData, Header};
use crate::jp2::ImageBoxes;
use crate::DEFAULT_MAX_DECODE_BYTES;

/// Shared checked accounting for simultaneously live decode-side owners.
#[derive(Default)]
pub(crate) struct DecodeOwnerBudget {
    bytes: usize,
}

impl DecodeOwnerBudget {
    pub(crate) fn from_retained_bytes(retained_bytes: usize) -> Result<Self> {
        let mut budget = Self::default();
        budget.include_bytes(retained_bytes)?;
        Ok(budget)
    }

    pub(crate) fn include_elements<T>(&mut self, count: usize) -> Result<()> {
        let additional =
            count
                .checked_mul(size_of::<T>())
                .ok_or(DecodeError::AllocationTooLarge {
                    what: "native decode retained allocations",
                    requested: usize::MAX,
                    cap: DEFAULT_MAX_DECODE_BYTES,
                })?;
        self.include_bytes(additional)
    }

    pub(crate) fn for_components(
        retained_bytes: usize,
        components: &[ComponentData],
        component_owner_capacity: usize,
    ) -> Result<Self> {
        let mut budget = Self::from_retained_bytes(retained_bytes)?;
        budget.include_components(components, component_owner_capacity)?;
        Ok(budget)
    }

    pub(crate) fn include_components(
        &mut self,
        components: &[ComponentData],
        component_owner_capacity: usize,
    ) -> Result<()> {
        self.include_elements::<ComponentData>(component_owner_capacity)?;
        for component in components {
            self.include_elements::<f32>(component.container.capacity())?;
            if let Some(integers) = &component.integer_container {
                self.include_elements::<i64>(integers.capacity())?;
            }
        }
        Ok(())
    }

    pub(crate) fn include_capacity_overage<T>(
        &mut self,
        planned_count: usize,
        actual_capacity: usize,
    ) -> Result<()> {
        if actual_capacity > planned_count {
            self.include_elements::<T>(actual_capacity - planned_count)?;
        }
        Ok(())
    }

    pub(crate) const fn bytes(&self) -> usize {
        self.bytes
    }

    fn include_bytes(&mut self, additional: usize) -> Result<()> {
        let updated =
            self.bytes
                .checked_add(additional)
                .ok_or(DecodeError::AllocationTooLarge {
                    what: "native decode retained allocations",
                    requested: usize::MAX,
                    cap: DEFAULT_MAX_DECODE_BYTES,
                })?;
        if updated > DEFAULT_MAX_DECODE_BYTES {
            return Err(DecodeError::AllocationTooLarge {
                what: "native decode retained allocations",
                requested: updated,
                cap: DEFAULT_MAX_DECODE_BYTES,
            });
        }
        self.bytes = updated;
        Ok(())
    }
}

pub(super) fn retained_metadata_bytes(
    header: &Header<'_>,
    boxes: &ImageBoxes,
    color_space: &ColorSpace,
) -> Result<usize> {
    let mut bytes = retained_container_metadata_bytes(header, boxes)?;
    if let ColorSpace::Icc { profile, .. } = color_space {
        include_retained_bytes(&mut bytes, profile.capacity())?;
    }
    Ok(bytes)
}

pub(crate) fn retained_container_metadata_bytes(
    header: &Header<'_>,
    boxes: &ImageBoxes,
) -> Result<usize> {
    let mut bytes = retained_header_bytes(header)?;
    include_retained_bytes(&mut bytes, boxes.allocated_bytes()?)?;
    Ok(bytes)
}

fn include_retained_bytes(total: &mut usize, additional: usize) -> Result<()> {
    let mut budget = DecodeOwnerBudget::from_retained_bytes(*total)?;
    budget.include_bytes(additional)?;
    *total = budget.bytes();
    Ok(())
}

pub(super) fn combine_retained_bytes(left: usize, right: usize) -> Result<usize> {
    let mut total = left;
    include_retained_bytes(&mut total, right)?;
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::{combine_retained_bytes, include_retained_bytes, DecodeOwnerBudget};
    use crate::error::DecodeError;
    use crate::j2c::ComponentData;
    use crate::math::{SimdBuffer, SIMD_WIDTH};
    use crate::DEFAULT_MAX_DECODE_BYTES;
    use alloc::{vec, vec::Vec};
    use core::mem::size_of;

    #[test]
    fn retained_image_boundary_accepts_exact_cap_and_rejects_one_over() {
        let mut exact = DEFAULT_MAX_DECODE_BYTES - 1;
        include_retained_bytes(&mut exact, 1).expect("exact retained metadata cap");
        assert_eq!(exact, DEFAULT_MAX_DECODE_BYTES);

        let mut over = DEFAULT_MAX_DECODE_BYTES;
        assert_eq!(
            include_retained_bytes(&mut over, 1),
            Err(DecodeError::AllocationTooLarge {
                what: "native decode retained allocations",
                requested: DEFAULT_MAX_DECODE_BYTES + 1,
                cap: DEFAULT_MAX_DECODE_BYTES,
            })
        );
    }

    #[test]
    fn paired_retained_boundary_accepts_exact_cap_and_rejects_one_over() {
        assert_eq!(
            combine_retained_bytes(DEFAULT_MAX_DECODE_BYTES - 7, 7)
                .expect("paired owners fit the exact decode cap"),
            DEFAULT_MAX_DECODE_BYTES
        );
        assert_eq!(
            combine_retained_bytes(DEFAULT_MAX_DECODE_BYTES - 7, 8),
            Err(DecodeError::AllocationTooLarge {
                what: "native decode retained allocations",
                requested: DEFAULT_MAX_DECODE_BYTES + 1,
                cap: DEFAULT_MAX_DECODE_BYTES,
            })
        );
    }

    #[test]
    fn shared_decode_budget_accepts_exact_capacity_and_rejects_one_over() {
        let mut exact = DecodeOwnerBudget::from_retained_bytes(DEFAULT_MAX_DECODE_BYTES - 1)
            .expect("one byte of decode headroom");
        exact
            .include_elements::<u8>(1)
            .expect("exact aggregate decode cap");

        assert_eq!(
            exact.include_elements::<u8>(1),
            Err(DecodeError::AllocationTooLarge {
                what: "native decode retained allocations",
                requested: DEFAULT_MAX_DECODE_BYTES + 1,
                cap: DEFAULT_MAX_DECODE_BYTES,
            })
        );
    }

    #[test]
    fn retained_decode_budget_reports_arithmetic_overflow_as_typed_resource_failure() {
        assert_eq!(
            DecodeOwnerBudget::from_retained_bytes(usize::MAX).map(|_| ()),
            Err(DecodeError::AllocationTooLarge {
                what: "native decode retained allocations",
                requested: usize::MAX,
                cap: DEFAULT_MAX_DECODE_BYTES,
            })
        );

        let mut budget = DecodeOwnerBudget::default();
        assert_eq!(
            budget.include_elements::<u64>(usize::MAX),
            Err(DecodeError::AllocationTooLarge {
                what: "native decode retained allocations",
                requested: usize::MAX,
                cap: DEFAULT_MAX_DECODE_BYTES,
            })
        );
    }

    #[test]
    fn component_owner_budget_accepts_exact_cap_and_rejects_one_over() {
        let mut integers = Vec::new();
        integers
            .try_reserve_exact(3)
            .expect("integer-shadow test capacity");
        integers.push(7_i64);
        let components = vec![ComponentData {
            container: SimdBuffer::<SIMD_WIDTH>::new(vec![1.0]),
            integer_container: Some(integers),
            bit_depth: 25,
            signed: false,
        }];
        let owner_bytes = components.capacity() * size_of::<ComponentData>()
            + components[0].container.capacity() * size_of::<f32>()
            + components[0]
                .integer_container
                .as_ref()
                .expect("integer shadow")
                .capacity()
                * size_of::<i64>();
        let exact_baseline = DEFAULT_MAX_DECODE_BYTES - owner_bytes;

        let exact =
            DecodeOwnerBudget::for_components(exact_baseline, &components, components.capacity())
                .expect("component owners fit the exact aggregate cap");
        assert_eq!(exact.bytes(), DEFAULT_MAX_DECODE_BYTES);
        assert_eq!(
            DecodeOwnerBudget::for_components(
                exact_baseline + 1,
                &components,
                components.capacity(),
            )
            .map(|_| ()),
            Err(DecodeError::AllocationTooLarge {
                what: "native decode retained allocations",
                requested: DEFAULT_MAX_DECODE_BYTES + 1,
                cap: DEFAULT_MAX_DECODE_BYTES,
            })
        );
    }
}
