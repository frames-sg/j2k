// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{Downscale, PixelFormat, Rect};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PreparedPlanKind {
    DirectGray,
    DirectColor,
    RegionScaledColor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PreparedPlanCacheKey<'a> {
    input: &'a [u8],
    format: PixelFormat,
    roi: Option<Rect>,
    scale: Downscale,
    kind: PreparedPlanKind,
}

impl<'a> PreparedPlanCacheKey<'a> {
    pub(crate) const fn direct_gray(input: &'a [u8], format: PixelFormat) -> Self {
        Self {
            input,
            format,
            roi: None,
            scale: Downscale::None,
            kind: PreparedPlanKind::DirectGray,
        }
    }

    pub(crate) const fn direct_color(input: &'a [u8], format: PixelFormat) -> Self {
        Self {
            input,
            format,
            roi: None,
            scale: Downscale::None,
            kind: PreparedPlanKind::DirectColor,
        }
    }

    pub(crate) const fn region_scaled_color(
        input: &'a [u8],
        format: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Self {
        Self {
            input,
            format,
            roi: Some(roi),
            scale,
            kind: PreparedPlanKind::RegionScaledColor,
        }
    }

    pub(super) const fn input_len(self) -> usize {
        self.input.len()
    }
}

pub(super) struct OwnedPreparedPlanCacheKey {
    input: Vec<u8>,
    format: PixelFormat,
    roi: Option<Rect>,
    scale: Downscale,
    kind: PreparedPlanKind,
}

impl OwnedPreparedPlanCacheKey {
    pub(super) fn try_from_borrowed(
        key: PreparedPlanCacheKey<'_>,
    ) -> Result<Self, std::collections::TryReserveError> {
        let mut input = Vec::new();
        input.try_reserve_exact(key.input.len())?;
        input.extend_from_slice(key.input);
        Ok(Self {
            input,
            format: key.format,
            roi: key.roi,
            scale: key.scale,
            kind: key.kind,
        })
    }

    pub(super) fn matches(&self, key: PreparedPlanCacheKey<'_>) -> bool {
        self.input.as_slice() == key.input
            && self.format == key.format
            && self.roi == key.roi
            && self.scale == key.scale
            && self.kind == key.kind
    }

    pub(super) const fn input_capacity(&self) -> usize {
        self.input.capacity()
    }

    #[cfg(test)]
    pub(super) fn with_input_capacity_for_test(capacity: usize, input: &[u8]) -> Self {
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(capacity)
            .expect("test key capacity reservation");
        owned.extend_from_slice(input);
        Self {
            input: owned,
            format: PixelFormat::Rgb8,
            roi: None,
            scale: Downscale::None,
            kind: PreparedPlanKind::DirectColor,
        }
    }
}
