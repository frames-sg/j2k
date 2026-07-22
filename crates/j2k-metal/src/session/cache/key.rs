// SPDX-License-Identifier: MIT OR Apache-2.0

use core::{
    hash::{Hash, Hasher},
    mem::size_of,
};
use std::sync::Arc;

use j2k::DecodeRequest;
use j2k_core::{Downscale, PixelFormat, Rect};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PreparedPlanKind {
    DirectGray,
    DirectColor,
    RegionScaledColor,
    PreparedGray,
    PreparedColor,
}

#[derive(Clone, Copy, Debug)]
enum PreparedPlanInput<'a> {
    Contents(&'a [u8]),
    ArcIdentity(&'a Arc<[u8]>),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PreparedPlanCacheKey<'a> {
    input: PreparedPlanInput<'a>,
    format: PixelFormat,
    roi: Option<Rect>,
    scale: Downscale,
    request: Option<DecodeRequest>,
    kind: PreparedPlanKind,
}

impl Hash for PreparedPlanCacheKey<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
        self.format.hash(state);
        self.roi.hash(state);
        self.scale.hash(state);
        self.request.hash(state);
        match self.input {
            PreparedPlanInput::Contents(input) => {
                0_u8.hash(state);
                input.hash(state);
            }
            PreparedPlanInput::ArcIdentity(input) => {
                1_u8.hash(state);
                Arc::as_ptr(input).addr().hash(state);
                input.len().hash(state);
            }
        }
    }
}

impl<'a> PreparedPlanCacheKey<'a> {
    pub(crate) const fn direct_gray(input: &'a [u8], format: PixelFormat) -> Self {
        Self {
            input: PreparedPlanInput::Contents(input),
            format,
            roi: None,
            scale: Downscale::None,
            request: None,
            kind: PreparedPlanKind::DirectGray,
        }
    }

    pub(crate) const fn direct_color(input: &'a [u8], format: PixelFormat) -> Self {
        Self {
            input: PreparedPlanInput::Contents(input),
            format,
            roi: None,
            scale: Downscale::None,
            request: None,
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
            input: PreparedPlanInput::Contents(input),
            format,
            roi: Some(roi),
            scale,
            request: None,
            kind: PreparedPlanKind::RegionScaledColor,
        }
    }

    pub(crate) const fn prepared_gray(
        input: &'a Arc<[u8]>,
        request: DecodeRequest,
        format: PixelFormat,
    ) -> Self {
        Self {
            input: PreparedPlanInput::ArcIdentity(input),
            format,
            roi: None,
            scale: Downscale::None,
            request: Some(request),
            kind: PreparedPlanKind::PreparedGray,
        }
    }

    pub(crate) const fn prepared_color(
        input: &'a Arc<[u8]>,
        request: DecodeRequest,
        format: PixelFormat,
    ) -> Self {
        Self {
            input: PreparedPlanInput::ArcIdentity(input),
            format,
            roi: None,
            scale: Downscale::None,
            request: Some(request),
            kind: PreparedPlanKind::PreparedColor,
        }
    }

    pub(super) fn input_len(self) -> usize {
        match self.input {
            PreparedPlanInput::Contents(input) => input.len(),
            PreparedPlanInput::ArcIdentity(input) => retained_arc_bytes(input),
        }
    }
}

enum OwnedPreparedPlanInput {
    Contents(Vec<u8>),
    ArcIdentity(Arc<[u8]>),
}

pub(super) struct OwnedPreparedPlanCacheKey {
    input: OwnedPreparedPlanInput,
    format: PixelFormat,
    roi: Option<Rect>,
    scale: Downscale,
    request: Option<DecodeRequest>,
    kind: PreparedPlanKind,
}

impl OwnedPreparedPlanCacheKey {
    pub(super) fn try_from_borrowed(
        key: PreparedPlanCacheKey<'_>,
    ) -> Result<Self, std::collections::TryReserveError> {
        let input = match key.input {
            PreparedPlanInput::Contents(source) => {
                let mut owned = Vec::new();
                owned.try_reserve_exact(source.len())?;
                owned.extend_from_slice(source);
                OwnedPreparedPlanInput::Contents(owned)
            }
            PreparedPlanInput::ArcIdentity(source) => {
                OwnedPreparedPlanInput::ArcIdentity(source.clone())
            }
        };
        Ok(Self {
            input,
            format: key.format,
            roi: key.roi,
            scale: key.scale,
            request: key.request,
            kind: key.kind,
        })
    }

    pub(super) fn matches(&self, key: PreparedPlanCacheKey<'_>) -> bool {
        let input_matches = match (&self.input, key.input) {
            (OwnedPreparedPlanInput::Contents(owned), PreparedPlanInput::Contents(input)) => {
                owned.as_slice() == input
            }
            (OwnedPreparedPlanInput::ArcIdentity(owned), PreparedPlanInput::ArcIdentity(input)) => {
                Arc::ptr_eq(owned, input)
            }
            (OwnedPreparedPlanInput::Contents(_), PreparedPlanInput::ArcIdentity(_))
            | (OwnedPreparedPlanInput::ArcIdentity(_), PreparedPlanInput::Contents(_)) => false,
        };
        input_matches
            && self.format == key.format
            && self.roi == key.roi
            && self.scale == key.scale
            && self.request == key.request
            && self.kind == key.kind
    }

    pub(super) fn input_capacity(&self) -> usize {
        match &self.input {
            OwnedPreparedPlanInput::Contents(input) => input.capacity(),
            OwnedPreparedPlanInput::ArcIdentity(input) => retained_arc_bytes(input),
        }
    }

    #[cfg(test)]
    pub(super) fn with_input_capacity_for_test(capacity: usize, input: &[u8]) -> Self {
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(capacity)
            .expect("test key capacity reservation");
        owned.extend_from_slice(input);
        Self {
            input: OwnedPreparedPlanInput::Contents(owned),
            format: PixelFormat::Rgb8,
            roi: None,
            scale: Downscale::None,
            request: None,
            kind: PreparedPlanKind::DirectColor,
        }
    }
}

fn retained_arc_bytes(input: &Arc<[u8]>) -> usize {
    input.len().saturating_add(2 * size_of::<usize>())
}
