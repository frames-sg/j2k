// SPDX-License-Identifier: MIT OR Apache-2.0

//! Facade-owned views over retained HTJ2K preparation metadata.

use alloc::sync::Arc;
use core::any::Any;

pub use j2k_types::{
    HtCodeBlockPayloadRanges as Htj2kPayloadRanges,
    J2kClassicCodeBlockPayload as ClassicCodeBlockPayload, J2kCodestreamRange,
};

/// Opaque retained classic JPEG 2000 execution plan for one prepared image.
///
/// Public callers inspect fragment byte ranges without depending on native
/// implementation types. Backend crates use [`Self::adapter_view`] to borrow
/// the immutable native geometry and segment metadata.
#[derive(Debug, Clone)]
pub struct PreparedClassicPlan {
    plan: Arc<j2k_native::J2kReferencedClassicPlan>,
}

impl PreparedClassicPlan {
    pub(super) fn from_native(plan: j2k_native::J2kReferencedClassicPlan) -> Self {
        Self {
            plan: Arc::new(plan),
        }
    }

    pub(crate) fn native_plan(&self) -> &j2k_native::J2kReferencedClassicPlan {
        &self.plan
    }

    /// Whether the retained geometry decodes one grayscale component.
    #[must_use]
    pub fn is_grayscale(&self) -> bool {
        !self.plan.tiles().is_empty()
            && self
                .plan
                .tiles()
                .iter()
                .all(|tile| tile.grayscale_geometry().is_some())
    }

    /// Whether the retained geometry decodes three color components.
    #[must_use]
    pub fn is_color(&self) -> bool {
        !self.plan.tiles().is_empty()
            && self
                .plan
                .tiles()
                .iter()
                .all(|tile| tile.color_geometry().is_some())
    }

    /// Whether the retained geometry decodes four components in R, G, B, A order.
    #[must_use]
    pub fn is_rgba(&self) -> bool {
        !self.plan.tiles().is_empty()
            && self
                .plan
                .tiles()
                .iter()
                .all(|tile| tile.rgba_geometry().is_some())
    }

    /// Number of referenced classic code-block payloads.
    #[must_use]
    pub fn payload_count(&self) -> usize {
        self.plan.payloads().len()
    }

    /// Number of encoded-input fragments across every code-block payload.
    #[must_use]
    pub fn range_count(&self) -> usize {
        self.plan.ranges().len()
    }

    /// Whether the plan has no referenced code-block payloads.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.plan.payloads().is_empty()
    }

    pub(super) fn uniform_wavelet_transform(&self) -> Option<j2k_native::J2kWaveletTransform> {
        uniform_wavelet_transform(self.plan.tiles())
    }

    /// Return one code-block payload descriptor by traversal index.
    #[must_use]
    pub fn payload(&self, index: usize) -> Option<ClassicCodeBlockPayload> {
        self.plan.payloads().get(index).copied()
    }

    /// Iterate over code-block payload descriptors in component/step/job order.
    pub fn payloads(&self) -> impl ExactSizeIterator<Item = ClassicCodeBlockPayload> + '_ {
        self.plan.payloads().iter().copied()
    }

    /// Return one original-input fragment range by flat index.
    #[must_use]
    pub fn range(&self, index: usize) -> Option<J2kCodestreamRange> {
        self.plan.ranges().get(index).copied()
    }

    /// Iterate over original-input fragment ranges in payload concatenation order.
    pub fn ranges(&self) -> impl ExactSizeIterator<Item = J2kCodestreamRange> + '_ {
        self.plan.ranges().iter().copied()
    }

    /// Borrow immutable backend-specific geometry for adapter downcasting.
    #[doc(hidden)]
    #[must_use]
    pub fn adapter_view(&self) -> &(dyn Any + Send + Sync) {
        self.plan.as_ref()
    }
}

/// Opaque retained HTJ2K execution plan for one prepared image.
///
/// Public callers can inspect payload byte ranges without depending on native
/// implementation types. Device backend crates use [`Self::adapter_view`] to
/// borrow the immutable native plan for the lifetime of this value.
#[derive(Debug, Clone)]
pub struct PreparedHtj2kPlan {
    plan: Arc<j2k_native::J2kReferencedHtj2kPlan>,
}

impl PreparedHtj2kPlan {
    pub(super) fn from_native(plan: j2k_native::J2kReferencedHtj2kPlan) -> Self {
        Self {
            plan: Arc::new(plan),
        }
    }

    pub(crate) fn native_plan(&self) -> &j2k_native::J2kReferencedHtj2kPlan {
        &self.plan
    }

    /// Whether the retained geometry decodes one grayscale component.
    #[must_use]
    pub fn is_grayscale(&self) -> bool {
        !self.plan.tiles().is_empty()
            && self
                .plan
                .tiles()
                .iter()
                .all(|tile| tile.grayscale_geometry().is_some())
    }

    /// Whether the retained geometry decodes three color components.
    #[must_use]
    pub fn is_color(&self) -> bool {
        !self.plan.tiles().is_empty()
            && self
                .plan
                .tiles()
                .iter()
                .all(|tile| tile.color_geometry().is_some())
    }

    /// Whether the retained geometry decodes four components in R, G, B, A order.
    #[must_use]
    pub fn is_rgba(&self) -> bool {
        !self.plan.tiles().is_empty()
            && self
                .plan
                .tiles()
                .iter()
                .all(|tile| tile.rgba_geometry().is_some())
    }

    /// Number of referenced HTJ2K code-block payloads.
    #[must_use]
    pub fn payload_count(&self) -> usize {
        self.plan.payloads().len()
    }

    /// Whether the plan has no referenced HTJ2K code-block payloads.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.plan.payloads().is_empty()
    }

    pub(super) fn uniform_wavelet_transform(&self) -> Option<j2k_native::J2kWaveletTransform> {
        uniform_wavelet_transform(self.plan.tiles())
    }

    /// Return one referenced code-block payload range by traversal index.
    #[must_use]
    pub fn payload(&self, index: usize) -> Option<Htj2kPayloadRanges> {
        self.plan.payloads().get(index).copied()
    }

    /// Iterate over referenced code-block payload ranges in geometry order.
    pub fn payloads(&self) -> impl ExactSizeIterator<Item = Htj2kPayloadRanges> + '_ {
        self.plan.payloads().iter().copied()
    }

    /// Borrow the immutable backend-specific plan for adapter downcasting.
    ///
    /// The returned view is tied to `self`, cannot be mutated, and does not
    /// expose an owning or raw handle. Backend crates that depend directly on
    /// `j2k-native` may downcast it to their supported plan implementation.
    #[doc(hidden)]
    #[must_use]
    pub fn adapter_view(&self) -> &(dyn Any + Send + Sync) {
        self.plan.as_ref()
    }
}

fn uniform_wavelet_transform(
    tiles: &[j2k_native::J2kReferencedTilePlan],
) -> Option<j2k_native::J2kWaveletTransform> {
    let first = tiles.first()?.wavelet_transform();
    tiles
        .iter()
        .all(|tile| tile.wavelet_transform() == first)
        .then_some(first)
}
