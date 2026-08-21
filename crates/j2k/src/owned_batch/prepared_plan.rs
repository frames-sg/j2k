// SPDX-License-Identifier: MIT OR Apache-2.0

//! Facade-owned views over retained HTJ2K preparation metadata.

use alloc::sync::Arc;

pub use j2k_types::{
    HtCodeBlockPayloadRanges as Htj2kPayloadRanges,
    J2kClassicCodeBlockPayload as ClassicCodeBlockPayload, J2kCodestreamRange,
};

/// Backend-neutral retained execution geometry for a classic JPEG 2000 image.
pub type ClassicPreparedGeometry = j2k_types::J2kReferencedClassicPlan;

/// Backend-neutral retained execution geometry for an HTJ2K image.
pub type Htj2kPreparedGeometry = j2k_types::J2kReferencedHtj2kPlan;

/// Image-level geometry shared by classic and HTJ2K prepared plans.
pub type PreparedImageGeometry<'a> = j2k_types::J2kReferencedImageGeometry<'a>;

/// Typed retained classic JPEG 2000 execution plan for one prepared image.
#[derive(Debug, Clone)]
pub struct PreparedClassicPlan {
    plan: Arc<ClassicPreparedGeometry>,
}

impl PreparedClassicPlan {
    pub(super) fn from_native(plan: ClassicPreparedGeometry) -> Self {
        Self {
            plan: Arc::new(plan),
        }
    }

    pub(crate) fn native_plan(&self) -> &ClassicPreparedGeometry {
        &self.plan
    }

    /// Borrow the immutable execution geometry shared by all decode backends.
    #[must_use]
    pub fn geometry(&self) -> &ClassicPreparedGeometry {
        &self.plan
    }

    /// Borrow image-level geometry whose semantics are independent of block coding mode.
    #[must_use]
    pub fn image_geometry(&self) -> PreparedImageGeometry<'_> {
        self.plan.image_geometry()
    }

    /// Whether the retained geometry decodes one grayscale component.
    #[must_use]
    pub fn is_grayscale(&self) -> bool {
        self.image_geometry().is_grayscale()
    }

    /// Whether the retained geometry decodes three color components.
    #[must_use]
    pub fn is_color(&self) -> bool {
        self.image_geometry().is_color()
    }

    /// Whether the retained geometry decodes four components in R, G, B, A order.
    #[must_use]
    pub fn is_rgba(&self) -> bool {
        self.image_geometry().is_rgba()
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
        self.image_geometry().uniform_wavelet_transform()
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
}

/// Typed retained HTJ2K execution plan for one prepared image.
#[derive(Debug, Clone)]
pub struct PreparedHtj2kPlan {
    plan: Arc<Htj2kPreparedGeometry>,
}

impl PreparedHtj2kPlan {
    pub(super) fn from_native(plan: Htj2kPreparedGeometry) -> Self {
        Self {
            plan: Arc::new(plan),
        }
    }

    pub(crate) fn native_plan(&self) -> &Htj2kPreparedGeometry {
        &self.plan
    }

    /// Borrow the immutable execution geometry shared by all decode backends.
    #[must_use]
    pub fn geometry(&self) -> &Htj2kPreparedGeometry {
        &self.plan
    }

    /// Borrow image-level geometry whose semantics are independent of block coding mode.
    #[must_use]
    pub fn image_geometry(&self) -> PreparedImageGeometry<'_> {
        self.plan.image_geometry()
    }

    /// Whether the retained geometry decodes one grayscale component.
    #[must_use]
    pub fn is_grayscale(&self) -> bool {
        self.image_geometry().is_grayscale()
    }

    /// Whether the retained geometry decodes three color components.
    #[must_use]
    pub fn is_color(&self) -> bool {
        self.image_geometry().is_color()
    }

    /// Whether the retained geometry decodes four components in R, G, B, A order.
    #[must_use]
    pub fn is_rgba(&self) -> bool {
        self.image_geometry().is_rgba()
    }

    /// Number of referenced HTJ2K payload records, including refinement continuations.
    #[must_use]
    pub fn payload_count(&self) -> usize {
        self.plan.payloads().len()
    }

    /// Whether the plan has no referenced HTJ2K payload records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.plan.payloads().is_empty()
    }

    pub(super) fn uniform_wavelet_transform(&self) -> Option<j2k_native::J2kWaveletTransform> {
        self.image_geometry().uniform_wavelet_transform()
    }

    /// Return one referenced payload record by traversal index.
    #[must_use]
    pub fn payload(&self, index: usize) -> Option<Htj2kPayloadRanges> {
        self.plan.payloads().get(index).copied()
    }

    /// Iterate over referenced payload records in geometry order.
    pub fn payloads(&self) -> impl ExactSizeIterator<Item = Htj2kPayloadRanges> + '_ {
        self.plan.payloads().iter().copied()
    }
}
