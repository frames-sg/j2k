// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decoded component-plane owners and named facade handoff contracts.

use alloc::vec::Vec;

use super::ColorSpace;
use crate::error::bail;
use crate::{checked_decode_sample_count, DecodingError, Result};

/// One owned decoded component plane at native bit depth.
pub struct NativeComponentPlane {
    pub(crate) data: Vec<u8>,
    pub(crate) dimensions: (u32, u32),
    pub(crate) bit_depth: u8,
    pub(crate) signed: bool,
    pub(crate) sampling: (u8, u8),
    pub(crate) bytes_per_sample: u8,
}

/// Named allocation-free handoff of one owned native component plane.
#[doc(hidden)]
pub struct NativeComponentPlaneParts {
    /// Packed component bytes.
    pub data: Vec<u8>,
    /// Component dimensions.
    pub dimensions: (u32, u32),
    /// Component bit depth.
    pub bit_depth: u8,
    /// Whether component samples are signed.
    pub signed: bool,
    /// Component sampling factors.
    pub sampling: (u8, u8),
    /// Bytes per packed sample.
    pub bytes_per_sample: u8,
}

impl NativeComponentPlane {
    /// Packed little-endian sample bytes for this component in row-major order.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    crate::__j2k_component_plane_metadata_accessors!();

    /// Bytes used for each packed little-endian sample in [`Self::data`].
    #[must_use]
    pub fn bytes_per_sample(&self) -> u8 {
        self.bytes_per_sample
    }

    /// Return the byte capacity owned by this plane.
    #[doc(hidden)]
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.data.capacity()
    }

    /// Consume this plane into an allocation-free named handoff.
    #[doc(hidden)]
    #[must_use]
    pub fn into_parts(self) -> NativeComponentPlaneParts {
        NativeComponentPlaneParts {
            data: self.data,
            dimensions: self.dimensions,
            bit_depth: self.bit_depth,
            signed: self.signed,
            sampling: self.sampling,
            bytes_per_sample: self.bytes_per_sample,
        }
    }
}

/// Owned decoded native-bit-depth component planes for an image.
pub struct DecodedNativeComponents {
    pub(crate) dimensions: (u32, u32),
    pub(crate) color_space: ColorSpace,
    pub(crate) has_alpha: bool,
    pub(crate) planes: Vec<NativeComponentPlane>,
}

/// Named allocation-free handoff of owned native component planes.
#[doc(hidden)]
pub struct DecodedNativeComponentsParts {
    /// Image dimensions.
    pub dimensions: (u32, u32),
    /// Decoded color space.
    pub color_space: ColorSpace,
    /// Whether an alpha plane is present.
    pub has_alpha: bool,
    /// Owned component planes.
    pub planes: Vec<NativeComponentPlane>,
}

impl DecodedNativeComponents {
    /// Dimensions of the decoded image represented by these planes.
    #[must_use]
    pub fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    /// Color space after JPEG 2000 color conversion has been applied.
    #[must_use]
    pub fn color_space(&self) -> &ColorSpace {
        &self.color_space
    }

    /// Whether the decoded image has an alpha channel.
    #[must_use]
    pub fn has_alpha(&self) -> bool {
        self.has_alpha
    }

    /// Decoded component planes in display order.
    #[must_use]
    pub fn planes(&self) -> &[NativeComponentPlane] {
        &self.planes
    }

    /// Return the actual heap capacity retained by this owned result.
    #[doc(hidden)]
    #[must_use]
    pub fn allocated_bytes(&self) -> Option<usize> {
        let mut bytes = self
            .planes
            .capacity()
            .checked_mul(core::mem::size_of::<NativeComponentPlane>())?;
        for plane in &self.planes {
            bytes = bytes.checked_add(plane.allocated_bytes())?;
        }
        if let ColorSpace::Icc { profile, .. } = &self.color_space {
            bytes = bytes.checked_add(profile.capacity())?;
        }
        Some(bytes)
    }

    /// Consume this result into an allocation-free named handoff.
    #[doc(hidden)]
    #[must_use]
    pub fn into_parts(self) -> DecodedNativeComponentsParts {
        DecodedNativeComponentsParts {
            dimensions: self.dimensions,
            color_space: self.color_space,
            has_alpha: self.has_alpha,
            planes: self.planes,
        }
    }
}

/// A borrowed decoded component plane.
pub struct ComponentPlane<'a> {
    pub(crate) samples: &'a [f32],
    pub(crate) dimensions: (u32, u32),
    pub(crate) bit_depth: u8,
    pub(crate) signed: bool,
    pub(crate) sampling: (u8, u8),
}

/// Named allocation-free handoff of one borrowed component plane.
#[doc(hidden)]
pub struct ComponentPlaneParts<'a> {
    /// Borrowed component samples.
    pub samples: &'a [f32],
    /// Component dimensions.
    pub dimensions: (u32, u32),
    /// Component bit depth.
    pub bit_depth: u8,
    /// Whether component samples are signed.
    pub signed: bool,
    /// Component sampling factors.
    pub sampling: (u8, u8),
}

impl<'a> ComponentPlane<'a> {
    /// Component samples in row-major order.
    #[must_use]
    pub fn samples(&self) -> &'a [f32] {
        self.samples
    }

    crate::__j2k_component_plane_metadata_accessors!();

    /// Consume this borrowed plane into an allocation-free named handoff.
    #[doc(hidden)]
    #[must_use]
    pub fn into_parts(self) -> ComponentPlaneParts<'a> {
        ComponentPlaneParts {
            samples: self.samples,
            dimensions: self.dimensions,
            bit_depth: self.bit_depth,
            signed: self.signed,
            sampling: self.sampling,
        }
    }
}

/// Borrowed decoded component planes for an image.
pub struct DecodedComponents<'a> {
    pub(crate) dimensions: (u32, u32),
    pub(crate) color_space: ColorSpace,
    pub(crate) has_alpha: bool,
    pub(crate) planes: Vec<ComponentPlane<'a>>,
    pub(crate) live_bytes: usize,
}

/// Named allocation-free handoff of borrowed decoded component planes.
#[doc(hidden)]
pub struct DecodedComponentsParts<'a> {
    /// Image dimensions.
    pub dimensions: (u32, u32),
    /// Decoded color space.
    pub color_space: ColorSpace,
    /// Whether an alpha plane is present.
    pub has_alpha: bool,
    /// Borrowed component planes.
    pub planes: Vec<ComponentPlane<'a>>,
}

impl<'a> DecodedComponents<'a> {
    /// Dimensions of the decoded image represented by these planes.
    #[must_use]
    pub fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    /// Color space after JPEG 2000 color conversion has been applied.
    #[must_use]
    pub fn color_space(&self) -> &ColorSpace {
        &self.color_space
    }

    /// Whether the decoded image has an alpha channel.
    #[must_use]
    pub fn has_alpha(&self) -> bool {
        self.has_alpha
    }

    /// Borrowed decoded component planes in display order.
    #[must_use]
    pub fn planes(&self) -> &[ComponentPlane<'a>] {
        &self.planes
    }

    /// Return retained heap capacity that remains live with this result.
    #[doc(hidden)]
    #[must_use]
    pub fn live_bytes(&self) -> usize {
        self.live_bytes
    }

    /// Consume this result into an allocation-free named handoff.
    #[doc(hidden)]
    #[must_use]
    pub fn into_parts(self) -> DecodedComponentsParts<'a> {
        DecodedComponentsParts {
            dimensions: self.dimensions,
            color_space: self.color_space,
            has_alpha: self.has_alpha,
            planes: self.planes,
        }
    }
}

pub(crate) fn native_component_plane_dimensions(
    reference_dimensions: (u32, u32),
    sampling: (u8, u8),
    sample_count: usize,
) -> Result<(u32, u32)> {
    let reference_sample_count =
        checked_decode_sample_count(reference_dimensions.0, reference_dimensions.1)?;
    if sample_count == reference_sample_count {
        return Ok(reference_dimensions);
    }

    let (x_rsiz, y_rsiz) = sampling;
    if x_rsiz == 0 || y_rsiz == 0 {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    let sampled_dimensions = (
        reference_dimensions.0.div_ceil(u32::from(x_rsiz)),
        reference_dimensions.1.div_ceil(u32::from(y_rsiz)),
    );
    let sampled_sample_count =
        checked_decode_sample_count(sampled_dimensions.0, sampled_dimensions.1)?;
    if sample_count == sampled_sample_count {
        return Ok(sampled_dimensions);
    }

    bail!(DecodingError::CodeBlockDecodeFailure)
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use core::mem::size_of;

    use super::*;

    #[test]
    fn native_component_handoff_preserves_owned_capacities() {
        let mut data = Vec::with_capacity(9);
        data.push(3);
        let mut planes = Vec::with_capacity(4);
        planes.push(NativeComponentPlane {
            data,
            dimensions: (1, 1),
            bit_depth: 8,
            signed: false,
            sampling: (1, 1),
            bytes_per_sample: 1,
        });
        let mut profile = Vec::with_capacity(7);
        profile.push(1);
        let decoded = DecodedNativeComponents {
            dimensions: (1, 1),
            color_space: ColorSpace::Icc {
                profile,
                num_channels: 1,
            },
            has_alpha: false,
            planes,
        };
        let expected = decoded.planes.capacity() * size_of::<NativeComponentPlane>()
            + decoded.planes[0].data.capacity()
            + match &decoded.color_space {
                ColorSpace::Icc { profile, .. } => profile.capacity(),
                _ => 0,
            };
        let plane_owner_capacity = decoded.planes.capacity();
        let data_capacity = decoded.planes[0].data.capacity();
        let profile_capacity = match &decoded.color_space {
            ColorSpace::Icc { profile, .. } => profile.capacity(),
            _ => 0,
        };
        assert_eq!(decoded.allocated_bytes(), Some(expected));

        let DecodedNativeComponentsParts {
            color_space,
            planes,
            ..
        } = decoded.into_parts();
        assert_eq!(planes.capacity(), plane_owner_capacity);
        assert_eq!(planes[0].allocated_bytes(), data_capacity);
        assert!(matches!(
            color_space,
            ColorSpace::Icc { profile, .. } if profile.capacity() == profile_capacity
        ));
    }

    #[test]
    fn borrowed_component_handoff_preserves_metadata_capacities() {
        let samples = [2.0_f32];
        let mut planes = Vec::with_capacity(3);
        planes.push(ComponentPlane {
            samples: &samples,
            dimensions: (1, 1),
            bit_depth: 8,
            signed: false,
            sampling: (1, 1),
        });
        let mut profile = Vec::with_capacity(5);
        profile.push(1);
        let decoded = DecodedComponents {
            dimensions: (1, 1),
            color_space: ColorSpace::Icc {
                profile,
                num_channels: 1,
            },
            has_alpha: false,
            planes,
            live_bytes: 123,
        };
        let plane_owner_capacity = decoded.planes.capacity();
        let profile_capacity = match &decoded.color_space {
            ColorSpace::Icc { profile, .. } => profile.capacity(),
            _ => 0,
        };

        assert_eq!(decoded.live_bytes(), 123);
        let DecodedComponentsParts {
            color_space,
            planes,
            ..
        } = decoded.into_parts();
        assert_eq!(planes.capacity(), plane_owner_capacity);
        assert!(core::ptr::eq(
            planes[0].samples().as_ptr(),
            samples.as_ptr()
        ));
        assert!(matches!(
            color_space,
            ColorSpace::Icc { profile, .. } if profile.capacity() == profile_capacity
        ));
    }
}
