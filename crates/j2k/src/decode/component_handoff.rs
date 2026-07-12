// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-bounded handoff from native decoded component owners.

use crate::{backend::ColorSpace, J2kError};
use alloc::vec::Vec;
use core::mem::size_of;
use j2k_core::{
    ensure_allocation_within_cap, try_host_vec_with_capacity, BufferError, HostAllocationError,
    DEFAULT_MAX_HOST_ALLOCATION_BYTES,
};

const COMPONENT_HANDOFF_WHAT: &str = "J2K decoded component facade handoff";
type NativePlaneParts = (Vec<u8>, (u32, u32), u8, bool, (u8, u8), u8);

macro_rules! impl_decoded_components_metadata_accessors {
    () => {
        /// Dimensions of the decoded image represented by these planes.
        #[must_use]
        pub fn dimensions(&self) -> (u32, u32) {
            self.dimensions
        }

        /// Color space after JPEG 2000 color conversion has been applied.
        #[must_use]
        pub fn color_space(&self) -> &J2kDecodedColorSpace {
            &self.color_space
        }

        /// Whether the decoded image has an alpha channel.
        #[must_use]
        pub fn has_alpha(&self) -> bool {
            self.has_alpha
        }
    };
}

/// Decoded JPEG 2000 color space metadata for component-plane outputs.
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum J2kDecodedColorSpace {
    /// Grayscale image data.
    Gray,
    /// RGB image data.
    Rgb,
    /// CMYK image data.
    Cmyk,
    /// Unknown image data with the given number of channels.
    Unknown {
        /// Number of channels represented by the color space.
        num_channels: u16,
    },
    /// ICC-described image data.
    Icc {
        /// ICC profile bytes.
        profile: Vec<u8>,
        /// Number of channels represented by the ICC profile.
        num_channels: u16,
    },
}

impl J2kDecodedColorSpace {
    fn from_native(color_space: ColorSpace) -> Self {
        match color_space {
            ColorSpace::Gray => Self::Gray,
            ColorSpace::RGB => Self::Rgb,
            ColorSpace::CMYK => Self::Cmyk,
            ColorSpace::Unknown { num_channels } => Self::Unknown { num_channels },
            ColorSpace::Icc {
                profile,
                num_channels,
            } => Self::Icc {
                profile,
                num_channels,
            },
        }
    }
}

/// One borrowed decoded component plane.
#[derive(Debug, Clone, Copy)]
pub struct J2kComponentPlane<'a> {
    samples: &'a [f32],
    dimensions: (u32, u32),
    bit_depth: u8,
    signed: bool,
    sampling: (u8, u8),
}

impl<'a> J2kComponentPlane<'a> {
    fn from_native(plane: j2k_native::ComponentPlane<'a>) -> Self {
        let (samples, dimensions, bit_depth, signed, sampling) = plane.into_parts();
        Self {
            samples,
            dimensions,
            bit_depth,
            signed,
            sampling,
        }
    }

    /// Component samples in row-major order.
    #[must_use]
    pub fn samples(&self) -> &'a [f32] {
        self.samples
    }

    j2k_native::__j2k_component_plane_metadata_accessors!();
}

/// Borrowed decoded component planes for an image.
#[derive(Debug)]
pub struct J2kDecodedComponents<'a> {
    dimensions: (u32, u32),
    color_space: J2kDecodedColorSpace,
    has_alpha: bool,
    planes: Vec<J2kComponentPlane<'a>>,
}

impl<'a> J2kDecodedComponents<'a> {
    pub(crate) fn try_from_native(
        decoded: j2k_native::DecodedComponents<'a>,
        retained_image_bytes: usize,
    ) -> Result<Self, J2kError> {
        let decoded_live_bytes = decoded.live_bytes();
        let (dimensions, color_space, has_alpha, native_planes) = decoded.into_parts();
        let mut planes = try_destination_metadata(
            native_planes.len(),
            retained_image_bytes,
            decoded_live_bytes,
        )?;
        planes.extend(
            native_planes
                .into_iter()
                .map(J2kComponentPlane::from_native),
        );
        Ok(Self {
            dimensions,
            color_space: J2kDecodedColorSpace::from_native(color_space),
            has_alpha,
            planes,
        })
    }

    impl_decoded_components_metadata_accessors!();

    /// Borrowed decoded component planes in display order.
    #[must_use]
    pub fn planes(&self) -> &[J2kComponentPlane<'a>] {
        &self.planes
    }
}

/// One owned decoded component plane at native bit depth.
#[derive(Debug, PartialEq, Eq)]
pub struct J2kNativeComponentPlane {
    data: Vec<u8>,
    dimensions: (u32, u32),
    bit_depth: u8,
    signed: bool,
    sampling: (u8, u8),
    bytes_per_sample: u8,
}

impl J2kNativeComponentPlane {
    fn from_native(plane: j2k_native::NativeComponentPlane) -> Self {
        Self::from_parts(plane.into_parts())
    }

    fn from_parts(
        (data, dimensions, bit_depth, signed, sampling, bytes_per_sample): NativePlaneParts,
    ) -> Self {
        Self {
            data,
            dimensions,
            bit_depth,
            signed,
            sampling,
            bytes_per_sample,
        }
    }

    /// Packed little-endian sample bytes for this component in row-major order.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    j2k_native::__j2k_component_plane_metadata_accessors!();

    /// Bytes used for each packed little-endian sample in [`Self::data`].
    #[must_use]
    pub fn bytes_per_sample(&self) -> u8 {
        self.bytes_per_sample
    }
}

/// Owned decoded native-bit-depth component planes for an image.
#[derive(Debug, PartialEq, Eq)]
pub struct J2kDecodedNativeComponents {
    dimensions: (u32, u32),
    color_space: J2kDecodedColorSpace,
    has_alpha: bool,
    planes: Vec<J2kNativeComponentPlane>,
}

impl J2kDecodedNativeComponents {
    pub(crate) fn try_from_native(
        decoded: j2k_native::DecodedNativeComponents,
        retained_image_bytes: usize,
    ) -> Result<Self, J2kError> {
        let decoded_live_bytes = decoded
            .allocated_bytes()
            .ok_or_else(handoff_size_overflow)?;
        let (dimensions, color_space, has_alpha, native_planes) = decoded.into_parts();
        let mut planes = try_destination_metadata(
            native_planes.len(),
            retained_image_bytes,
            decoded_live_bytes,
        )?;
        planes.extend(
            native_planes
                .into_iter()
                .map(J2kNativeComponentPlane::from_native),
        );
        Ok(Self {
            dimensions,
            color_space: J2kDecodedColorSpace::from_native(color_space),
            has_alpha,
            planes,
        })
    }

    impl_decoded_components_metadata_accessors!();

    /// Decoded component planes in display order.
    #[must_use]
    pub fn planes(&self) -> &[J2kNativeComponentPlane] {
        &self.planes
    }
}

fn try_destination_metadata<T>(
    len: usize,
    retained_image_bytes: usize,
    decoded_live_bytes: usize,
) -> Result<Vec<T>, J2kError> {
    checked_handoff_bytes::<T>(retained_image_bytes, decoded_live_bytes, len)?;
    let values = try_host_vec_with_capacity(len).map_err(host_allocation_error)?;
    checked_handoff_bytes::<T>(retained_image_bytes, decoded_live_bytes, values.capacity())?;
    Ok(values)
}

fn checked_handoff_bytes<T>(
    retained_image_bytes: usize,
    decoded_live_bytes: usize,
    destination_capacity: usize,
) -> Result<usize, J2kError> {
    let destination_bytes = destination_capacity
        .checked_mul(size_of::<T>())
        .ok_or_else(handoff_size_overflow)?;
    let requested = retained_image_bytes
        .checked_add(decoded_live_bytes)
        .ok_or_else(handoff_size_overflow)?
        .checked_add(destination_bytes)
        .ok_or_else(handoff_size_overflow)?;
    ensure_allocation_within_cap(
        requested,
        DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        COMPONENT_HANDOFF_WHAT,
    )
    .map_err(J2kError::from)
}

fn handoff_size_overflow() -> J2kError {
    BufferError::SizeOverflow {
        what: COMPONENT_HANDOFF_WHAT,
    }
    .into()
}

fn host_allocation_error(error: HostAllocationError) -> J2kError {
    BufferError::HostAllocationFailed {
        bytes: error.requested_bytes(),
        what: COMPONENT_HANDOFF_WHAT,
    }
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_handoff_has_an_exact_shared_cap_boundary() {
        assert_eq!(
            checked_handoff_bytes::<u8>(DEFAULT_MAX_HOST_ALLOCATION_BYTES - 2, 1, 1)
                .expect("exact cap"),
            DEFAULT_MAX_HOST_ALLOCATION_BYTES
        );
        assert!(matches!(
            checked_handoff_bytes::<u8>(DEFAULT_MAX_HOST_ALLOCATION_BYTES - 1, 1, 1),
            Err(J2kError::Buffer(BufferError::AllocationTooLarge {
                requested,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                what: COMPONENT_HANDOFF_WHAT,
            })) if requested == DEFAULT_MAX_HOST_ALLOCATION_BYTES + 1
        ));
    }

    #[test]
    fn native_plane_parts_move_payload_without_copying() {
        let data = Vec::from([1u8, 2, 3, 4]);
        let pointer = data.as_ptr();
        let plane = J2kNativeComponentPlane::from_parts((data, (2, 2), 8, false, (1, 1), 1));
        assert_eq!(plane.data().as_ptr(), pointer);
        assert_eq!(plane.data(), [1, 2, 3, 4]);
    }

    #[test]
    fn native_color_handoff_moves_icc_profile_without_copying() {
        let profile = Vec::from([3u8, 7, 11]);
        let pointer = profile.as_ptr();
        let color = J2kDecodedColorSpace::from_native(ColorSpace::Icc {
            profile,
            num_channels: 3,
        });
        let J2kDecodedColorSpace::Icc { profile, .. } = color else {
            panic!("ICC handoff changed color kind");
        };
        assert_eq!(profile.as_ptr(), pointer);
    }
}
