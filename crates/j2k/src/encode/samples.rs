// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{format, string::ToString};

use j2k_core::Unsupported;

use super::contracts::{MAX_JPEG2000_PART1_COMPONENTS, MAX_PART1_SAMPLE_BIT_DEPTH};
use crate::J2kError;

pub(super) fn raw_pixel_bytes_per_sample(bit_depth: u8) -> usize {
    usize::from(bit_depth).div_ceil(8).max(1)
}

/// Borrowed interleaved samples and image geometry for lossless encoding.
#[derive(Debug, Clone, Copy)]
pub struct J2kLosslessSamples<'a> {
    /// Interleaved sample bytes.
    pub data: &'a [u8],
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Component count. Component counts beyond four are encoded as independent
    /// component planes without a multi-component transform.
    pub components: u16,
    /// Significant bits per component sample.
    pub bit_depth: u8,
    /// Whether component samples are signed.
    pub signed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SampleGeometry {
    expected_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
struct SampleGeometryRequest<'a> {
    data: &'a [u8],
    width: u32,
    height: u32,
    components: u16,
    bit_depth: u8,
    max_bit_depth: u8,
    component_what: &'static str,
    bit_depth_what: &'static str,
}

fn validate_sample_geometry(
    request: SampleGeometryRequest<'_>,
) -> Result<SampleGeometry, J2kError> {
    let SampleGeometryRequest {
        data,
        width,
        height,
        components,
        bit_depth,
        max_bit_depth,
        component_what,
        bit_depth_what,
    } = request;
    if width == 0 || height == 0 {
        return Err(J2kError::InvalidSamples {
            what: "dimensions must be non-zero".to_string(),
        });
    }
    if components == 0 || components > MAX_JPEG2000_PART1_COMPONENTS {
        return Err(J2kError::Unsupported(Unsupported {
            what: component_what,
        }));
    }
    if bit_depth == 0 || bit_depth > max_bit_depth {
        return Err(J2kError::Unsupported(Unsupported {
            what: bit_depth_what,
        }));
    }
    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth);
    let expected_bytes = (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(usize::from(components)))
        .and_then(|samples| samples.checked_mul(bytes_per_sample))
        .ok_or(J2kError::DimensionOverflow { width, height })?;
    if data.len() != expected_bytes {
        let what = if data.len() < expected_bytes {
            format!(
                "pixel data too short: expected {expected_bytes} bytes, got {}",
                data.len()
            )
        } else {
            format!(
                "pixel data has trailing bytes: expected {expected_bytes} bytes, got {}",
                data.len()
            )
        };
        return Err(J2kError::InvalidSamples { what });
    }
    Ok(SampleGeometry { expected_bytes })
}

impl<'a> J2kLosslessSamples<'a> {
    /// Validate and construct a sample descriptor.
    pub fn new(
        data: &'a [u8],
        width: u32,
        height: u32,
        components: u16,
        bit_depth: u8,
        signed: bool,
    ) -> Result<Self, J2kError> {
        let geometry = validate_sample_geometry(SampleGeometryRequest {
            data,
            width,
            height,
            components,
            bit_depth,
            max_bit_depth: MAX_PART1_SAMPLE_BIT_DEPTH,
            component_what: "JPEG 2000 lossless encode supports 1-16384 component samples",
            bit_depth_what: "JPEG 2000 lossless encode supports 1-38 bits per sample for classic reversible codestreams",
        })?;
        debug_assert_eq!(geometry.expected_bytes, data.len());
        Ok(Self {
            data,
            width,
            height,
            components,
            bit_depth,
            signed,
        })
    }
}

/// Rectangular region-of-interest request for lossless JPEG 2000 maxshift
/// encoding.
///
/// The rectangle is expressed in full-resolution image pixels. All regions for
/// one component must use the same non-zero `shift`, because JPEG 2000 stores
/// one RGN maxshift value per component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct J2kRoiRegion {
    /// Component index to which the ROI applies.
    pub component: u16,
    /// Left edge in image pixels.
    pub x: u32,
    /// Top edge in image pixels.
    pub y: u32,
    /// Width in image pixels.
    pub width: u32,
    /// Height in image pixels.
    pub height: u32,
    /// Maxshift value to write for this component.
    pub shift: u8,
}

/// Borrowed samples for one lossless component plane.
#[derive(Debug, Clone, Copy)]
pub struct J2kLosslessComponentPlane<'a> {
    /// Row-major little-endian samples for this component's own SIZ grid.
    pub data: &'a [u8],
    /// Horizontal SIZ sampling factor (`XRsiz`).
    pub x_rsiz: u8,
    /// Vertical SIZ sampling factor (`YRsiz`).
    pub y_rsiz: u8,
}

/// Borrowed component-plane samples and reference-grid image geometry for
/// lossless encoding.
#[derive(Debug, Clone, Copy)]
pub struct J2kLosslessComponentSamples<'a> {
    /// Component planes in codestream order.
    pub planes: &'a [J2kLosslessComponentPlane<'a>],
    /// Reference-grid image width in pixels.
    pub width: u32,
    /// Reference-grid image height in pixels.
    pub height: u32,
    /// Significant bits per component sample. Mixed component bit depths are
    /// not yet supported by the encode facade.
    pub bit_depth: u8,
    /// Whether every component sample is signed. Mixed signedness is not yet
    /// supported by the encode facade.
    pub signed: bool,
}

impl<'a> J2kLosslessComponentSamples<'a> {
    /// Validate and construct a component-plane sample descriptor.
    pub fn new(
        planes: &'a [J2kLosslessComponentPlane<'a>],
        width: u32,
        height: u32,
        bit_depth: u8,
        signed: bool,
    ) -> Result<Self, J2kError> {
        if width == 0 || height == 0 {
            return Err(J2kError::InvalidSamples {
                what: "dimensions must be non-zero".to_string(),
            });
        }
        if planes.is_empty() || planes.len() > usize::from(MAX_JPEG2000_PART1_COMPONENTS) {
            return Err(J2kError::Unsupported(Unsupported {
                what: "JPEG 2000 lossless component-plane encode supports 1-16384 components",
            }));
        }
        if bit_depth == 0 || bit_depth > MAX_PART1_SAMPLE_BIT_DEPTH {
            return Err(J2kError::Unsupported(Unsupported {
                what: "JPEG 2000 lossless component-plane encode supports 1-38 bits per sample",
            }));
        }
        for (index, plane) in planes.iter().enumerate() {
            validate_component_plane_geometry(plane, width, height, bit_depth, index)?;
        }
        Ok(Self {
            planes,
            width,
            height,
            bit_depth,
            signed,
        })
    }

    /// Return the component count.
    #[must_use]
    pub fn components(&self) -> u16 {
        u16::try_from(self.planes.len()).unwrap_or(MAX_JPEG2000_PART1_COMPONENTS)
    }
}

/// Borrowed samples for one typed lossless component plane.
#[derive(Debug, Clone, Copy)]
pub struct J2kLosslessTypedComponentPlane<'a> {
    /// Row-major little-endian samples for this component's own SIZ grid.
    pub data: &'a [u8],
    /// Horizontal SIZ sampling factor (`XRsiz`).
    pub x_rsiz: u8,
    /// Vertical SIZ sampling factor (`YRsiz`).
    pub y_rsiz: u8,
    /// Significant bits per sample for this component.
    pub bit_depth: u8,
    /// Whether samples in this component are signed.
    pub signed: bool,
}

/// Borrowed typed component-plane samples and reference-grid image geometry for
/// lossless encoding.
#[derive(Debug, Clone, Copy)]
pub struct J2kLosslessTypedComponentSamples<'a> {
    /// Component planes in codestream order.
    pub planes: &'a [J2kLosslessTypedComponentPlane<'a>],
    /// Reference-grid image width in pixels.
    pub width: u32,
    /// Reference-grid image height in pixels.
    pub height: u32,
}

impl<'a> J2kLosslessTypedComponentSamples<'a> {
    /// Validate and construct a typed component-plane sample descriptor.
    pub fn new(
        planes: &'a [J2kLosslessTypedComponentPlane<'a>],
        width: u32,
        height: u32,
    ) -> Result<Self, J2kError> {
        if width == 0 || height == 0 {
            return Err(J2kError::InvalidSamples {
                what: "dimensions must be non-zero".to_string(),
            });
        }
        if planes.is_empty() || planes.len() > usize::from(MAX_JPEG2000_PART1_COMPONENTS) {
            return Err(J2kError::Unsupported(Unsupported {
                what: "JPEG 2000 lossless typed component-plane encode supports 1-16384 components",
            }));
        }
        for (index, plane) in planes.iter().enumerate() {
            validate_typed_component_plane_geometry(plane, width, height, index)?;
        }
        Ok(Self {
            planes,
            width,
            height,
        })
    }

    /// Return the component count.
    #[must_use]
    pub fn components(&self) -> u16 {
        u16::try_from(self.planes.len()).unwrap_or(MAX_JPEG2000_PART1_COMPONENTS)
    }

    /// Return the maximum significant bit depth across all components.
    #[must_use]
    pub fn max_bit_depth(&self) -> u8 {
        self.planes
            .iter()
            .map(|plane| plane.bit_depth)
            .max()
            .unwrap_or(0)
    }

    /// Return whether every component is signed.
    #[must_use]
    pub fn all_components_signed(&self) -> bool {
        self.planes.iter().all(|plane| plane.signed)
    }
}

fn validate_component_plane_geometry(
    plane: &J2kLosslessComponentPlane<'_>,
    width: u32,
    height: u32,
    bit_depth: u8,
    index: usize,
) -> Result<(), J2kError> {
    if plane.x_rsiz == 0 || plane.y_rsiz == 0 {
        return Err(J2kError::InvalidSamples {
            what: format!("component plane {index} sampling factors must be non-zero"),
        });
    }
    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth);
    let component_width = width.div_ceil(u32::from(plane.x_rsiz));
    let component_height = height.div_ceil(u32::from(plane.y_rsiz));
    let expected_bytes = (component_width as usize)
        .checked_mul(component_height as usize)
        .and_then(|samples| samples.checked_mul(bytes_per_sample))
        .ok_or(J2kError::DimensionOverflow { width, height })?;
    if plane.data.len() != expected_bytes {
        return Err(J2kError::InvalidSamples {
            what: format!(
                "component plane {index} data length mismatch: expected {expected_bytes} bytes, got {}",
                plane.data.len()
            ),
        });
    }
    Ok(())
}

fn validate_typed_component_plane_geometry(
    plane: &J2kLosslessTypedComponentPlane<'_>,
    width: u32,
    height: u32,
    index: usize,
) -> Result<(), J2kError> {
    if plane.x_rsiz == 0 || plane.y_rsiz == 0 {
        return Err(J2kError::InvalidSamples {
            what: format!("component plane {index} sampling factors must be non-zero"),
        });
    }
    if plane.bit_depth == 0 || plane.bit_depth > MAX_PART1_SAMPLE_BIT_DEPTH {
        return Err(J2kError::Unsupported(Unsupported {
            what: "JPEG 2000 lossless typed component-plane encode supports 1-38 bits per sample",
        }));
    }
    let bytes_per_sample = raw_pixel_bytes_per_sample(plane.bit_depth);
    let component_width = width.div_ceil(u32::from(plane.x_rsiz));
    let component_height = height.div_ceil(u32::from(plane.y_rsiz));
    let expected_bytes = (component_width as usize)
        .checked_mul(component_height as usize)
        .and_then(|samples| samples.checked_mul(bytes_per_sample))
        .ok_or(J2kError::DimensionOverflow { width, height })?;
    if plane.data.len() != expected_bytes {
        return Err(J2kError::InvalidSamples {
            what: format!(
                "component plane {index} data length mismatch: expected {expected_bytes} bytes, got {}",
                plane.data.len()
            ),
        });
    }
    Ok(())
}

/// Borrowed interleaved samples and image geometry for lossy encoding.
#[derive(Debug, Clone, Copy)]
pub struct J2kLossySamples<'a> {
    /// Interleaved sample bytes.
    pub data: &'a [u8],
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Component count. Component counts beyond four are encoded as independent
    /// component planes without a multi-component transform.
    pub components: u16,
    /// Significant bits per component sample.
    pub bit_depth: u8,
    /// Whether component samples are signed.
    pub signed: bool,
}

impl<'a> J2kLossySamples<'a> {
    /// Validate and construct a lossy sample descriptor.
    pub fn new(
        data: &'a [u8],
        width: u32,
        height: u32,
        components: u16,
        bit_depth: u8,
        signed: bool,
    ) -> Result<Self, J2kError> {
        let geometry = validate_sample_geometry(SampleGeometryRequest {
            data,
            width,
            height,
            components,
            bit_depth,
            max_bit_depth: MAX_PART1_SAMPLE_BIT_DEPTH,
            component_what: "JPEG 2000 lossy encode supports 1-16384 component samples",
            bit_depth_what: "JPEG 2000 lossy encode supports 1-38 bits per sample",
        })?;
        debug_assert_eq!(geometry.expected_bytes, data.len());
        Ok(Self {
            data,
            width,
            height,
            components,
            bit_depth,
            signed,
        })
    }
}
