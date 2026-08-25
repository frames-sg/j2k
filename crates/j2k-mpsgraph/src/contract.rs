// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{BatchGroupInfo, BatchLayout};
use j2k_core::SampleType;

use crate::Error;

/// Native integer element type exposed to `MPSGraph`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MpsGraphElementType {
    /// Unsigned 8-bit integer samples.
    U8,
    /// Unsigned 16-bit integer samples.
    U16,
    /// Signed 16-bit integer samples.
    I16,
}

/// Validated rank-four `MPSGraph` tensor contract for one homogeneous codec group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MpsGraphTensorSpec {
    shape: [usize; 4],
    element_type: MpsGraphElementType,
}

impl MpsGraphTensorSpec {
    /// Construct and validate an explicit static rank-four tensor contract.
    pub fn new(shape: [usize; 4], element_type: MpsGraphElementType) -> Result<Self, Error> {
        if shape.contains(&0) {
            return Err(Error::InvalidTensorContract {
                reason: "MPSGraph static image dimensions must be nonzero",
            });
        }
        shape
            .into_iter()
            .try_fold(1_usize, usize::checked_mul)
            .ok_or(Error::TensorShapeOverflow)?;
        Ok(Self {
            shape,
            element_type,
        })
    }

    /// Derive a tensor contract from codec-owned group metadata.
    pub fn from_group_info(info: &BatchGroupInfo, image_count: usize) -> Result<Self, Error> {
        if image_count == 0 {
            return Err(Error::InvalidTensorContract {
                reason: "MPSGraph image count must be nonzero",
            });
        }
        let width = usize::try_from(info.dimensions.0).map_err(|_| Error::TensorShapeOverflow)?;
        let height = usize::try_from(info.dimensions.1).map_err(|_| Error::TensorShapeOverflow)?;
        let channels = info.color.channels();
        image_count
            .checked_mul(width)
            .and_then(|samples| samples.checked_mul(height))
            .and_then(|samples| samples.checked_mul(channels))
            .ok_or(Error::TensorShapeOverflow)?;
        let element_type = match info.sample_type {
            SampleType::U8 => MpsGraphElementType::U8,
            SampleType::U16 => MpsGraphElementType::U16,
            SampleType::I16 => MpsGraphElementType::I16,
            _ => {
                return Err(Error::InvalidTensorContract {
                    reason: "MPSGraph direct batches require U8, U16, or I16 samples",
                })
            }
        };
        let shape = match info.layout {
            BatchLayout::Nchw => [image_count, channels, height, width],
            BatchLayout::Nhwc => [image_count, height, width, channels],
            _ => {
                return Err(Error::InvalidTensorContract {
                    reason: "MPSGraph direct batches require NCHW or NHWC layout",
                })
            }
        };
        Self::new(shape, element_type)
    }

    #[must_use]
    /// Return the static `[N, C, H, W]` or `[N, H, W, C]` dimensions.
    pub const fn shape(self) -> [usize; 4] {
        self.shape
    }

    #[must_use]
    /// Return the native integer element type.
    pub const fn element_type(self) -> MpsGraphElementType {
        self.element_type
    }

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    pub(crate) fn byte_len(self) -> Result<usize, Error> {
        let element_size = match self.element_type() {
            MpsGraphElementType::U8 => 1,
            MpsGraphElementType::U16 | MpsGraphElementType::I16 => 2,
        };
        self.shape()
            .into_iter()
            .try_fold(element_size, usize::checked_mul)
            .ok_or(Error::TensorShapeOverflow)
    }
}
