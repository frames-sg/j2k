// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend-neutral resident buffer descriptors for transcode handoff.

use core::fmt;
use core::marker::PhantomData;

use j2k_core::{BackendKind, DeviceMemoryRange};

/// Error returned by resident transcode handoff descriptor constructors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentHandoffError {
    /// Buffer range length is zero.
    EmptyRange,
    /// Buffer range offset plus length overflowed.
    OffsetOverflow,
    /// Buffer range exceeds the caller-supplied allocation length.
    RangeExceedsAllocation,
    /// The buffer range belongs to a different backend than the descriptor requires.
    BackendMismatch {
        /// Backend required by the descriptor.
        expected: BackendKind,
        /// Backend carried by the memory range.
        actual: BackendKind,
    },
    /// Image or component dimensions must be nonzero.
    ZeroDimension,
    /// Component sampling factors must be nonzero.
    ZeroSampling,
    /// Bit depth must be in the supported 1..=32 range.
    InvalidBitDepth,
    /// Byte stride or element size must be nonzero.
    ZeroByteStride,
    /// Row layout metadata exceeds the resident buffer range.
    LayoutExceedsBuffer,
    /// Codestream byte length exceeds the resident buffer capacity.
    CodestreamExceedsCapacity,
}

impl fmt::Display for ResidentHandoffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRange => f.write_str("resident buffer range is empty"),
            Self::OffsetOverflow => f.write_str("resident buffer range offset overflows"),
            Self::RangeExceedsAllocation => {
                f.write_str("resident buffer range exceeds allocation length")
            }
            Self::BackendMismatch { expected, actual } => write!(
                f,
                "resident buffer backend mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::ZeroDimension => f.write_str("resident component dimensions must be nonzero"),
            Self::ZeroSampling => f.write_str("resident component sampling must be nonzero"),
            Self::InvalidBitDepth => f.write_str("resident sample bit depth must be 1..=32"),
            Self::ZeroByteStride => f.write_str("resident byte stride must be nonzero"),
            Self::LayoutExceedsBuffer => f.write_str("resident row layout exceeds buffer range"),
            Self::CodestreamExceedsCapacity => {
                f.write_str("resident codestream byte length exceeds buffer capacity")
            }
        }
    }
}

impl std::error::Error for ResidentHandoffError {}

/// Borrowed, backend-visible memory range with a caller-owned lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentBufferRef<'a> {
    memory: DeviceMemoryRange,
    _lifetime: PhantomData<&'a ()>,
}

impl ResidentBufferRef<'_> {
    /// Build a borrowed resident buffer reference after validating basic range shape.
    pub fn new(memory: DeviceMemoryRange) -> Result<Self, ResidentHandoffError> {
        if memory.len == 0 {
            return Err(ResidentHandoffError::EmptyRange);
        }
        memory
            .offset
            .checked_add(memory.len)
            .ok_or(ResidentHandoffError::OffsetOverflow)?;
        Ok(Self {
            memory,
            _lifetime: PhantomData,
        })
    }

    /// Build a borrowed resident buffer reference and validate it against an allocation length.
    pub fn with_allocation_len(
        memory: DeviceMemoryRange,
        allocation_len: usize,
    ) -> Result<Self, ResidentHandoffError> {
        let buffer = Self::new(memory)?;
        let end = memory
            .offset
            .checked_add(memory.len)
            .ok_or(ResidentHandoffError::OffsetOverflow)?;
        if end > allocation_len {
            return Err(ResidentHandoffError::RangeExceedsAllocation);
        }
        Ok(buffer)
    }

    /// Return the opaque memory range.
    pub const fn memory_range(&self) -> DeviceMemoryRange {
        self.memory
    }

    /// Backend that owns this resident range.
    pub const fn backend(&self) -> BackendKind {
        self.memory.backend
    }

    /// Byte length of the resident range.
    pub const fn byte_len(&self) -> usize {
        self.memory.len
    }

    fn require_backend(self, backend: BackendKind) -> Result<Self, ResidentHandoffError> {
        if self.memory.backend == backend {
            Ok(self)
        } else {
            Err(ResidentHandoffError::BackendMismatch {
                expected: backend,
                actual: self.memory.backend,
            })
        }
    }
}

/// Component sampling factors preserved through JPEG-to-HTJ2K transcode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentSampling {
    /// JPEG 2000 `XRsiz` horizontal sampling factor.
    pub x_rsiz: u8,
    /// JPEG 2000 `YRsiz` vertical sampling factor.
    pub y_rsiz: u8,
}

impl ResidentSampling {
    /// Build nonzero sampling metadata.
    pub const fn new(x_rsiz: u8, y_rsiz: u8) -> Result<Self, ResidentHandoffError> {
        if x_rsiz == 0 || y_rsiz == 0 {
            return Err(ResidentHandoffError::ZeroSampling);
        }
        Ok(Self { x_rsiz, y_rsiz })
    }
}

/// Sample precision metadata for resident handoff buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentSampleInfo {
    /// Component precision in bits.
    pub bit_depth: u8,
    /// Whether sample or coefficient values are signed.
    pub signed: bool,
}

impl ResidentSampleInfo {
    /// Build sample metadata with a supported bit depth.
    pub const fn new(bit_depth: u8, signed: bool) -> Result<Self, ResidentHandoffError> {
        if bit_depth == 0 || bit_depth > 32 {
            return Err(ResidentHandoffError::InvalidBitDepth);
        }
        Ok(Self { bit_depth, signed })
    }
}

/// Color interpretation carried by a resident transcode handoff buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentColorModel {
    /// Color interpretation is unknown or intentionally deferred.
    Unknown,
    /// Single-component grayscale.
    Grayscale,
    /// RGB-like component ordering.
    Rgb,
    /// JPEG YCbCr/YBR component ordering.
    YCbCr,
    /// RGBA-like component ordering.
    Rgba,
}

/// Per-component geometry shared by resident handoff descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentComponentGeometry {
    /// Component index in source/component order.
    pub component_index: usize,
    /// Native component width in samples.
    pub width: u32,
    /// Native component height in samples.
    pub height: u32,
    /// Component sampling factors.
    pub sampling: ResidentSampling,
}

impl ResidentComponentGeometry {
    /// Build component geometry with nonzero dimensions and sampling.
    pub const fn new(
        component_index: usize,
        width: u32,
        height: u32,
        sampling: ResidentSampling,
    ) -> Result<Self, ResidentHandoffError> {
        if width == 0 || height == 0 {
            return Err(ResidentHandoffError::ZeroDimension);
        }
        Ok(Self {
            component_index,
            width,
            height,
            sampling,
        })
    }
}

/// Coefficient ordering used by a resident JPEG DCT-grid buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentDctCoefficientOrder {
    /// Natural raster order within each 8x8 block.
    Natural,
    /// JPEG zig-zag order within each 8x8 block.
    ZigZag,
}

/// Resident JPEG DCT coefficient grid descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentJpegDctGrid<'a> {
    /// Backend-visible coefficient buffer.
    pub buffer: ResidentBufferRef<'a>,
    /// Component geometry and sampling.
    pub component: ResidentComponentGeometry,
    /// Coefficient precision and signedness.
    pub sample: ResidentSampleInfo,
    /// Color interpretation of the source image/component set.
    pub color: ResidentColorModel,
    /// Padded DCT block columns.
    pub block_cols: u32,
    /// Padded DCT block rows.
    pub block_rows: u32,
    /// Byte stride between consecutive block rows in the resident buffer.
    pub row_pitch_bytes: usize,
    /// Bytes per coefficient in the resident buffer.
    pub bytes_per_coefficient: usize,
    /// Coefficient order within each DCT block.
    pub coefficient_order: ResidentDctCoefficientOrder,
}

impl<'a> ResidentJpegDctGrid<'a> {
    /// Build a resident JPEG DCT-grid descriptor.
    pub fn new(
        buffer: ResidentBufferRef<'a>,
        component: ResidentComponentGeometry,
        sample: ResidentSampleInfo,
        color: ResidentColorModel,
        layout: ResidentDctGridLayout,
    ) -> Result<Self, ResidentHandoffError> {
        if layout.block_cols == 0 || layout.block_rows == 0 {
            return Err(ResidentHandoffError::ZeroDimension);
        }
        if layout.row_pitch_bytes == 0 || layout.bytes_per_coefficient == 0 {
            return Err(ResidentHandoffError::ZeroByteStride);
        }
        let row_coefficients = usize::try_from(layout.block_cols)
            .ok()
            .and_then(|cols| cols.checked_mul(64))
            .ok_or(ResidentHandoffError::OffsetOverflow)?;
        validate_row_layout_fits_buffer(
            buffer,
            row_coefficients,
            layout.block_rows,
            layout.row_pitch_bytes,
            layout.bytes_per_coefficient,
        )?;
        Ok(Self {
            buffer,
            component,
            sample,
            color,
            block_cols: layout.block_cols,
            block_rows: layout.block_rows,
            row_pitch_bytes: layout.row_pitch_bytes,
            bytes_per_coefficient: layout.bytes_per_coefficient,
            coefficient_order: layout.coefficient_order,
        })
    }

    /// Validate this descriptor is backed by the expected backend.
    pub fn require_backend(self, backend: BackendKind) -> Result<Self, ResidentHandoffError> {
        self.buffer.require_backend(backend)?;
        Ok(self)
    }
}

/// Layout metadata for a resident JPEG DCT-grid descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentDctGridLayout {
    /// Padded DCT block columns.
    pub block_cols: u32,
    /// Padded DCT block rows.
    pub block_rows: u32,
    /// Byte stride between consecutive block rows in the resident buffer.
    pub row_pitch_bytes: usize,
    /// Bytes per coefficient in the resident buffer.
    pub bytes_per_coefficient: usize,
    /// Coefficient order within each DCT block.
    pub coefficient_order: ResidentDctCoefficientOrder,
}

/// Wavelet subband represented by a resident DWT buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentDwtSubbandKind {
    /// Low-low subband.
    LowLow,
    /// High-low subband.
    HighLow,
    /// Low-high subband.
    LowHigh,
    /// High-high subband.
    HighHigh,
}

/// Resident projected DWT subband descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentDwtSubband<'a> {
    /// Backend-visible subband buffer.
    pub buffer: ResidentBufferRef<'a>,
    /// Component geometry and sampling.
    pub component: ResidentComponentGeometry,
    /// Coefficient precision and signedness.
    pub sample: ResidentSampleInfo,
    /// Color interpretation of the source image/component set.
    pub color: ResidentColorModel,
    /// DWT decomposition level.
    pub level: u8,
    /// Subband kind within the level.
    pub subband: ResidentDwtSubbandKind,
    /// Native subband width in coefficients.
    pub width: u32,
    /// Native subband height in coefficients.
    pub height: u32,
    /// Byte stride between subband rows.
    pub row_pitch_bytes: usize,
    /// Bytes per coefficient in the resident buffer.
    pub bytes_per_coefficient: usize,
}

impl<'a> ResidentDwtSubband<'a> {
    /// Build a resident DWT subband descriptor.
    pub fn new(
        buffer: ResidentBufferRef<'a>,
        component: ResidentComponentGeometry,
        sample: ResidentSampleInfo,
        color: ResidentColorModel,
        layout: ResidentDwtSubbandLayout,
    ) -> Result<Self, ResidentHandoffError> {
        if layout.width == 0 || layout.height == 0 {
            return Err(ResidentHandoffError::ZeroDimension);
        }
        if layout.row_pitch_bytes == 0 || layout.bytes_per_coefficient == 0 {
            return Err(ResidentHandoffError::ZeroByteStride);
        }
        validate_row_layout_fits_buffer(
            buffer,
            usize::try_from(layout.width).map_err(|_| ResidentHandoffError::OffsetOverflow)?,
            layout.height,
            layout.row_pitch_bytes,
            layout.bytes_per_coefficient,
        )?;
        Ok(Self {
            buffer,
            component,
            sample,
            color,
            level: layout.level,
            subband: layout.subband,
            width: layout.width,
            height: layout.height,
            row_pitch_bytes: layout.row_pitch_bytes,
            bytes_per_coefficient: layout.bytes_per_coefficient,
        })
    }

    /// Validate this descriptor is backed by the expected backend.
    pub fn require_backend(self, backend: BackendKind) -> Result<Self, ResidentHandoffError> {
        self.buffer.require_backend(backend)?;
        Ok(self)
    }
}

fn validate_row_layout_fits_buffer(
    buffer: ResidentBufferRef<'_>,
    row_values: usize,
    rows: u32,
    row_pitch_bytes: usize,
    bytes_per_value: usize,
) -> Result<(), ResidentHandoffError> {
    let row_bytes = row_values
        .checked_mul(bytes_per_value)
        .ok_or(ResidentHandoffError::OffsetOverflow)?;
    if row_pitch_bytes < row_bytes {
        return Err(ResidentHandoffError::LayoutExceedsBuffer);
    }
    let rows = usize::try_from(rows).map_err(|_| ResidentHandoffError::OffsetOverflow)?;
    let last_row_offset = rows
        .saturating_sub(1)
        .checked_mul(row_pitch_bytes)
        .ok_or(ResidentHandoffError::OffsetOverflow)?;
    let required_len = last_row_offset
        .checked_add(row_bytes)
        .ok_or(ResidentHandoffError::OffsetOverflow)?;
    if required_len > buffer.byte_len() {
        return Err(ResidentHandoffError::LayoutExceedsBuffer);
    }
    Ok(())
}

/// Layout metadata for a resident DWT subband descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentDwtSubbandLayout {
    /// DWT decomposition level.
    pub level: u8,
    /// Subband kind within the level.
    pub subband: ResidentDwtSubbandKind,
    /// Native subband width in coefficients.
    pub width: u32,
    /// Native subband height in coefficients.
    pub height: u32,
    /// Byte stride between subband rows.
    pub row_pitch_bytes: usize,
    /// Bytes per coefficient in the resident buffer.
    pub bytes_per_coefficient: usize,
}

/// Resident codestream buffer descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentCodestreamBuffer<'a> {
    /// Backend-visible codestream buffer.
    pub buffer: ResidentBufferRef<'a>,
    /// Number of valid codestream bytes.
    pub byte_len: usize,
    /// Allocated codestream capacity.
    pub capacity: usize,
}

impl<'a> ResidentCodestreamBuffer<'a> {
    /// Build a resident codestream descriptor and validate byte length.
    pub fn new(
        buffer: ResidentBufferRef<'a>,
        byte_len: usize,
        capacity: usize,
    ) -> Result<Self, ResidentHandoffError> {
        if byte_len > capacity || capacity > buffer.byte_len() {
            return Err(ResidentHandoffError::CodestreamExceedsCapacity);
        }
        Ok(Self {
            buffer,
            byte_len,
            capacity,
        })
    }

    /// Validate this descriptor is backed by the expected backend.
    pub fn require_backend(self, backend: BackendKind) -> Result<Self, ResidentHandoffError> {
        self.buffer.require_backend(backend)?;
        Ok(self)
    }
}
