// SPDX-License-Identifier: MIT OR Apache-2.0

/// Integer sample width used by a pixel format or row sink.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SampleType {
    /// Unsigned 8-bit samples.
    U8,
    /// Unsigned 16-bit samples.
    U16,
    /// Signed 16-bit samples.
    I16,
}

/// Supported integer sample type for row-oriented APIs.
pub trait Sample: Copy + Default + Send + Sync + 'static {
    /// Runtime sample type tag.
    const TYPE: SampleType;
    /// Number of significant bits in the sample type.
    const BITS: u8;
}

impl Sample for u8 {
    const TYPE: SampleType = SampleType::U8;
    const BITS: u8 = 8;
}

impl Sample for u16 {
    const TYPE: SampleType = SampleType::U16;
    const BITS: u8 = 16;
}

impl Sample for i16 {
    const TYPE: SampleType = SampleType::I16;
    const BITS: u8 = 16;
}
