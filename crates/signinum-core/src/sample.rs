// SPDX-License-Identifier: Apache-2.0

/// Scalar sample storage type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SampleType {
    /// Unsigned 8-bit sample.
    U8,
    /// Unsigned 16-bit sample.
    U16,
}

/// Marker trait for sample types accepted by typed row-streaming APIs.
pub trait Sample: Copy + Default + Send + Sync + 'static {
    /// Runtime sample type identifier.
    const TYPE: SampleType;
    /// Number of significant bits in the storage type.
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
