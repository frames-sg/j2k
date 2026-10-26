// SPDX-License-Identifier: MIT OR Apache-2.0

//! Thin Burn tensor adapter for JPEG, JPEG 2000, and HTJ2K batch codecs.
//!
//! The codec crates own parsing, grouping, decoding, and accelerator execution.
//! This crate materializes CPU groups, decodes CUDA groups directly into
//! Burn-owned device storage, or stages completed CUDA and Metal codec output
//! through host memory before an ordinary Burn upload. Casting and normalization
//! stay in ordinary Burn tensor operations after decode.
//!
//! JPEG tensor output remains RGB uint8 by default, with grayscale expanded to
//! three identical channels. Grayscale JPEG 2000 and HTJ2K remain one channel.
//! Output metadata describes the returned tensor. A later explicit
//! `channels="native"` option may preserve grayscale JPEG as one channel without
//! changing existing defaults. Source component counts, when exposed, are per
//! input row and remain aligned with `source_indices`.

#![deny(missing_docs)]

use burn_core::tensor::{backend::Backend, Int, Tensor};
use j2k::{BatchAlpha, BatchColor, BatchGroupInfo, BatchLayout, J2kDecodeWarning, Rect};

#[cfg(any(
    feature = "cpu",
    feature = "cuda",
    all(feature = "metal", target_os = "macos")
))]
mod batch_contract;
#[cfg(test)]
mod completion;
mod error;
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
mod staging;

#[cfg(feature = "cpu")]
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "cpu")]
pub use cpu::CpuBurnDecoder;
#[cfg(feature = "cuda")]
pub use cuda::{
    CudaBurnDecoder, CudaDirectBurnDecoder, CudaUploadBurnDecoder, SubmittedCudaBurnBatch,
    SubmittedCudaUploadBurnBatch,
};
pub use error::BurnDecodeError;
#[cfg(feature = "metal")]
pub use metal::{
    MetalBurnDecoder, MetalUploadBurnDecoder, SubmittedMetalBurnBatch,
    SubmittedMetalUploadBurnBatch,
};

/// Ordinary rank-4 Burn integer tensor tagged with its exact codec sample type.
#[derive(Debug)]
#[non_exhaustive]
pub enum BurnBatchTensor<B: Backend> {
    /// Unsigned samples with precision at most eight bits.
    U8(Tensor<B, 4, Int>),
    /// Unsigned samples with precision from nine through sixteen bits.
    U16(Tensor<B, 4, Int>),
    /// Signed samples with precision at most sixteen bits.
    I16(Tensor<B, 4, Int>),
}

impl<B: Backend> BurnBatchTensor<B> {
    /// Borrow the ordinary Burn integer tensor regardless of its codec dtype tag.
    #[must_use]
    pub const fn tensor(&self) -> &Tensor<B, 4, Int> {
        match self {
            Self::U8(tensor) | Self::U16(tensor) | Self::I16(tensor) => tensor,
        }
    }

    /// Consume the codec dtype tag and return the ordinary Burn integer tensor.
    #[must_use]
    pub fn into_tensor(self) -> Tensor<B, 4, Int> {
        match self {
            Self::U8(tensor) | Self::U16(tensor) | Self::I16(tensor) => tensor,
        }
    }
}

/// Borrowed complete compressed input for a mixed CPU batch.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum ImageInput<'a> {
    /// JPEG bytes and interpretation hints.
    #[cfg(feature = "cpu")]
    Jpeg(j2k_jpeg::JpegBatchInput<'a>),
    /// JPEG 2000 or HTJ2K bytes and decode request.
    Jpeg2000(&'a j2k::EncodedImage),
}

/// Codec-tagged per-image decode failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ImageDecodeError {
    /// JPEG parse or decode failure.
    #[cfg(feature = "cpu")]
    #[error("JPEG decode failed: {0}")]
    Jpeg(#[from] j2k_jpeg::JpegError),
    /// JPEG 2000 or HTJ2K parse or decode failure.
    #[error("JPEG 2000 decode failed: {0}")]
    Jpeg2000(#[from] j2k::BatchItemError),
}

/// Codec-tagged diagnostic for one successful tensor row.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ImageDecodeWarning {
    /// JPEG diagnostic.
    #[cfg(feature = "cpu")]
    Jpeg(j2k_jpeg::Warning),
    /// JPEG 2000 or HTJ2K diagnostic.
    Jpeg2000(J2kDecodeWarning),
}

#[derive(Debug)]
pub(crate) enum GroupInfo {
    Jpeg2000(BatchGroupInfo),
    #[cfg(feature = "cpu")]
    Jpeg {
        dimensions: (u32, u32),
        layout: BatchLayout,
    },
}

/// One homogeneous decoded tensor group and its returned sample contract.
///
/// Group order is unspecified. Source indices are authoritative for label
/// alignment. JPEG is RGB8, including expanded grayscale; grayscale JPEG 2000
/// and HTJ2K remain one channel.
#[derive(Debug)]
pub struct BurnBatchGroup<B: Backend> {
    pub(crate) tensor: BurnBatchTensor<B>,
    pub(crate) info: GroupInfo,
    pub(crate) source_indices: Vec<usize>,
    pub(crate) decoded_rects: Vec<Rect>,
    pub(crate) warnings: Vec<Vec<ImageDecodeWarning>>,
    pub(crate) recovered: Vec<bool>,
}

impl<B: Backend> BurnBatchGroup<B> {
    /// Borrow the exact integer tensor.
    #[must_use]
    pub const fn tensor(&self) -> &BurnBatchTensor<B> {
        &self.tensor
    }
    /// Consume the group and return its exact integer tensor.
    #[must_use]
    pub fn into_tensor(self) -> BurnBatchTensor<B> {
        self.tensor
    }
    /// Decoded width and height.
    #[must_use]
    pub const fn dimensions(&self) -> (u32, u32) {
        match &self.info {
            GroupInfo::Jpeg2000(info) => info.dimensions,
            #[cfg(feature = "cpu")]
            GroupInfo::Jpeg { dimensions, .. } => *dimensions,
        }
    }
    /// Dense tensor ordering.
    #[must_use]
    pub const fn layout(&self) -> BatchLayout {
        match &self.info {
            GroupInfo::Jpeg2000(info) => info.layout,
            #[cfg(feature = "cpu")]
            GroupInfo::Jpeg { layout, .. } => *layout,
        }
    }
    /// Returned channel interpretation.
    #[must_use]
    pub const fn color(&self) -> BatchColor {
        match &self.info {
            GroupInfo::Jpeg2000(info) => info.color,
            #[cfg(feature = "cpu")]
            GroupInfo::Jpeg { .. } => BatchColor::Rgb,
        }
    }
    /// Returned alpha interpretation.
    #[must_use]
    pub const fn alpha(&self) -> BatchAlpha {
        match &self.info {
            GroupInfo::Jpeg2000(info) => info.alpha,
            #[cfg(feature = "cpu")]
            GroupInfo::Jpeg { .. } => BatchAlpha::None,
        }
    }
    /// Significant sample precision.
    #[must_use]
    pub const fn precision(&self) -> u8 {
        match &self.info {
            GroupInfo::Jpeg2000(info) => info.precision,
            #[cfg(feature = "cpu")]
            GroupInfo::Jpeg { .. } => 8,
        }
    }
    /// Whether returned samples are signed.
    #[must_use]
    pub const fn signed(&self) -> bool {
        match &self.info {
            GroupInfo::Jpeg2000(info) => info.signed,
            #[cfg(feature = "cpu")]
            GroupInfo::Jpeg { .. } => false,
        }
    }
    /// Original input indices, aligned with tensor rows.
    #[must_use]
    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }
    /// Actual decoded source rectangles, aligned with tensor rows.
    #[must_use]
    pub fn decoded_rects(&self) -> &[Rect] {
        &self.decoded_rects
    }
    /// Typed per-row diagnostics.
    #[must_use]
    pub fn warnings(&self) -> &[Vec<ImageDecodeWarning>] {
        &self.warnings
    }
    /// Rows accepted using lenient recovery or unverified completion.
    #[must_use]
    pub fn recovered(&self) -> &[bool] {
        &self.recovered
    }
    /// Detailed JPEG 2000 metadata, absent for JPEG rows.
    #[must_use]
    pub const fn j2k_info(&self) -> Option<&BatchGroupInfo> {
        match &self.info {
            GroupInfo::Jpeg2000(info) => Some(info),
            #[cfg(feature = "cpu")]
            GroupInfo::Jpeg { .. } => None,
        }
    }
}

#[cfg(any(
    feature = "cpu",
    feature = "cuda",
    all(feature = "metal", target_os = "macos")
))]
pub(crate) fn unrecovered(count: usize) -> Result<Vec<bool>, BurnDecodeError> {
    Ok(
        j2k_core::BatchAllocationBudget::new("Burn recovery flags").try_filled(
            count,
            false,
            "Burn recovery flags",
        )?,
    )
}

#[cfg(any(
    feature = "cpu",
    feature = "cuda",
    all(feature = "metal", target_os = "macos")
))]
pub(crate) fn tagged_warnings<W>(
    warnings: Vec<Vec<W>>,
    tag: impl Fn(W) -> ImageDecodeWarning,
) -> Result<Vec<Vec<ImageDecodeWarning>>, BurnDecodeError> {
    let mut budget = j2k_core::BatchAllocationBudget::new("Burn warning metadata");
    let mut output = budget.try_vec(warnings.len(), "Burn warning rows")?;
    for warnings in warnings {
        let mut row = budget.try_vec(warnings.len(), "Burn row warnings")?;
        row.extend(warnings.into_iter().map(&tag));
        output.push(row);
    }
    Ok(output)
}

#[cfg(any(
    feature = "cpu",
    feature = "cuda",
    all(feature = "metal", target_os = "macos")
))]
pub(crate) fn j2k_errors(
    errors: Vec<j2k::IndexedBatchError>,
) -> Result<Vec<j2k_core::TileBatchError<ImageDecodeError>>, BurnDecodeError> {
    let mut output = j2k_core::BatchAllocationBudget::new("Burn errors")
        .try_vec(errors.len(), "Burn indexed errors")?;
    output.extend(errors.into_iter().map(|error| j2k_core::TileBatchError {
        index: error.index,
        source: ImageDecodeError::Jpeg2000(error.source),
    }));
    Ok(output)
}

/// Failure while submitting or completing one homogeneous Burn tensor group.
///
/// No partially written tensor from the affected group is exposed. Other
/// homogeneous groups may still succeed when the retained codec and framework
/// sessions remain usable.
#[derive(Debug, thiserror::Error)]
#[error("Burn batch group containing source indices {source_indices:?} failed: {source}")]
pub struct BurnBatchGroupError {
    source_indices: Vec<usize>,
    #[source]
    source: BurnDecodeError,
}

impl BurnBatchGroupError {
    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos"), test))]
    pub(crate) fn new(source_indices: Vec<usize>, source: BurnDecodeError) -> Self {
        Self {
            source_indices,
            source,
        }
    }

    /// Original input indices whose dense tensor group was discarded.
    #[must_use]
    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }

    /// Structured codec, framework, or interop failure for this group.
    #[must_use]
    pub const fn source(&self) -> &BurnDecodeError {
        &self.source
    }

    /// Consume the group failure into affected indices and its source.
    #[must_use]
    pub fn into_parts(self) -> (Vec<usize>, BurnDecodeError) {
        (self.source_indices, self.source)
    }
}

/// Successful tensor groups plus indexed preparation and homogeneous execution failures.
#[derive(Debug)]
pub struct BurnBatchDecode<B: Backend> {
    /// Successfully decoded homogeneous tensor groups.
    pub(crate) groups: Vec<BurnBatchGroup<B>>,
    /// Structured preparation or decode failures in original input order.
    pub(crate) errors: Vec<j2k_core::TileBatchError<ImageDecodeError>>,
    /// Homogeneous groups discarded after adapter submission or completion failed.
    pub(crate) group_errors: Vec<BurnBatchGroupError>,
}

impl<B: Backend> BurnBatchDecode<B> {
    /// Successful groups in unspecified order; indices identify their rows.
    #[must_use]
    pub fn groups(&self) -> &[BurnBatchGroup<B>] {
        &self.groups
    }
    /// Indexed per-input failures in original input order.
    #[must_use]
    pub fn errors(&self) -> &[j2k_core::TileBatchError<ImageDecodeError>] {
        &self.errors
    }
    /// Groups discarded after submission or completion failed.
    #[must_use]
    pub fn group_errors(&self) -> &[BurnBatchGroupError] {
        &self.group_errors
    }
    /// Consume the result into its successful groups.
    #[must_use]
    pub fn into_groups(self) -> Vec<BurnBatchGroup<B>> {
        self.groups
    }
}
