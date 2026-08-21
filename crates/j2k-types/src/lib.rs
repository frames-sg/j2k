// j2k-coverage: shared-accelerator-host
//! Shared JPEG 2000 and HTJ2K codec contracts for j2k.
//!
//! Stable codec value types are grouped by transform, Tier-1, packetization,
//! and prepared-plan ownership. The crate's dispatch contracts form the
//! explicit low-level integration surface for encode-stage accelerators.

#![no_std]
#![forbid(unsafe_code)]
#![forbid(missing_docs)]
extern crate alloc;

mod decode_payload;
pub use decode_payload::{
    HtCodeBlockPayloadRanges, J2kClassicCodeBlockPayload, J2kCodestreamRange,
};

mod decode_plan;
pub use decode_plan::{
    DecodePlanAllocationError, HtOwnedCodeBlockBatchJob, HtOwnedSubBandPlan, J2kDirectBandId,
    J2kDirectColorPlan, J2kDirectGrayscalePlan, J2kDirectGrayscaleStep, J2kDirectIdwtStep,
    J2kDirectRgbaPlan, J2kDirectStoreStep, J2kOwnedCodeBlockBatchJob, J2kOwnedSubBandPlan, J2kRect,
    J2kReferencedClassicPlan, J2kReferencedHtj2kPlan, J2kReferencedImageGeometry,
    J2kReferencedPayloadRecordSpan, J2kReferencedTileGeometry, J2kReferencedTilePlan,
    J2kWaveletTransform, DEFAULT_MAX_CODEC_BYTES, DEFAULT_MAX_DECODE_BYTES,
};

mod dispatch;
pub use dispatch::{
    CpuOnlyJ2kEncodeStageAccelerator, J2kDeinterleaveMctToF32Job, J2kDeinterleaveToF32Job,
    J2kEncodeContext, J2kEncodeDispatchReport, J2kEncodeStageAccelerator, J2kEncodeStageError,
    J2kEncodeStageErrorKind, J2kEncodeStageResult, J2kHtj2kTileEncodeJob,
};

#[doc(hidden)]
pub mod encode_geometry;

/// Sort packet descriptors in the requested progression order.
pub fn sort_packet_descriptors_for_progression(
    descriptors: &mut [J2kPacketizationPacketDescriptor],
    progression_order: J2kPacketizationProgressionOrder,
) {
    encode_geometry::sort_packet_descriptors_for_progression(descriptors, progression_order);
}

mod limits;
#[doc(hidden)]
pub use limits::{MAX_JPEG2000_PART1_COMPONENTS, MAX_JPEG2000_PART1_SAMPLE_BIT_DEPTH};
mod move_only;

mod packetization;
pub use packetization::{
    J2kPacketizationBlockCodingMode, J2kPacketizationCodeBlock, J2kPacketizationEncodeJob,
    J2kPacketizationPacketDescriptor, J2kPacketizationProgressionOrder, J2kPacketizationResolution,
    J2kPacketizationSubband,
};

mod prepared_plan;
pub use prepared_plan::{
    PrecomputedHtj2k53Component, PrecomputedHtj2k53Image, PrecomputedHtj2k97Component,
    PrecomputedHtj2k97Image, PreencodedHtj2k97CodeBlock, PreencodedHtj2k97CompactCodeBlock,
    PreencodedHtj2k97CompactComponent, PreencodedHtj2k97CompactImage,
    PreencodedHtj2k97CompactResolution, PreencodedHtj2k97CompactSubband,
    PreencodedHtj2k97Component, PreencodedHtj2k97Image, PreencodedHtj2k97Resolution,
    PreencodedHtj2k97Subband, PrequantizedHtj2k97CodeBlock, PrequantizedHtj2k97Component,
    PrequantizedHtj2k97Image, PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband,
};

mod resident;
#[doc(hidden)]
pub use resident::{
    J2kResidentEncodeInput, J2kResidentEncodeInputError, J2kResidentHtj2kTileEncodeJob,
};
mod tier1;
pub use tier1::{
    EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock, J2kCodeBlockSegment, J2kCodeBlockStyle,
    J2kHtCodeBlockEncodeJob, J2kHtSubbandEncodeJob, J2kSubBandType, J2kTier1CodeBlockEncodeJob,
};

mod transform;
pub use transform::{
    IrreversibleQuantizationStep, IrreversibleQuantizationSubbandScales, J2kForwardDwt53Job,
    J2kForwardDwt53Level, J2kForwardDwt53Output, J2kForwardDwt97Job, J2kForwardDwt97Level,
    J2kForwardDwt97Output, J2kForwardIctJob, J2kForwardRctJob, J2kQuantizeSubbandJob,
};
