// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

mod compare;
mod encoder;
mod manifest;
mod normalize;
mod pgx;
mod report;
#[cfg(feature = "runner")]
pub mod runner;

pub use compare::{
    compare_peak_samples, compare_samples, Comparison, ComparisonError, ErrorBounds, PeakComparison,
};
pub use encoder::{
    EncoderBlockCoding, EncoderCase, EncoderIcs, EncoderIut, EncoderMarker, EncoderMatrix,
    EncoderMatrixError, EncoderMode, EncoderOperation, EncoderPayload, EncoderProgression,
    EncoderReferenceDecoder,
};
pub use manifest::{
    CorpusFile, DecoderCase, HtAdditionalError, HtBset, HtClaimSet, HtCodestream,
    HtComplianceClass, Jp2Case, JphBset, JphCodestream, ManifestError, Part15CaseMetadata,
    T803Manifest, T803Source, T803Suite,
};
pub use normalize::{normalize_component, Component, NormalizationError, NormalizationTarget};
pub use pgx::{parse_pgx, PgxError, PgxImage};
pub use report::{
    AcceleratorExecutionEvidence, CaseReport, CaseStatus, DecoderRouteSummary, EncodeRouteStage,
    EncodeRouteStageName, EncoderCaseReport, EncoderDispatchEvidence, EncoderEvidence,
    EncoderQualityStatus, EncoderReferenceIdentity, EncoderSupplementalReferenceIdentity,
    ExecutionLocation, HtCodeBlockSetMode, IutIdentity, NativeComponentOracleEvidence,
    NativeHtCoverageAxis, NativeHtCoverageCase, NativeHtCoverageEvidence, Part15CaseEvidence,
    Part15CodestreamEvidence, Part15EvidenceClassification, PlatformIdentity, ReportError,
    ReportStatus, RouteKind, RouteStage, RouteStageName, T803Report,
};
