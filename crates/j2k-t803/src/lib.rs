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
    EncoderCase, EncoderIcs, EncoderIut, EncoderMarker, EncoderMatrix, EncoderMatrixError,
    EncoderMode, EncoderProgression,
};
pub use manifest::{CorpusFile, DecoderCase, Jp2Case, ManifestError, T803Manifest, T803Source};
pub use normalize::{normalize_component, Component, NormalizationError, NormalizationTarget};
pub use pgx::{parse_pgx, PgxError, PgxImage};
pub use report::{
    CaseReport, CaseStatus, DecoderRouteSummary, EncodeRouteStage, EncodeRouteStageName,
    EncoderCaseReport, EncoderEvidence, EncoderQualityStatus, EncoderReferenceIdentity,
    ExecutionLocation, IutIdentity, NativeComponentOracleEvidence, PlatformIdentity, ReportError,
    ReportStatus, RouteKind, RouteStage, RouteStageName, T803Report,
};
