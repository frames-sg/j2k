// SPDX-License-Identifier: MIT OR Apache-2.0

//! Validation harness helpers for the experimental JPEG-DCT to HTJ2K path.

use core::fmt;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use j2k_transcode::{
    jpeg_to_htj2k, JpegToHtj2kError, JpegToHtj2kOptions, TranscodeValidationClassification,
};

/// External WSI JPEG roots, separated using the platform path separator.
pub const TRANSCODE_WSI_ROOT_ENV: &str = "J2K_TRANSCODE_WSI_ROOT";
/// Require `TRANSCODE_WSI_ROOT_ENV` to be configured and non-empty.
pub const REQUIRE_TRANSCODE_WSI_ROOT_ENV: &str = "J2K_REQUIRE_TRANSCODE_WSI_ROOT";
/// Maximum number of external WSI tiles to include. `0` means no limit.
pub const TRANSCODE_WSI_TILE_LIMIT_ENV: &str = "J2K_TRANSCODE_WSI_TILE_LIMIT";
/// Maximum accepted external JPEG payload size in bytes.
pub const TRANSCODE_WSI_MAX_PAYLOAD_BYTES_ENV: &str = "J2K_TRANSCODE_WSI_MAX_PAYLOAD_BYTES";

const DEFAULT_EXTERNAL_TILE_LIMIT: usize = 8;
const DEFAULT_MAX_EXTERNAL_PAYLOAD_BYTES: u64 = 67_108_864;

/// Borrowed JPEG fixture for corpus validation.
#[derive(Debug, Clone, Copy)]
pub struct CorpusFixture<'a> {
    /// Human-readable fixture name used in failure messages and reports.
    pub name: &'a str,
    /// JPEG codestream bytes.
    pub bytes: &'a [u8],
}

/// Owned JPEG fixture loaded from a local corpus path.
#[derive(Debug, Clone)]
pub struct OwnedCorpusFixture {
    /// Human-readable fixture name, usually a path.
    pub name: String,
    /// JPEG codestream bytes.
    pub bytes: Vec<u8>,
}

impl OwnedCorpusFixture {
    /// Borrow this owned fixture for validation.
    #[must_use]
    pub fn as_fixture(&self) -> CorpusFixture<'_> {
        CorpusFixture {
            name: &self.name,
            bytes: &self.bytes,
        }
    }
}

/// Options for deterministic and optional external corpus validation.
#[derive(Debug, Clone)]
pub struct CorpusValidationOptions {
    /// Options passed to `jpeg_to_htj2k`. Validation metrics are enabled by the
    /// corpus harness regardless of this value.
    pub transcode_options: JpegToHtj2kOptions,
    /// Optional local roots containing extracted WSI JPEG tiles.
    pub external_wsi_roots: Vec<PathBuf>,
    /// Whether missing or empty external roots are hard failures.
    pub require_external_wsi: bool,
    /// Maximum number of external JPEG tiles to load. `0` means no limit.
    pub external_tile_limit: usize,
    /// Maximum accepted external JPEG payload size in bytes.
    pub max_external_payload_bytes: u64,
}

impl Default for CorpusValidationOptions {
    fn default() -> Self {
        Self {
            transcode_options: JpegToHtj2kOptions::default(),
            external_wsi_roots: Vec::new(),
            require_external_wsi: false,
            external_tile_limit: DEFAULT_EXTERNAL_TILE_LIMIT,
            max_external_payload_bytes: DEFAULT_MAX_EXTERNAL_PAYLOAD_BYTES,
        }
    }
}

impl CorpusValidationOptions {
    /// Build options from the opt-in external corpus environment variables.
    #[must_use]
    pub fn from_env() -> Self {
        let mut options = Self::default();
        if let Some(roots) = std::env::var_os(TRANSCODE_WSI_ROOT_ENV) {
            options.external_wsi_roots = std::env::split_paths(&roots).collect();
        }
        options.require_external_wsi = std::env::var_os(REQUIRE_TRANSCODE_WSI_ROOT_ENV).is_some();
        if let Ok(limit) = std::env::var(TRANSCODE_WSI_TILE_LIMIT_ENV) {
            if let Ok(limit) = limit.parse() {
                options.external_tile_limit = limit;
            }
        }
        if let Ok(max_bytes) = std::env::var(TRANSCODE_WSI_MAX_PAYLOAD_BYTES_ENV) {
            if let Ok(max_bytes) = max_bytes.parse() {
                options.max_external_payload_bytes = max_bytes;
            }
        }
        options
    }
}

/// Aggregate validation report across all tested fixtures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusValidationReport {
    /// Number of JPEG fixtures validated.
    pub fixture_count: usize,
    /// Number of compared wavelet coefficients.
    pub sample_count: usize,
    /// Number of coefficients matching the float oracle exactly after
    /// rounding.
    pub exact_match_count: usize,
    /// Maximum absolute rounded-coefficient error.
    pub max_abs_error: i64,
    /// Threshold classification for the aggregate corpus metrics.
    pub classification: TranscodeValidationClassification,
    /// Absolute-error histogram keyed by LSB distance.
    pub histogram_buckets: BTreeMap<i64, usize>,
    /// Per-fixture summaries.
    pub fixtures: Vec<CorpusFixtureReport>,
}

/// Validation summary for one JPEG fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusFixtureReport {
    /// Fixture name.
    pub name: String,
    /// Source reference-grid dimensions.
    pub dimensions: (u32, u32),
    /// Number of JPEG/HTJ2K components.
    pub component_count: usize,
    /// Compared coefficient count.
    pub sample_count: usize,
    /// Exact coefficient matches after rounding.
    pub exact_match_count: usize,
    /// Maximum absolute rounded-coefficient error.
    pub max_abs_error: i64,
    /// Threshold classification for this fixture's integer-reference metrics.
    pub classification: TranscodeValidationClassification,
}

/// Corpus validation failure.
#[derive(Debug)]
pub enum CorpusValidationError {
    /// No fixtures were provided to the deterministic validation pass.
    EmptyCorpus,
    /// External WSI roots were required but not configured or empty.
    MissingRequiredExternalCorpus(&'static str),
    /// Reading an external fixture failed.
    Io {
        /// Path that failed.
        path: PathBuf,
        /// Source IO error.
        source: std::io::Error,
    },
    /// Transcoding one fixture failed.
    Transcode {
        /// Fixture name.
        name: String,
        /// Source transcode error.
        source: JpegToHtj2kError,
    },
    /// The transcode completed without the required validation metrics.
    MissingMetrics {
        /// Fixture name.
        name: String,
    },
}

impl fmt::Display for CorpusValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCorpus => write!(f, "corpus validation requires at least one fixture"),
            Self::MissingRequiredExternalCorpus(reason) => {
                write!(f, "required external corpus is unavailable: {reason}")
            }
            Self::Io { path, source } => {
                write!(
                    f,
                    "failed to read external fixture {}: {source}",
                    path.display()
                )
            }
            Self::Transcode { name, source } => {
                write!(f, "failed to transcode fixture {name}: {source}")
            }
            Self::MissingMetrics { name } => {
                write!(f, "fixture {name} did not report validation metrics")
            }
        }
    }
}

impl std::error::Error for CorpusValidationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Transcode { source, .. } => Some(source),
            Self::EmptyCorpus
            | Self::MissingRequiredExternalCorpus(_)
            | Self::MissingMetrics { .. } => None,
        }
    }
}

/// Validate a deterministic set of JPEG fixtures against the integer
/// ISLOW-IDCT-then-reversible-5/3 oracle and aggregate coefficient error
/// metrics.
pub fn validate_transcode_corpus(
    fixtures: &[CorpusFixture<'_>],
    options: &CorpusValidationOptions,
) -> Result<CorpusValidationReport, CorpusValidationError> {
    if fixtures.is_empty() {
        return Err(CorpusValidationError::EmptyCorpus);
    }

    let mut report = CorpusValidationReport {
        fixture_count: 0,
        sample_count: 0,
        exact_match_count: 0,
        max_abs_error: 0,
        classification: TranscodeValidationClassification::Exact,
        histogram_buckets: BTreeMap::new(),
        fixtures: Vec::with_capacity(fixtures.len()),
    };

    for fixture in fixtures.iter().copied() {
        let validated = validate_fixture(fixture, options)?;
        report.fixture_count += 1;
        report.sample_count += validated.report.sample_count;
        report.exact_match_count += validated.report.exact_match_count;
        report.max_abs_error = report.max_abs_error.max(validated.report.max_abs_error);
        for (error, count) in validated.histogram_buckets {
            *report.histogram_buckets.entry(error).or_insert(0) += count;
        }
        report.fixtures.push(validated.report);
    }
    report.classification = classify_corpus_report(&report);

    Ok(report)
}

struct ValidatedFixture {
    report: CorpusFixtureReport,
    histogram_buckets: BTreeMap<i64, usize>,
}

fn validate_fixture(
    fixture: CorpusFixture<'_>,
    options: &CorpusValidationOptions,
) -> Result<ValidatedFixture, CorpusValidationError> {
    let mut transcode_options = options.transcode_options.clone();
    transcode_options.validate_against_integer_reference = true;
    let encoded = jpeg_to_htj2k(fixture.bytes, &transcode_options).map_err(|source| {
        CorpusValidationError::Transcode {
            name: fixture.name.to_string(),
            source,
        }
    })?;
    let metrics = encoded
        .report
        .integer_reference_metrics
        .as_ref()
        .ok_or_else(|| CorpusValidationError::MissingMetrics {
            name: fixture.name.to_string(),
        })?;

    Ok(ValidatedFixture {
        report: CorpusFixtureReport {
            name: fixture.name.to_string(),
            dimensions: (encoded.report.width, encoded.report.height),
            component_count: encoded.report.component_count,
            sample_count: metrics.total,
            exact_match_count: metrics.exact_matches,
            max_abs_error: metrics.max_abs_error,
            classification: TranscodeValidationClassification::classify_metrics(metrics),
        },
        histogram_buckets: metrics.absolute_error_histogram.clone(),
    })
}

fn classify_corpus_report(report: &CorpusValidationReport) -> TranscodeValidationClassification {
    if report.sample_count == 0 {
        return TranscodeValidationClassification::Exact;
    }
    if report.exact_match_count == report.sample_count && report.max_abs_error == 0 {
        TranscodeValidationClassification::Exact
    } else {
        let exact_match_rate = report.exact_match_count as f64 / report.sample_count as f64;
        if report.max_abs_error <= 1 && exact_match_rate >= 0.999 {
            TranscodeValidationClassification::OneLsbBounded
        } else {
            TranscodeValidationClassification::OutsideThreshold
        }
    }
}

/// Load optional external WSI JPEG fixtures from `options.external_wsi_roots`.
///
/// Normal CI should leave the root list empty. Signoff hosts can set
/// `J2K_TRANSCODE_WSI_ROOT` and build options with
/// [`CorpusValidationOptions::from_env`].
pub fn load_external_wsi_fixtures(
    options: &CorpusValidationOptions,
) -> Result<Vec<OwnedCorpusFixture>, CorpusValidationError> {
    if options.external_wsi_roots.is_empty() {
        if options.require_external_wsi {
            return Err(CorpusValidationError::MissingRequiredExternalCorpus(
                "no roots configured",
            ));
        }
        return Ok(Vec::new());
    }

    let mut paths = Vec::new();
    for root in &options.external_wsi_roots {
        collect_jpegs(root, &mut paths);
    }
    paths.sort();
    if paths.is_empty() && options.require_external_wsi {
        return Err(CorpusValidationError::MissingRequiredExternalCorpus(
            "configured roots contained no JPEG files",
        ));
    }

    let limit = if options.external_tile_limit == 0 {
        usize::MAX
    } else {
        options.external_tile_limit
    };
    let mut fixtures = Vec::new();
    for path in paths.into_iter().take(limit) {
        let metadata = fs::metadata(&path).map_err(|source| CorpusValidationError::Io {
            path: path.clone(),
            source,
        })?;
        if metadata.len() > options.max_external_payload_bytes {
            continue;
        }
        let bytes = fs::read(&path).map_err(|source| CorpusValidationError::Io {
            path: path.clone(),
            source,
        })?;
        fixtures.push(OwnedCorpusFixture {
            name: path.display().to_string(),
            bytes,
        });
    }

    Ok(fixtures)
}

fn collect_jpegs(root: &Path, out: &mut Vec<PathBuf>) {
    if root.is_file() {
        if is_jpeg_path(root) {
            out.push(root.to_path_buf());
        }
        return;
    }
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_jpegs(&path, out);
        } else if is_jpeg_path(&path) {
            out.push(path);
        }
    }
}

fn is_jpeg_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg"))
}
