// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// A pinned external workload manifest for Auto-routing benchmarks.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AutoRoutingManifest {
    pub schema_version: u32,
    pub corpus: String,
    pub source_url: String,
    pub cases: Vec<AutoRoutingManifestCase>,
}

/// One hash-pinned input in an Auto-routing workload manifest.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AutoRoutingManifestCase {
    pub id: String,
    pub path: String,
    pub kind: AutoRoutingWorkloadKind,
    pub codec: Option<AutoRoutingCodec>,
    pub container: Option<AutoRoutingContainer>,
    pub pixel_format: AutoRoutingPixelFormat,
    pub sha256: String,
}

/// JPEG 2000 coding system exercised by one routing workload.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum AutoRoutingCodec {
    #[serde(rename = "jpeg-2000-part-1")]
    Jpeg2000Part1,
    #[serde(rename = "htj2k-part-15")]
    Htj2kPart15,
}

impl AutoRoutingCodec {
    /// Whether this workload uses Part 15 high-throughput block coding.
    #[must_use]
    pub const fn is_high_throughput(self) -> bool {
        matches!(self, Self::Htj2kPart15)
    }
}

/// Compressed payload shape produced or consumed by one routing workload.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AutoRoutingContainer {
    Codestream,
    Jp2,
    Jph,
}

/// Whether a workload is a compressed decode input or an uncompressed encode input.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AutoRoutingWorkloadKind {
    Decode,
    Encode,
}

/// Pixel layout used for route-parity comparisons.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AutoRoutingPixelFormat {
    Gray8,
    Rgb8,
}

/// One validated, in-memory external workload.
#[derive(Clone, Debug)]
pub struct AutoRoutingWorkload {
    pub id: String,
    pub path: PathBuf,
    pub kind: AutoRoutingWorkloadKind,
    pub codec: AutoRoutingCodec,
    pub container: AutoRoutingContainer,
    pub pixel_format: AutoRoutingPixelFormat,
    pub bytes: Vec<u8>,
}

/// A validated manifest, its exact hash, and the inputs it names.
#[derive(Clone, Debug)]
pub struct AutoRoutingWorkloadSet {
    pub manifest: AutoRoutingManifest,
    pub manifest_sha256: String,
    pub workloads: Vec<AutoRoutingWorkload>,
}

/// Validated 8-bit PGM/PPM input for an encode benchmark cell.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AutoRoutingPnm {
    pub id: String,
    pub codec: AutoRoutingCodec,
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub components: u16,
}

/// Accelerator lane that produced route evidence.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AutoRoutingBackend {
    Cuda,
    Metal,
}

/// Hardware and software identity for one benchmark lane.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AutoRoutingPlatform {
    pub os: String,
    pub arch: String,
    pub hardware: String,
    pub driver: String,
}

/// Workload class evaluated for a fixed Auto-routing decision.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AutoRoutingOperation {
    FullDecode,
    RoiDecode,
    ScaledDecode,
    BatchDecode,
    LosslessEncode,
    LossyEncode,
}

/// Actual execution class of a measured route.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AutoRoutingExecution {
    Cpu,
    Hybrid,
    DeviceNative,
}

/// Criterion result identity and exact output produced by one route.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AutoRoutingRoute {
    pub criterion_id: String,
    pub execution: AutoRoutingExecution,
    pub output_sha256: String,
}

/// CPU, hybrid, and optional device-native measurements for one workload class.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AutoRoutingCell {
    pub id: String,
    pub operation: AutoRoutingOperation,
    pub source: String,
    pub workload: String,
    pub cpu: AutoRoutingRoute,
    pub hybrid: AutoRoutingRoute,
    pub strict_device_supported: bool,
    pub strict_device: Option<AutoRoutingRoute>,
}

/// Versioned route evidence emitted beside Criterion estimates.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AutoRoutingEvidence {
    pub schema_version: u32,
    pub candidate_sha: String,
    pub backend: AutoRoutingBackend,
    pub platform: AutoRoutingPlatform,
    pub external_manifest_sha256: String,
    pub external_case_count: usize,
    pub cells: Vec<AutoRoutingCell>,
}
