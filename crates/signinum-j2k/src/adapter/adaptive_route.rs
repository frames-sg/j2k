// SPDX-License-Identifier: Apache-2.0

//! Adaptive JPEG 2000 / HTJ2K CPU-device route planning.

use alloc::vec::Vec;

use crate::J2kError;
use signinum_core::{BackendCapabilities, BackendKind, BackendRequest, Unsupported};

/// Caller intent for adaptive JPEG 2000-family routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum J2kAdaptiveBackendRequest {
    /// Use the best benchmark-approved CPU/device split available on the host.
    Accelerated,
    /// Force all stages onto the portable CPU route.
    CpuOnly,
    /// Require proof of the requested device path and fail if unavailable.
    StrictDevice(BackendKind),
}

impl J2kAdaptiveBackendRequest {
    /// Convert a shared backend request into adaptive JPEG 2000 route intent.
    #[must_use]
    pub const fn from_backend_request(request: BackendRequest) -> Self {
        match request {
            BackendRequest::Auto => Self::Accelerated,
            BackendRequest::Cpu => Self::CpuOnly,
            BackendRequest::Metal => Self::StrictDevice(BackendKind::Metal),
            BackendRequest::Cuda => Self::StrictDevice(BackendKind::Cuda),
        }
    }
}

/// High-level operation being routed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum J2kAdaptiveOperation {
    /// JPEG 2000-family encode.
    Encode,
    /// JPEG 2000-family decode.
    Decode,
    /// JPEG 2000-family transcode or recode.
    Transcode,
}

/// Codestream family being routed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum J2kAdaptiveCodecMode {
    /// Classic JPEG 2000 Part 1 block coding.
    ClassicJ2k,
    /// High-throughput JPEG 2000 Part 15 block coding.
    Htj2k,
}

/// Quality mode being routed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum J2kAdaptiveQualityMode {
    /// Reversible/lossless path.
    Lossless,
    /// Irreversible/lossy path.
    Lossy,
}

/// Desired ownership of produced buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum J2kAdaptiveOutputResidency {
    /// Return host-visible output.
    Host,
    /// Keep output resident on the selected device when the adapter supports it.
    Device,
}

/// One JPEG 2000-family workload shape used by the adaptive planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct J2kAdaptiveWorkload {
    /// Operation being routed.
    pub operation: J2kAdaptiveOperation,
    /// Classic J2K versus HTJ2K mode.
    pub codec_mode: J2kAdaptiveCodecMode,
    /// Lossless versus lossy mode.
    pub quality_mode: J2kAdaptiveQualityMode,
    /// Number of image components.
    pub components: u8,
    /// Significant bits per component sample.
    pub bit_depth: u8,
    /// Tile dimensions in pixels.
    pub tile_size: (u32, u32),
    /// Number of same-shaped tiles or frames in the route.
    pub batch_size: u16,
    /// Whether this route decodes or transcodes a source ROI.
    pub roi: bool,
    /// Whether this route decodes or transcodes at reduced resolution.
    pub scaled: bool,
    /// Number of cumulative quality layers requested or present.
    pub quality_layers: u16,
    /// Desired output residency.
    pub output_residency: J2kAdaptiveOutputResidency,
}

impl J2kAdaptiveWorkload {
    /// Build a workload with full-frame host output and one quality layer.
    #[must_use]
    pub const fn new(
        operation: J2kAdaptiveOperation,
        codec_mode: J2kAdaptiveCodecMode,
        quality_mode: J2kAdaptiveQualityMode,
        components: u8,
        bit_depth: u8,
        tile_size: (u32, u32),
        batch_size: u16,
    ) -> Self {
        Self {
            operation,
            codec_mode,
            quality_mode,
            components,
            bit_depth,
            tile_size,
            batch_size,
            roi: false,
            scaled: false,
            quality_layers: 1,
            output_residency: J2kAdaptiveOutputResidency::Host,
        }
    }

    /// Return the workload with ROI routing enabled or disabled.
    #[must_use]
    pub const fn with_roi(mut self, roi: bool) -> Self {
        self.roi = roi;
        self
    }

    /// Return the workload with scaled routing enabled or disabled.
    #[must_use]
    pub const fn with_scaled(mut self, scaled: bool) -> Self {
        self.scaled = scaled;
        self
    }

    /// Return the workload with a quality-layer count.
    #[must_use]
    pub const fn with_quality_layers(mut self, quality_layers: u16) -> Self {
        self.quality_layers = quality_layers;
        self
    }

    /// Return the workload with the requested output residency.
    #[must_use]
    pub const fn with_output_residency(
        mut self,
        output_residency: J2kAdaptiveOutputResidency,
    ) -> Self {
        self.output_residency = output_residency;
        self
    }

    /// Classify a stage for this workload before benchmark gating.
    #[must_use]
    pub fn logical_owner_for(self, stage: J2kAdaptiveStage) -> J2kAdaptiveStageOwner {
        if self.is_small_cpu_workload() {
            return J2kAdaptiveStageOwner::Cpu;
        }

        match stage {
            J2kAdaptiveStage::MarkerParsing
            | J2kAdaptiveStage::CopySync
            | J2kAdaptiveStage::Validation => J2kAdaptiveStageOwner::Cpu,
            J2kAdaptiveStage::Mct => {
                if self.components >= 3 && self.is_wsi_shaped() {
                    J2kAdaptiveStageOwner::Gpu
                } else {
                    J2kAdaptiveStageOwner::Variable
                }
            }
            J2kAdaptiveStage::Dwt => match self.operation {
                J2kAdaptiveOperation::Encode | J2kAdaptiveOperation::Transcode
                    if self.is_wsi_shaped() =>
                {
                    J2kAdaptiveStageOwner::Gpu
                }
                _ => J2kAdaptiveStageOwner::Cpu,
            },
            J2kAdaptiveStage::Idwt => match self.operation {
                J2kAdaptiveOperation::Decode | J2kAdaptiveOperation::Transcode
                    if self.is_wsi_shaped() =>
                {
                    J2kAdaptiveStageOwner::Gpu
                }
                _ => J2kAdaptiveStageOwner::Cpu,
            },
            J2kAdaptiveStage::Quantization => {
                if self.quality_mode == J2kAdaptiveQualityMode::Lossy && self.is_wsi_shaped() {
                    J2kAdaptiveStageOwner::Gpu
                } else {
                    J2kAdaptiveStageOwner::Cpu
                }
            }
            J2kAdaptiveStage::HtBlockCoding => {
                if self.codec_mode == J2kAdaptiveCodecMode::Htj2k && self.is_wsi_shaped() {
                    J2kAdaptiveStageOwner::Gpu
                } else {
                    J2kAdaptiveStageOwner::Cpu
                }
            }
            J2kAdaptiveStage::Tier1
            | J2kAdaptiveStage::PcrdRateControl
            | J2kAdaptiveStage::Packetization
            | J2kAdaptiveStage::CodestreamAssembly => J2kAdaptiveStageOwner::Variable,
        }
    }

    fn is_wsi_shaped(self) -> bool {
        let pixels = u64::from(self.tile_size.0).saturating_mul(u64::from(self.tile_size.1));
        pixels >= 512 * 512 || self.batch_size >= 16
    }

    fn is_small_cpu_workload(self) -> bool {
        let pixels = u64::from(self.tile_size.0).saturating_mul(u64::from(self.tile_size.1));
        pixels < 512 * 512 && self.batch_size <= 1
    }
}

/// Pipeline stage represented in adaptive route reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum J2kAdaptiveStage {
    /// Marker parsing and main/tile header processing.
    MarkerParsing,
    /// Multi-component transform, including fused deinterleave plus RCT/ICT.
    Mct,
    /// Forward wavelet transform.
    Dwt,
    /// Inverse wavelet transform.
    Idwt,
    /// Irreversible quantization or dequantization.
    Quantization,
    /// Classic EBCOT Tier-1 block coding.
    Tier1,
    /// HTJ2K cleanup/refinement block coding.
    HtBlockCoding,
    /// PCRD and rate-control decisions.
    PcrdRateControl,
    /// Packet ordering and packet body assembly.
    Packetization,
    /// Codestream marker and tile-part assembly.
    CodestreamAssembly,
    /// Host-device copies, synchronization, and residency transitions.
    CopySync,
    /// Decode/round-trip/output validation.
    Validation,
}

impl J2kAdaptiveStage {
    /// Every stage emitted by adaptive route reports.
    pub const ALL: [Self; 12] = [
        Self::MarkerParsing,
        Self::Mct,
        Self::Dwt,
        Self::Idwt,
        Self::Quantization,
        Self::Tier1,
        Self::HtBlockCoding,
        Self::PcrdRateControl,
        Self::Packetization,
        Self::CodestreamAssembly,
        Self::CopySync,
        Self::Validation,
    ];

    /// Stable profiling label for this stage.
    #[must_use]
    pub const fn profile_label(self) -> &'static str {
        match self {
            Self::MarkerParsing => "marker_parsing",
            Self::Mct => "mct_rct_ict",
            Self::Dwt => "dwt",
            Self::Idwt => "idwt",
            Self::Quantization => "quantization",
            Self::Tier1 => "tier1",
            Self::HtBlockCoding => "ht_block_coding",
            Self::PcrdRateControl => "pcrd_rate_control",
            Self::Packetization => "packetization",
            Self::CodestreamAssembly => "codestream_assembly",
            Self::CopySync => "copy_sync",
            Self::Validation => "validation",
        }
    }
}

/// Logical owner class before benchmark gates are applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum J2kAdaptiveStageOwner {
    /// CPU-shaped work.
    Cpu,
    /// GPU-shaped work that must still pass stage and end-to-end gates.
    Gpu,
    /// Workload-dependent stage requiring benchmark evidence before default device routing.
    Variable,
}

/// RCA reason for a logical GPU stage that did not pass its gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum J2kAdaptiveRcaReason {
    /// The device implementation uses the wrong algorithmic structure.
    AlgorithmicMismatch,
    /// Transfer or synchronization overhead dominates useful device work.
    TransferSyncOverhead,
    /// The implementation does not batch enough independent work.
    MissingBatching,
    /// The route loses residency and pays unnecessary host-device movement.
    MissingResidency,
    /// This exact workload is too small for device routing.
    TooSmallWorkload,
    /// The benchmark does not measure the intended workload shape.
    BenchmarkMismatch,
    /// Evidence shows the optimized CPU is genuinely better for this exact shape.
    CpuGenuinelyBetter,
}

/// RCA classification for a blocked stage/backend pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct J2kAdaptiveRcaFinding {
    /// Stage covered by the finding.
    pub stage: J2kAdaptiveStage,
    /// Device backend covered by the finding.
    pub backend: BackendKind,
    /// RCA reason.
    pub reason: J2kAdaptiveRcaReason,
    /// Whether the finding permits CPU routing for this exact stage/backend.
    pub reclassify_cpu: bool,
}

impl J2kAdaptiveRcaFinding {
    /// Record that RCA permits CPU routing for this exact stage/backend.
    #[must_use]
    pub const fn reclassify_cpu(
        stage: J2kAdaptiveStage,
        backend: BackendKind,
        reason: J2kAdaptiveRcaReason,
    ) -> Self {
        Self {
            stage,
            backend,
            reason,
            reclassify_cpu: true,
        }
    }
}

/// Benchmark evidence for one stage or one end-to-end adaptive route.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct J2kAdaptiveBenchmarkEvidence {
    /// Scope covered by the evidence.
    pub scope: J2kAdaptiveBenchmarkScope,
    /// Device backend measured against CPU.
    pub backend: BackendKind,
    /// Optimized CPU time in nanoseconds.
    pub cpu_ns: u64,
    /// Optimized accelerated or adaptive time in nanoseconds.
    pub accelerated_ns: u64,
    /// Criterion noise bound in percentage points.
    pub criterion_noise_percent: f64,
}

impl J2kAdaptiveBenchmarkEvidence {
    /// Build stage benchmark evidence.
    #[must_use]
    pub const fn stage(
        stage: J2kAdaptiveStage,
        backend: BackendKind,
        cpu_ns: u64,
        accelerated_ns: u64,
        criterion_noise_percent: f64,
    ) -> Self {
        Self {
            scope: J2kAdaptiveBenchmarkScope::Stage(stage),
            backend,
            cpu_ns,
            accelerated_ns,
            criterion_noise_percent,
        }
    }

    /// Build end-to-end route benchmark evidence.
    #[must_use]
    pub const fn end_to_end(
        backend: BackendKind,
        cpu_ns: u64,
        accelerated_ns: u64,
        criterion_noise_percent: f64,
    ) -> Self {
        Self {
            scope: J2kAdaptiveBenchmarkScope::EndToEnd,
            backend,
            cpu_ns,
            accelerated_ns,
            criterion_noise_percent,
        }
    }

    /// Percent speedup of accelerated over CPU.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn improvement_percent(self) -> f64 {
        if self.accelerated_ns == 0 {
            return f64::INFINITY;
        }
        ((self.cpu_ns as f64 / self.accelerated_ns as f64) - 1.0) * 100.0
    }

    fn passes(self, policy: J2kAdaptiveGatePolicy) -> bool {
        self.cpu_ns > 0
            && self.accelerated_ns > 0
            && self.improvement_percent()
                >= policy.min_speedup_percent + self.criterion_noise_percent
    }
}

/// Scope covered by benchmark evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum J2kAdaptiveBenchmarkScope {
    /// Evidence for one stage.
    Stage(J2kAdaptiveStage),
    /// Evidence for the full adaptive route.
    EndToEnd,
}

/// Benchmark evidence set used by the planner.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct J2kAdaptiveBenchmarks {
    stage: Vec<J2kAdaptiveBenchmarkEvidence>,
    end_to_end: Vec<J2kAdaptiveBenchmarkEvidence>,
}

impl J2kAdaptiveBenchmarks {
    /// Add stage evidence. Later evidence for the same stage/backend takes precedence.
    pub fn push_stage(&mut self, evidence: J2kAdaptiveBenchmarkEvidence) {
        debug_assert!(matches!(
            evidence.scope,
            J2kAdaptiveBenchmarkScope::Stage(_)
        ));
        self.stage.push(evidence);
    }

    /// Add end-to-end evidence. Later evidence for the same backend takes precedence.
    pub fn push_end_to_end(&mut self, evidence: J2kAdaptiveBenchmarkEvidence) {
        debug_assert!(matches!(
            evidence.scope,
            J2kAdaptiveBenchmarkScope::EndToEnd
        ));
        self.end_to_end.push(evidence);
    }

    fn stage_for(
        &self,
        stage: J2kAdaptiveStage,
        backend: BackendKind,
    ) -> Option<J2kAdaptiveBenchmarkEvidence> {
        self.stage.iter().rev().copied().find(|evidence| {
            evidence.backend == backend && evidence.scope == J2kAdaptiveBenchmarkScope::Stage(stage)
        })
    }

    fn end_to_end_for(&self, backend: BackendKind) -> Option<J2kAdaptiveBenchmarkEvidence> {
        self.end_to_end
            .iter()
            .rev()
            .copied()
            .find(|evidence| evidence.backend == backend)
    }

    fn has_evidence_for(&self, backend: BackendKind) -> bool {
        self.end_to_end_for(backend).is_some()
            || self
                .stage
                .iter()
                .any(|evidence| evidence.backend == backend)
    }

    fn best_observed_ns_for(&self, backend: BackendKind) -> Option<u64> {
        let end_to_end = self
            .end_to_end_for(backend)
            .map(|evidence| evidence.accelerated_ns);
        let stage = self
            .stage
            .iter()
            .rev()
            .find(|evidence| evidence.backend == backend)
            .map(|evidence| evidence.accelerated_ns);
        end_to_end.or(stage)
    }
}

/// Adaptive route gate policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct J2kAdaptiveGatePolicy {
    /// Minimum speedup required before Criterion noise is added.
    pub min_speedup_percent: f64,
}

impl Default for J2kAdaptiveGatePolicy {
    fn default() -> Self {
        Self {
            min_speedup_percent: 10.0,
        }
    }
}

/// Route selected by the planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum J2kAdaptiveRouteKind {
    /// Portable CPU route.
    CpuOnly,
    /// CPU plus benchmark-approved device stages.
    Hybrid,
    /// Strict device proof route.
    StrictDevice,
}

/// Gate status for one stage decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum J2kAdaptiveStageGateStatus {
    /// CPU-shaped stage, or CPU requested explicitly.
    CpuShaped,
    /// Variable stage kept on CPU because no passing stage benchmark exists.
    VariableCpuDefault,
    /// Stage passed its benchmark gate and may route to device.
    Approved,
    /// Required benchmark evidence is missing.
    BenchmarkGateMissing,
    /// A logical GPU stage failed its gate and needs RCA before default routing.
    BlockedNeedsRca,
    /// RCA permits CPU routing for this exact stage/backend/workload.
    ReclassifiedCpu,
    /// Strict device proof route bypasses adaptive performance gates.
    StrictDeviceProof,
    /// End-to-end route evidence is missing or below threshold.
    EndToEndGateBlocked,
}

/// One stage placement and gate result.
#[derive(Debug, Clone, PartialEq)]
pub struct J2kAdaptiveStageDecision {
    /// Pipeline stage.
    pub stage: J2kAdaptiveStage,
    /// Logical owner before gates.
    pub logical_owner: J2kAdaptiveStageOwner,
    /// Backend selected for this stage in the returned route.
    pub selected_backend: BackendKind,
    /// Gate status for this stage.
    pub gate_status: J2kAdaptiveStageGateStatus,
    /// Measured stage speedup when stage evidence was available.
    pub improvement_percent: Option<f64>,
    /// RCA finding applied to this stage, if any.
    pub rca_reason: Option<J2kAdaptiveRcaReason>,
}

impl J2kAdaptiveStageDecision {
    /// Return true when this stage blocks default GPU routing pending evidence or RCA.
    #[must_use]
    pub fn requires_rca(&self) -> bool {
        matches!(
            self.gate_status,
            J2kAdaptiveStageGateStatus::BenchmarkGateMissing
                | J2kAdaptiveStageGateStatus::BlockedNeedsRca
        )
    }
}

/// Adaptive planner output.
#[derive(Debug, Clone, PartialEq)]
pub struct J2kAdaptiveRouteReport {
    /// Requested route intent.
    pub request: J2kAdaptiveBackendRequest,
    /// Selected route kind.
    pub route_kind: J2kAdaptiveRouteKind,
    /// Device used by the route, when any stage is device-backed.
    pub selected_device: Option<BackendKind>,
    /// Stage decisions, always covering every [`J2kAdaptiveStage::ALL`] entry.
    pub stages: Vec<J2kAdaptiveStageDecision>,
}

impl J2kAdaptiveRouteReport {
    /// Return the decision for a stage.
    #[must_use]
    pub fn stage(&self, stage: J2kAdaptiveStage) -> Option<&J2kAdaptiveStageDecision> {
        self.stages.iter().find(|decision| decision.stage == stage)
    }

    /// Return true when any logical GPU stage remains unresolved.
    #[must_use]
    pub fn has_unresolved_rca(&self) -> bool {
        self.stages
            .iter()
            .any(J2kAdaptiveStageDecision::requires_rca)
    }
}

/// Adaptive JPEG 2000 route planner.
#[derive(Debug, Clone, PartialEq)]
pub struct J2kAdaptiveRoutePlanner {
    capabilities: BackendCapabilities,
    policy: J2kAdaptiveGatePolicy,
    rca_findings: Vec<J2kAdaptiveRcaFinding>,
}

impl J2kAdaptiveRoutePlanner {
    /// Build a planner from detected or test-provided capabilities.
    #[must_use]
    pub fn new(capabilities: BackendCapabilities) -> Self {
        Self {
            capabilities,
            policy: J2kAdaptiveGatePolicy::default(),
            rca_findings: Vec::new(),
        }
    }

    /// Build a planner with runtime-detected capabilities.
    #[must_use]
    pub fn detected() -> Self {
        Self::new(BackendCapabilities::detect())
    }

    /// Return a planner with a different gate policy.
    #[must_use]
    pub const fn with_policy(mut self, policy: J2kAdaptiveGatePolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Return a planner with an RCA finding.
    #[must_use]
    pub fn with_rca_finding(mut self, finding: J2kAdaptiveRcaFinding) -> Self {
        self.rca_findings.push(finding);
        self
    }

    /// Plan a route for the workload and benchmark evidence.
    pub fn plan(
        &self,
        workload: J2kAdaptiveWorkload,
        request: J2kAdaptiveBackendRequest,
        benchmarks: &J2kAdaptiveBenchmarks,
    ) -> Result<J2kAdaptiveRouteReport, J2kError> {
        match request {
            J2kAdaptiveBackendRequest::CpuOnly => Ok(Self::cpu_only_report(workload, request)),
            J2kAdaptiveBackendRequest::StrictDevice(backend) => {
                self.strict_device_report(workload, request, backend)
            }
            J2kAdaptiveBackendRequest::Accelerated => {
                Ok(self.accelerated_report(workload, request, benchmarks))
            }
        }
    }

    fn cpu_only_report(
        workload: J2kAdaptiveWorkload,
        request: J2kAdaptiveBackendRequest,
    ) -> J2kAdaptiveRouteReport {
        let stages = J2kAdaptiveStage::ALL
            .into_iter()
            .map(|stage| J2kAdaptiveStageDecision {
                stage,
                logical_owner: workload.logical_owner_for(stage),
                selected_backend: BackendKind::Cpu,
                gate_status: J2kAdaptiveStageGateStatus::CpuShaped,
                improvement_percent: None,
                rca_reason: None,
            })
            .collect();
        J2kAdaptiveRouteReport {
            request,
            route_kind: J2kAdaptiveRouteKind::CpuOnly,
            selected_device: None,
            stages,
        }
    }

    fn strict_device_report(
        &self,
        workload: J2kAdaptiveWorkload,
        request: J2kAdaptiveBackendRequest,
        backend: BackendKind,
    ) -> Result<J2kAdaptiveRouteReport, J2kError> {
        if !self.supports_backend(backend) {
            return Err(Unsupported {
                what: "strict JPEG 2000 device route is unavailable",
            }
            .into());
        }

        let stages = J2kAdaptiveStage::ALL
            .into_iter()
            .map(|stage| {
                let logical_owner = workload.logical_owner_for(stage);
                let selected_backend = if logical_owner == J2kAdaptiveStageOwner::Cpu {
                    BackendKind::Cpu
                } else {
                    backend
                };
                J2kAdaptiveStageDecision {
                    stage,
                    logical_owner,
                    selected_backend,
                    gate_status: if selected_backend == BackendKind::Cpu {
                        J2kAdaptiveStageGateStatus::CpuShaped
                    } else {
                        J2kAdaptiveStageGateStatus::StrictDeviceProof
                    },
                    improvement_percent: None,
                    rca_reason: None,
                }
            })
            .collect();

        Ok(J2kAdaptiveRouteReport {
            request,
            route_kind: J2kAdaptiveRouteKind::StrictDevice,
            selected_device: Some(backend),
            stages,
        })
    }

    fn accelerated_report(
        &self,
        workload: J2kAdaptiveWorkload,
        request: J2kAdaptiveBackendRequest,
        benchmarks: &J2kAdaptiveBenchmarks,
    ) -> J2kAdaptiveRouteReport {
        let backend = if let Some(backend) = self.best_approved_device(workload, benchmarks) {
            backend
        } else {
            return self.gated_cpu_report(
                workload,
                request,
                self.best_candidate_device(benchmarks),
                benchmarks,
            );
        };

        let mut stages = Vec::with_capacity(J2kAdaptiveStage::ALL.len());
        let mut unresolved = false;
        for stage in J2kAdaptiveStage::ALL {
            let decision = self.stage_decision(workload, stage, backend, benchmarks, true);
            unresolved |= decision.requires_rca();
            stages.push(decision);
        }

        if unresolved {
            for decision in &mut stages {
                decision.selected_backend = BackendKind::Cpu;
            }
            return J2kAdaptiveRouteReport {
                request,
                route_kind: J2kAdaptiveRouteKind::CpuOnly,
                selected_device: None,
                stages,
            };
        }

        let has_device_stage = stages
            .iter()
            .any(|decision| decision.selected_backend == backend);
        J2kAdaptiveRouteReport {
            request,
            route_kind: if has_device_stage {
                J2kAdaptiveRouteKind::Hybrid
            } else {
                J2kAdaptiveRouteKind::CpuOnly
            },
            selected_device: has_device_stage.then_some(backend),
            stages,
        }
    }

    fn gated_cpu_report(
        &self,
        workload: J2kAdaptiveWorkload,
        request: J2kAdaptiveBackendRequest,
        backend: Option<BackendKind>,
        benchmarks: &J2kAdaptiveBenchmarks,
    ) -> J2kAdaptiveRouteReport {
        let stages = J2kAdaptiveStage::ALL
            .into_iter()
            .map(|stage| {
                let mut decision = if let Some(backend) = backend {
                    let end_to_end_passed = benchmarks
                        .end_to_end_for(backend)
                        .is_some_and(|evidence| evidence.passes(self.policy));
                    self.stage_decision(workload, stage, backend, benchmarks, end_to_end_passed)
                } else {
                    let logical_owner = workload.logical_owner_for(stage);
                    J2kAdaptiveStageDecision {
                        stage,
                        logical_owner,
                        selected_backend: BackendKind::Cpu,
                        gate_status: if logical_owner == J2kAdaptiveStageOwner::Gpu {
                            J2kAdaptiveStageGateStatus::BenchmarkGateMissing
                        } else {
                            J2kAdaptiveStageGateStatus::CpuShaped
                        },
                        improvement_percent: None,
                        rca_reason: None,
                    }
                };
                decision.selected_backend = BackendKind::Cpu;
                decision
            })
            .collect();

        J2kAdaptiveRouteReport {
            request,
            route_kind: J2kAdaptiveRouteKind::CpuOnly,
            selected_device: None,
            stages,
        }
    }

    fn stage_decision(
        &self,
        workload: J2kAdaptiveWorkload,
        stage: J2kAdaptiveStage,
        backend: BackendKind,
        benchmarks: &J2kAdaptiveBenchmarks,
        end_to_end_passed: bool,
    ) -> J2kAdaptiveStageDecision {
        let logical_owner = workload.logical_owner_for(stage);
        match logical_owner {
            J2kAdaptiveStageOwner::Cpu => J2kAdaptiveStageDecision {
                stage,
                logical_owner,
                selected_backend: BackendKind::Cpu,
                gate_status: J2kAdaptiveStageGateStatus::CpuShaped,
                improvement_percent: None,
                rca_reason: None,
            },
            J2kAdaptiveStageOwner::Variable => {
                let evidence = benchmarks.stage_for(stage, backend);
                let approved = end_to_end_passed
                    && evidence.is_some_and(|evidence| evidence.passes(self.policy));
                J2kAdaptiveStageDecision {
                    stage,
                    logical_owner,
                    selected_backend: if approved { backend } else { BackendKind::Cpu },
                    gate_status: if approved {
                        J2kAdaptiveStageGateStatus::Approved
                    } else {
                        J2kAdaptiveStageGateStatus::VariableCpuDefault
                    },
                    improvement_percent: evidence
                        .map(J2kAdaptiveBenchmarkEvidence::improvement_percent),
                    rca_reason: None,
                }
            }
            J2kAdaptiveStageOwner::Gpu => {
                if let Some(finding) = self.rca_for(stage, backend) {
                    return J2kAdaptiveStageDecision {
                        stage,
                        logical_owner,
                        selected_backend: BackendKind::Cpu,
                        gate_status: J2kAdaptiveStageGateStatus::ReclassifiedCpu,
                        improvement_percent: benchmarks
                            .stage_for(stage, backend)
                            .map(J2kAdaptiveBenchmarkEvidence::improvement_percent),
                        rca_reason: Some(finding.reason),
                    };
                }

                let evidence = benchmarks.stage_for(stage, backend);
                let gate_status = match (end_to_end_passed, evidence) {
                    (false, _) => J2kAdaptiveStageGateStatus::EndToEndGateBlocked,
                    (true, None) => J2kAdaptiveStageGateStatus::BenchmarkGateMissing,
                    (true, Some(evidence)) if evidence.passes(self.policy) => {
                        J2kAdaptiveStageGateStatus::Approved
                    }
                    (true, Some(_)) => J2kAdaptiveStageGateStatus::BlockedNeedsRca,
                };
                J2kAdaptiveStageDecision {
                    stage,
                    logical_owner,
                    selected_backend: if gate_status == J2kAdaptiveStageGateStatus::Approved {
                        backend
                    } else {
                        BackendKind::Cpu
                    },
                    gate_status,
                    improvement_percent: evidence
                        .map(J2kAdaptiveBenchmarkEvidence::improvement_percent),
                    rca_reason: None,
                }
            }
        }
    }

    fn best_approved_device(
        &self,
        workload: J2kAdaptiveWorkload,
        benchmarks: &J2kAdaptiveBenchmarks,
    ) -> Option<BackendKind> {
        [BackendKind::Metal, BackendKind::Cuda]
            .into_iter()
            .filter(|backend| self.supports_backend(*backend))
            .filter_map(|backend| {
                benchmarks
                    .end_to_end_for(backend)
                    .filter(|evidence| evidence.passes(self.policy))
                    .map(|evidence| (backend, evidence.accelerated_ns))
            })
            .filter(|(backend, _)| {
                J2kAdaptiveStage::ALL.into_iter().all(|stage| {
                    !self
                        .stage_decision(workload, stage, *backend, benchmarks, true)
                        .requires_rca()
                })
            })
            .min_by_key(|(_, accelerated_ns)| *accelerated_ns)
            .map(|(backend, _)| backend)
    }

    fn best_candidate_device(&self, benchmarks: &J2kAdaptiveBenchmarks) -> Option<BackendKind> {
        [BackendKind::Metal, BackendKind::Cuda]
            .into_iter()
            .filter(|backend| self.supports_backend(*backend))
            .filter(|backend| benchmarks.has_evidence_for(*backend))
            .min_by_key(|backend| {
                benchmarks
                    .best_observed_ns_for(*backend)
                    .unwrap_or(u64::MAX)
            })
    }

    fn supports_backend(&self, backend: BackendKind) -> bool {
        match backend {
            BackendKind::Cpu => true,
            BackendKind::Metal => self.capabilities.metal,
            BackendKind::Cuda => self.capabilities.cuda,
        }
    }

    fn rca_for(
        &self,
        stage: J2kAdaptiveStage,
        backend: BackendKind,
    ) -> Option<J2kAdaptiveRcaFinding> {
        self.rca_findings.iter().copied().find(|finding| {
            finding.stage == stage && finding.backend == backend && finding.reclassify_cpu
        })
    }
}
