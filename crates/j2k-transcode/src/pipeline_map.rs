// SPDX-License-Identifier: MIT OR Apache-2.0

//! Stage-level residency report for JPEG-to-HTJ2K transcode timings.

use core::fmt;

use crate::{BatchTranscodeReport, TranscodeReport, TranscodeTimingReport};

/// Logical stages in the JPEG-to-J2K/HTJ2K transcode pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscodePipelineStageKind {
    /// JPEG marker parsing, entropy decode, and DCT coefficient extraction.
    EntropyDecode,
    /// Coefficient repacking and host/device input preparation.
    CoefficientPrep,
    /// DCT-grid to wavelet-domain transform.
    Transform,
    /// Quantization, code-block layout, and pre-packet code-block work.
    QuantizationCodeBlockPrep,
    /// Packet header and packet body formation.
    Packetization,
    /// Final marker, tile-part, and codestream byte assembly.
    CodestreamAssembly,
}

impl TranscodePipelineStageKind {
    /// Stable snake-case label used by debug reports and logs.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EntropyDecode => "entropy_decode",
            Self::CoefficientPrep => "coefficient_prep",
            Self::Transform => "transform",
            Self::QuantizationCodeBlockPrep => "quantization_code_block_prep",
            Self::Packetization => "packetization",
            Self::CodestreamAssembly => "codestream_assembly",
        }
    }
}

impl fmt::Display for TranscodePipelineStageKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Observed residency for a transcode stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscodeStageProcessor {
    /// Work is currently observed at CPU/native Rust or native encoder boundaries.
    Cpu,
    /// Work is observed through the Metal/accelerator stage counters.
    Metal,
    /// Existing counters show both CPU and Metal/accelerator work for this stage.
    Hybrid,
}

impl TranscodeStageProcessor {
    /// Stable label used by debug reports and logs.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "Cpu",
            Self::Metal => "Metal",
            Self::Hybrid => "Hybrid",
        }
    }
}

impl fmt::Display for TranscodeStageProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One stage in a transcode pipeline residency map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TranscodePipelineStageReport {
    /// Logical transcode stage.
    pub stage: TranscodePipelineStageKind,
    /// Observed CPU/Metal residency for this stage.
    pub processor: TranscodeStageProcessor,
    /// CPU/native time currently visible for this stage, in microseconds.
    pub cpu_us: u128,
    /// Metal/accelerator time currently visible for this stage, in microseconds.
    pub metal_us: u128,
    /// Host/device transfer time visible for this stage, in microseconds.
    pub transfer_us: u128,
    /// Logical host/device transfer operations visible for this stage.
    pub transfer_count: usize,
    /// Host/device transfer bytes visible for this stage.
    pub transfer_bytes: u64,
    /// Validated resident handoff descriptors visible for this stage.
    pub resident_handoff_count: usize,
    /// Dispatches observed for this stage.
    pub dispatches: usize,
    /// Component jobs that used CPU fallback at this stage.
    pub fallback_jobs: usize,
    /// Short interpretation of the counters behind this stage.
    pub note: &'static str,
}

/// Recommended next stage to evaluate for Metal residency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TranscodeResidentStageRecommendation {
    /// Stage that should be evaluated next.
    pub stage: TranscodePipelineStageKind,
    /// Existing measured time supporting the recommendation, in microseconds.
    pub evidence_us: u128,
    /// Existing dispatch count supporting the recommendation.
    pub evidence_dispatches: usize,
    /// Why this stage is the next candidate.
    pub reason: &'static str,
}

/// Stage-by-stage transcode residency map derived from existing timings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscodePipelineMap {
    /// Ordered stage reports from JPEG input through codestream output.
    pub stages: Vec<TranscodePipelineStageReport>,
    /// Next resident-stage candidate derived from the observed counters.
    pub recommendation: TranscodeResidentStageRecommendation,
}

impl TranscodePipelineMap {
    /// Build a pipeline map from an existing timing report.
    #[must_use]
    pub fn from_timings(timings: &TranscodeTimingReport) -> Self {
        Self {
            stages: vec![
                entropy_decode_stage(timings),
                coefficient_prep_stage(timings),
                transform_stage(timings),
                quantization_code_block_stage(timings),
                packetization_stage(timings),
                codestream_assembly_stage(timings),
            ],
            recommendation: recommend_next_resident_stage(timings),
        }
    }
}

impl TranscodeTimingReport {
    /// Convert this timing report into a CPU/Metal transcode pipeline map.
    #[must_use]
    pub fn pipeline_map(&self) -> TranscodePipelineMap {
        TranscodePipelineMap::from_timings(self)
    }
}

impl TranscodeReport {
    /// Convert this transcode report into a CPU/Metal pipeline map.
    #[must_use]
    pub fn pipeline_map(&self) -> TranscodePipelineMap {
        self.timings.pipeline_map()
    }
}

impl BatchTranscodeReport {
    /// Convert this batch transcode report into a CPU/Metal pipeline map.
    #[must_use]
    pub fn pipeline_map(&self) -> TranscodePipelineMap {
        self.timings.pipeline_map()
    }
}

fn entropy_decode_stage(timings: &TranscodeTimingReport) -> TranscodePipelineStageReport {
    TranscodePipelineStageReport {
        stage: TranscodePipelineStageKind::EntropyDecode,
        processor: TranscodeStageProcessor::Cpu,
        cpu_us: timings.jpeg_dct_extract_us,
        metal_us: 0,
        transfer_us: 0,
        transfer_count: 0,
        transfer_bytes: 0,
        resident_handoff_count: 0,
        dispatches: 0,
        fallback_jobs: 0,
        note: "JPEG marker parsing, entropy decode, dequantization, and DCT coefficient extraction stay on CPU",
    }
}

fn coefficient_prep_stage(timings: &TranscodeTimingReport) -> TranscodePipelineStageReport {
    let transfer_us = timings.dwt97_batch_pack_upload_us;
    let transfer_count = transfer_count_or_timing(
        timings.dwt97_batch_pack_upload_transfers,
        transfer_us,
        timings.dwt97_batch_pack_upload_bytes,
    );
    let transfer_dispatch = usize::from(
        transfer_us > 0 || transfer_count > 0 || timings.dwt97_batch_pack_upload_bytes > 0,
    );
    TranscodePipelineStageReport {
        stage: TranscodePipelineStageKind::CoefficientPrep,
        processor: processor_for(
            timings.jpeg_dct_repack_us,
            0,
            transfer_us,
            transfer_dispatch,
            0,
        ),
        cpu_us: timings.jpeg_dct_repack_us,
        metal_us: 0,
        transfer_us,
        transfer_count,
        transfer_bytes: timings.dwt97_batch_pack_upload_bytes,
        resident_handoff_count: timings.dwt97_batch_resident_dct_handoff_count,
        dispatches: transfer_dispatch,
        fallback_jobs: 0,
        note: "DCT coefficient repack and Metal buffer pack/upload are visible before transform dispatch",
    }
}

fn transform_stage(timings: &TranscodeTimingReport) -> TranscodePipelineStageReport {
    let dwt97_kernel_us = timings
        .dwt97_batch_idct_row_lift_us
        .saturating_add(timings.dwt97_batch_column_lift_us);
    let metal_us = if dwt97_kernel_us > 0 {
        dwt97_kernel_us
    } else if timings.accelerator_dispatches > 0 {
        timings.dct_to_wavelet_accelerator_us
    } else {
        0
    };
    let cpu_us = timings
        .dct_to_wavelet_cpu_fallback_us
        .saturating_add(timings.dwt_decompose_us);
    TranscodePipelineStageReport {
        stage: TranscodePipelineStageKind::Transform,
        processor: processor_for(
            cpu_us,
            metal_us,
            timings.dwt97_batch_readback_us,
            timings.accelerator_dispatches,
            timings.cpu_fallback_jobs,
        ),
        cpu_us,
        metal_us,
        transfer_us: timings.dwt97_batch_readback_us,
        transfer_count: transfer_count_or_timing(
            timings.dwt97_batch_readback_transfers,
            timings.dwt97_batch_readback_us,
            timings.dwt97_batch_readback_bytes,
        ),
        transfer_bytes: timings.dwt97_batch_readback_bytes,
        resident_handoff_count: timings.dwt97_batch_resident_dwt_handoff_count,
        dispatches: timings.accelerator_dispatches,
        fallback_jobs: timings.cpu_fallback_jobs,
        note: "DCT-grid to DWT projection uses accelerator dispatches when available; Ok(None) jobs remain caller CPU fallback",
    }
}

fn quantization_code_block_stage(timings: &TranscodeTimingReport) -> TranscodePipelineStageReport {
    let metal_us = timings
        .dwt97_batch_quantize_codeblock_us
        .saturating_add(timings.dwt97_batch_ht_encode_us)
        .saturating_add(timings.dwt97_batch_ht_kernel_us)
        .saturating_add(timings.dwt97_batch_ht_compact_us);
    let transfer_us = timings
        .dwt97_batch_ht_status_readback_us
        .saturating_add(timings.dwt97_batch_ht_output_readback_us);
    let transfer_count = timings
        .dwt97_batch_ht_status_readback_transfers
        .saturating_add(timings.dwt97_batch_ht_output_readback_transfers);
    let transfer_bytes = timings
        .dwt97_batch_ht_status_readback_bytes
        .saturating_add(timings.dwt97_batch_ht_output_readback_bytes);
    let transfer_count = transfer_count_or_timing(transfer_count, transfer_us, transfer_bytes);
    let dispatches = timings
        .dwt97_batch_ht_codeblock_dispatches
        .saturating_add(timings.htj2k_encode_ht_code_block_dispatches);
    TranscodePipelineStageReport {
        stage: TranscodePipelineStageKind::QuantizationCodeBlockPrep,
        processor: processor_for(0, metal_us, transfer_us, dispatches, 0),
        cpu_us: 0,
        metal_us,
        transfer_us,
        transfer_count,
        transfer_bytes,
        resident_handoff_count: 0,
        dispatches,
        fallback_jobs: 0,
        note: "9/7 quantization/code-block layout is only isolated when a backend reports resident stage timings; otherwise it is inside native encode time",
    }
}

fn packetization_stage(timings: &TranscodeTimingReport) -> TranscodePipelineStageReport {
    let dispatches = timings.htj2k_encode_packetization_dispatches;
    TranscodePipelineStageReport {
        stage: TranscodePipelineStageKind::Packetization,
        processor: processor_for(0, 0, 0, dispatches, 0),
        cpu_us: 0,
        metal_us: 0,
        transfer_us: 0,
        transfer_count: 0,
        transfer_bytes: 0,
        resident_handoff_count: 0,
        dispatches,
        fallback_jobs: 0,
        note: "Packetization dispatches are counted separately when an encode-stage accelerator handles them; CPU time is otherwise inside native encode time",
    }
}

fn codestream_assembly_stage(timings: &TranscodeTimingReport) -> TranscodePipelineStageReport {
    TranscodePipelineStageReport {
        stage: TranscodePipelineStageKind::CodestreamAssembly,
        processor: TranscodeStageProcessor::Cpu,
        cpu_us: timings.htj2k_encode_us,
        metal_us: 0,
        transfer_us: 0,
        transfer_count: 0,
        transfer_bytes: 0,
        resident_handoff_count: 0,
        dispatches: 0,
        fallback_jobs: 0,
        note: "Final marker, tile-part, packet byte ordering, and codestream assembly remain at the CPU/native encoder boundary",
    }
}

fn recommend_next_resident_stage(
    timings: &TranscodeTimingReport,
) -> TranscodeResidentStageRecommendation {
    let transform_readback_us = timings.dwt97_batch_readback_us;
    let code_block_readback_us = timings
        .dwt97_batch_ht_status_readback_us
        .saturating_add(timings.dwt97_batch_ht_output_readback_us);
    let has_resident_transform = timings.accelerator_dispatches > 0
        && (timings.dwt97_batch_idct_row_lift_us > 0
            || timings.dwt97_batch_column_lift_us > 0
            || timings.dct_to_wavelet_accelerator_us > 0);

    if timings.cpu_fallback_jobs > 0 && timings.accelerator_dispatches == 0 {
        return TranscodeResidentStageRecommendation {
            stage: TranscodePipelineStageKind::Transform,
            evidence_us: timings.dct_to_wavelet_cpu_fallback_us,
            evidence_dispatches: timings.accelerator_attempts,
            reason: "transform jobs are still completing through caller CPU fallback",
        };
    }

    if has_resident_transform && timings.dwt97_batch_quantize_codeblock_us == 0 {
        return TranscodeResidentStageRecommendation {
            stage: TranscodePipelineStageKind::QuantizationCodeBlockPrep,
            evidence_us: transform_readback_us.saturating_add(timings.htj2k_encode_us),
            evidence_dispatches: timings.accelerator_dispatches,
            reason: "resident transform output is read back before quantization/code-block prep and native encode",
        };
    }

    if timings.dwt97_batch_quantize_codeblock_us > 0
        && timings.dwt97_batch_ht_codeblock_dispatches == 0
    {
        return TranscodeResidentStageRecommendation {
            stage: TranscodePipelineStageKind::QuantizationCodeBlockPrep,
            evidence_us: transform_readback_us
                .saturating_add(code_block_readback_us)
                .saturating_add(timings.htj2k_encode_us),
            evidence_dispatches: timings.accelerator_dispatches,
            reason: "Metal already reaches 9/7 code-block prep; extend residency through HT code-block encode before packetization",
        };
    }

    if timings.cpu_fallback_jobs > 0 {
        return TranscodeResidentStageRecommendation {
            stage: TranscodePipelineStageKind::Transform,
            evidence_us: timings.dct_to_wavelet_cpu_fallback_us,
            evidence_dispatches: timings.accelerator_attempts,
            reason: "some transform jobs still use CPU fallback after accelerator attempts",
        };
    }

    let coefficient_prep_us = timings
        .jpeg_dct_repack_us
        .saturating_add(timings.dwt97_batch_pack_upload_us);
    if coefficient_prep_us > 0 {
        return TranscodeResidentStageRecommendation {
            stage: TranscodePipelineStageKind::CoefficientPrep,
            evidence_us: coefficient_prep_us,
            evidence_dispatches: timings.accelerator_dispatches,
            reason:
                "coefficient repack and host-to-device upload are visible before Metal work starts",
        };
    }

    if timings.htj2k_encode_packetization_dispatches == 0 && timings.htj2k_encode_us > 0 {
        return TranscodeResidentStageRecommendation {
            stage: TranscodePipelineStageKind::Packetization,
            evidence_us: timings.htj2k_encode_us,
            evidence_dispatches: timings.htj2k_encode_accelerator_dispatches,
            reason:
                "packetization and codestream assembly remain inside the CPU/native encode boundary",
        };
    }

    TranscodeResidentStageRecommendation {
        stage: TranscodePipelineStageKind::CodestreamAssembly,
        evidence_us: timings.htj2k_encode_us,
        evidence_dispatches: timings.htj2k_encode_accelerator_dispatches,
        reason: "no stronger resident-stage candidate is visible from the current counters",
    }
}

fn processor_for(
    cpu_us: u128,
    metal_us: u128,
    transfer_us: u128,
    dispatches: usize,
    fallback_jobs: usize,
) -> TranscodeStageProcessor {
    let has_cpu = cpu_us > 0 || fallback_jobs > 0;
    let has_metal = metal_us > 0 || transfer_us > 0 || dispatches > 0;
    match (has_cpu, has_metal) {
        (true, true) => TranscodeStageProcessor::Hybrid,
        (false, true) => TranscodeStageProcessor::Metal,
        (_, false) => TranscodeStageProcessor::Cpu,
    }
}

fn transfer_count_or_timing(
    explicit_count: usize,
    transfer_us: u128,
    transfer_bytes: u64,
) -> usize {
    if explicit_count > 0 {
        explicit_count
    } else {
        usize::from(transfer_us > 0 || transfer_bytes > 0)
    }
}
