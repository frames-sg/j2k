// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{ErrorMetrics, JpegToHtj2kCoefficientPath};

/// Aggregate report for multi-tile transcode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchTranscodeReport {
    /// Number of input tiles.
    pub tile_count: usize,
    /// Number of successfully encoded output tiles.
    pub successful_tiles: usize,
    /// Number of tile-local failures.
    pub failed_tiles: usize,
    /// Number of transformed components across successful extracted tiles.
    pub transformed_components: usize,
    /// Number of same-geometry reversible 5/3 batches submitted.
    pub reversible_dwt53_batches: usize,
    /// Number of reversible 5/3 component jobs in submitted batches.
    pub reversible_dwt53_batch_jobs: usize,
    /// Batch extraction time in microseconds.
    pub extract_us: u128,
    /// Batch DCT-to-wavelet time in microseconds.
    pub transform_us: u128,
    /// Batch HTJ2K encode time in microseconds.
    pub encode_us: u128,
    /// Detailed stage timings for the batch. Batch-accelerated 5/3 transform
    /// timings stay here instead of being copied into every tile report.
    pub timings: TranscodeTimingReport,
    /// Coefficient path used by the batch.
    pub coefficient_path: JpegToHtj2kCoefficientPath,
}

/// Stable profile request label for transcode batch telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscodeBatchProfileRequest {
    /// CPU-only transcode request.
    Cpu,
    /// Auto-routing request that may use an accelerator.
    MetalAuto,
    /// Explicit Metal request.
    MetalExplicit,
}

impl TranscodeBatchProfileRequest {
    /// Stable `request` label emitted in `j2k_profile` rows.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::MetalAuto => "metal_auto",
            Self::MetalExplicit => "metal_explicit",
        }
    }

    /// Stable `transform_processor` label for this request and timing report.
    #[must_use]
    pub fn transform_processor(self, timings: &TranscodeTimingReport) -> &'static str {
        if matches!(self, Self::MetalAuto | Self::MetalExplicit)
            && timings.accelerator_work_observed()
        {
            "metal"
        } else {
            "cpu"
        }
    }

    /// Stable `path` label for this request and timing report.
    #[must_use]
    pub fn profile_path(self, timings: &TranscodeTimingReport) -> &'static str {
        if self.transform_processor(timings) != "metal" {
            return "cpu";
        }
        match self {
            Self::Cpu => "cpu",
            Self::MetalAuto => "auto",
            Self::MetalExplicit => "metal",
        }
    }
}

/// Shared transcode batch profile row fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscodeBatchProfileRow {
    fields: Vec<(&'static str, String)>,
}

pub(super) type TranscodeBatchProfileFields = Vec<(&'static str, String)>;

impl TranscodeBatchProfileRow {
    /// Build profile fields for a batch transcode report.
    #[must_use]
    pub fn new(
        report: &BatchTranscodeReport,
        context: impl AsRef<str>,
        request: TranscodeBatchProfileRequest,
    ) -> Self {
        let timings = report.timings;
        let context = context.as_ref().replace(' ', "_");
        let coefficient_path = format!("{:?}", report.coefficient_path);
        let total_us = report
            .extract_us
            .saturating_add(report.transform_us)
            .saturating_add(report.encode_us);
        let transform_processor = request.transform_processor(&timings);
        let path = request.profile_path(&timings);

        let mut fields = Vec::with_capacity(68);
        Self::push_route_fields(
            &mut fields,
            request,
            path,
            context,
            coefficient_path,
            transform_processor,
        );
        Self::push_batch_fields(&mut fields, report, total_us);
        Self::push_input_timing_fields(&mut fields, &timings);
        Self::push_dwt97_timing_fields(&mut fields, &timings);
        Self::push_transfer_fields(&mut fields, &timings);
        Self::push_encode_timing_fields(&mut fields, &timings);
        Self::push_accelerator_fields(&mut fields, &timings);
        Self { fields }
    }

    fn push_route_fields(
        fields: &mut TranscodeBatchProfileFields,
        request: TranscodeBatchProfileRequest,
        path: &str,
        context: String,
        coefficient_path: String,
        transform_processor: &str,
    ) {
        fields.extend([
            ("codec", "transcode".to_string()),
            ("op", "transcode_batch".to_string()),
            ("request", request.as_str().to_string()),
            ("path", path.to_string()),
            ("pipeline", "jpeg_to_htj2k".to_string()),
            ("context", context),
            ("coefficient_path", coefficient_path),
            ("extract_processor", "cpu".to_string()),
            ("transform_processor", transform_processor.to_string()),
            ("encode_processor", "cpu".to_string()),
        ]);
    }

    fn push_batch_fields(
        fields: &mut TranscodeBatchProfileFields,
        report: &BatchTranscodeReport,
        total_us: u128,
    ) {
        fields.extend([
            ("tile_count", report.tile_count.to_string()),
            ("successful_tiles", report.successful_tiles.to_string()),
            ("failed_tiles", report.failed_tiles.to_string()),
            (
                "transformed_components",
                report.transformed_components.to_string(),
            ),
            (
                "reversible_dwt53_batches",
                report.reversible_dwt53_batches.to_string(),
            ),
            (
                "reversible_dwt53_batch_jobs",
                report.reversible_dwt53_batch_jobs.to_string(),
            ),
            ("extract_us", report.extract_us.to_string()),
            ("transform_us", report.transform_us.to_string()),
            ("encode_us", report.encode_us.to_string()),
            ("total_us", total_us.to_string()),
        ]);
    }

    fn push_input_timing_fields(
        fields: &mut TranscodeBatchProfileFields,
        timings: &TranscodeTimingReport,
    ) {
        fields.extend([
            (
                "source_raw_probe_us",
                timings.source_raw_probe_us.to_string(),
            ),
            (
                "read_region_decode_us",
                timings.read_region_decode_us.to_string(),
            ),
            ("compose_pad_us", timings.compose_pad_us.to_string()),
            (
                "generated_jpeg_encode_us",
                timings.generated_jpeg_encode_us.to_string(),
            ),
            (
                "jpeg_dct_extract_us",
                timings.jpeg_dct_extract_us.to_string(),
            ),
            ("jpeg_dct_repack_us", timings.jpeg_dct_repack_us.to_string()),
            (
                "dct_to_wavelet_total_us",
                timings.dct_to_wavelet_total_us.to_string(),
            ),
            (
                "dct_to_wavelet_accelerator_us",
                timings.dct_to_wavelet_accelerator_us.to_string(),
            ),
            (
                "dct_to_wavelet_cpu_fallback_us",
                timings.dct_to_wavelet_cpu_fallback_us.to_string(),
            ),
            ("dwt_decompose_us", timings.dwt_decompose_us.to_string()),
        ]);
    }

    fn push_dwt97_timing_fields(
        fields: &mut TranscodeBatchProfileFields,
        timings: &TranscodeTimingReport,
    ) {
        fields.extend([
            (
                "dwt97_batch_pack_upload_us",
                timings.dwt97_batch_pack_upload_us.to_string(),
            ),
            (
                "dwt97_batch_pack_upload_transfers",
                timings.dwt97_batch_pack_upload_transfers.to_string(),
            ),
            (
                "dwt97_batch_pack_upload_bytes",
                timings.dwt97_batch_pack_upload_bytes.to_string(),
            ),
            (
                "dwt97_batch_resident_dct_handoff_count",
                timings.dwt97_batch_resident_dct_handoff_count.to_string(),
            ),
            (
                "dwt97_batch_idct_row_lift_us",
                timings.dwt97_batch_idct_row_lift_us.to_string(),
            ),
            (
                "dwt97_batch_column_lift_us",
                timings.dwt97_batch_column_lift_us.to_string(),
            ),
            (
                "dwt97_batch_resident_dwt_handoff_count",
                timings.dwt97_batch_resident_dwt_handoff_count.to_string(),
            ),
            (
                "dwt97_batch_quantize_codeblock_us",
                timings.dwt97_batch_quantize_codeblock_us.to_string(),
            ),
            (
                "dwt97_batch_ht_encode_us",
                timings.dwt97_batch_ht_encode_us.to_string(),
            ),
            (
                "dwt97_batch_ht_codeblock_dispatches",
                timings.dwt97_batch_ht_codeblock_dispatches.to_string(),
            ),
        ]);
    }

    fn push_transfer_fields(
        fields: &mut TranscodeBatchProfileFields,
        timings: &TranscodeTimingReport,
    ) {
        let device_to_host_transfer_count = timings
            .dwt97_batch_readback_transfers
            .saturating_add(timings.dwt97_batch_ht_status_readback_transfers)
            .saturating_add(timings.dwt97_batch_ht_output_readback_transfers);
        let device_to_host_transfer_bytes = timings
            .dwt97_batch_readback_bytes
            .saturating_add(timings.dwt97_batch_ht_status_readback_bytes)
            .saturating_add(timings.dwt97_batch_ht_output_readback_bytes);

        fields.extend([
            (
                "dwt97_batch_ht_status_readback_us",
                timings.dwt97_batch_ht_status_readback_us.to_string(),
            ),
            (
                "dwt97_batch_ht_status_readback_transfers",
                timings.dwt97_batch_ht_status_readback_transfers.to_string(),
            ),
            (
                "dwt97_batch_ht_status_readback_bytes",
                timings.dwt97_batch_ht_status_readback_bytes.to_string(),
            ),
            (
                "dwt97_batch_ht_output_readback_us",
                timings.dwt97_batch_ht_output_readback_us.to_string(),
            ),
            (
                "dwt97_batch_ht_output_readback_transfers",
                timings.dwt97_batch_ht_output_readback_transfers.to_string(),
            ),
            (
                "dwt97_batch_ht_output_readback_bytes",
                timings.dwt97_batch_ht_output_readback_bytes.to_string(),
            ),
            (
                "dwt97_batch_readback_us",
                timings.dwt97_batch_readback_us.to_string(),
            ),
            (
                "dwt97_batch_readback_transfers",
                timings.dwt97_batch_readback_transfers.to_string(),
            ),
            (
                "dwt97_batch_readback_bytes",
                timings.dwt97_batch_readback_bytes.to_string(),
            ),
            (
                "host_to_device_transfer_count",
                timings.dwt97_batch_pack_upload_transfers.to_string(),
            ),
            (
                "host_to_device_transfer_bytes",
                timings.dwt97_batch_pack_upload_bytes.to_string(),
            ),
            (
                "device_to_host_transfer_count",
                device_to_host_transfer_count.to_string(),
            ),
            (
                "device_to_host_transfer_bytes",
                device_to_host_transfer_bytes.to_string(),
            ),
        ]);
    }

    fn push_encode_timing_fields(
        fields: &mut TranscodeBatchProfileFields,
        timings: &TranscodeTimingReport,
    ) {
        fields.extend([
            ("htj2k_encode_us", timings.htj2k_encode_us.to_string()),
            (
                "htj2k_encode_accelerator_dispatches",
                timings.htj2k_encode_accelerator_dispatches.to_string(),
            ),
            (
                "htj2k_encode_ht_code_block_dispatches",
                timings.htj2k_encode_ht_code_block_dispatches.to_string(),
            ),
            (
                "htj2k_encode_packetization_dispatches",
                timings.htj2k_encode_packetization_dispatches.to_string(),
            ),
            ("component_count", timings.component_count.to_string()),
            ("batch_count", timings.batch_count.to_string()),
            ("batch_jobs", timings.batch_jobs.to_string()),
        ]);
    }

    fn push_accelerator_fields(
        fields: &mut TranscodeBatchProfileFields,
        timings: &TranscodeTimingReport,
    ) {
        fields.extend([
            (
                "accelerator_attempts",
                timings.accelerator_attempts.to_string(),
            ),
            ("accelerator_jobs", timings.accelerator_jobs.to_string()),
            (
                "accelerator_dispatches",
                timings.accelerator_dispatches.to_string(),
            ),
            (
                "accelerator_dispatched_jobs",
                timings.accelerator_dispatched_jobs.to_string(),
            ),
            ("cpu_fallback_jobs", timings.cpu_fallback_jobs.to_string()),
        ]);
    }

    /// Ordered profile row fields.
    #[must_use]
    pub fn fields(&self) -> &[(&'static str, String)] {
        &self.fields
    }

    /// Stable profile codec label.
    #[must_use]
    pub fn codec(&self) -> &str {
        self.required_field("codec")
    }

    /// Stable profile operation label.
    #[must_use]
    pub fn op(&self) -> &str {
        self.required_field("op")
    }

    /// Stable profile path label.
    #[must_use]
    pub fn path(&self) -> &str {
        self.required_field("path")
    }

    fn required_field(&self, key: &str) -> &str {
        self.fields
            .iter()
            .find_map(|(field_key, value)| (*field_key == key).then_some(value.as_str()))
            .expect("transcode batch profile row includes required prefix field")
    }
}

impl BatchTranscodeReport {
    /// Build shared profile fields for a batch transcode report.
    #[must_use]
    pub fn profile_row(
        &self,
        context: impl AsRef<str>,
        request: TranscodeBatchProfileRequest,
    ) -> TranscodeBatchProfileRow {
        TranscodeBatchProfileRow::new(self, context, request)
    }
}

/// Detailed timing and dispatch counters for JPEG-to-HTJ2K transcode.
///
/// Durations are wall-clock microseconds measured around the current Rust API
/// boundaries. Accelerator time includes backend submission and wait overhead
/// visible to this crate; backend-specific hardware counters are not exposed
/// here.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TranscodeTimingReport {
    /// Raw compressed-tile probe/read time before JPEG DCT extraction.
    pub source_raw_probe_us: u128,
    /// Source region decode time for strip/retile workflows.
    pub read_region_decode_us: u128,
    /// Region compose/pad time for generated regular tiles.
    pub compose_pad_us: u128,
    /// JPEG encode time when the workflow generates regular JPEG tiles.
    pub generated_jpeg_encode_us: u128,
    /// JPEG DCT extraction time in microseconds.
    pub jpeg_dct_extract_us: u128,
    /// Time spent repacking integer DCT coefficients into float block grids.
    pub jpeg_dct_repack_us: u128,
    /// Total wall time spent producing DWT bands from JPEG DCT coefficients.
    pub dct_to_wavelet_total_us: u128,
    /// Wall time spent inside accelerator hook calls.
    pub dct_to_wavelet_accelerator_us: u128,
    /// Wall time spent in scalar CPU fallback transforms.
    pub dct_to_wavelet_cpu_fallback_us: u128,
    /// Time spent decomposing first-level DWT output into requested levels.
    pub dwt_decompose_us: u128,
    /// Backend 9/7 batch host pack/upload time in microseconds.
    pub dwt97_batch_pack_upload_us: u128,
    /// Logical host-to-device transfers during backend 9/7 batch pack/upload.
    pub dwt97_batch_pack_upload_transfers: usize,
    /// Host-to-device bytes during backend 9/7 batch pack/upload.
    pub dwt97_batch_pack_upload_bytes: u64,
    /// Resident JPEG DCT-grid descriptors validated during backend 9/7 batches.
    pub dwt97_batch_resident_dct_handoff_count: usize,
    /// Backend 9/7 batch IDCT plus horizontal row-lift time in microseconds.
    pub dwt97_batch_idct_row_lift_us: u128,
    /// Backend 9/7 batch vertical column-lift time in microseconds.
    pub dwt97_batch_column_lift_us: u128,
    /// Resident DWT subband descriptors validated during backend 9/7 batches.
    pub dwt97_batch_resident_dwt_handoff_count: usize,
    /// Backend 9/7 batch quantize/code-block layout time in microseconds.
    pub dwt97_batch_quantize_codeblock_us: u128,
    /// Backend 9/7 resident HT code-block encode time in microseconds.
    pub dwt97_batch_ht_encode_us: u128,
    /// Backend 9/7 resident HT cleanup-pass encode kernel time in microseconds.
    pub dwt97_batch_ht_kernel_us: u128,
    /// Backend 9/7 resident HT status-buffer device-to-host readback time in microseconds.
    pub dwt97_batch_ht_status_readback_us: u128,
    /// Logical device-to-host status readbacks after resident HT encode.
    pub dwt97_batch_ht_status_readback_transfers: usize,
    /// Device-to-host status bytes after resident HT encode.
    pub dwt97_batch_ht_status_readback_bytes: u64,
    /// Backend 9/7 resident HT encoded-byte compaction kernel time in microseconds.
    pub dwt97_batch_ht_compact_us: u128,
    /// Backend 9/7 resident HT compacted encoded-byte device-to-host readback time in microseconds.
    pub dwt97_batch_ht_output_readback_us: u128,
    /// Logical device-to-host output readbacks after resident HT compaction.
    pub dwt97_batch_ht_output_readback_transfers: usize,
    /// Device-to-host output bytes after resident HT compaction.
    pub dwt97_batch_ht_output_readback_bytes: u64,
    /// Backend 9/7 resident HT code-block encode dispatches.
    pub dwt97_batch_ht_codeblock_dispatches: usize,
    /// Backend 9/7 batch output readback/unpack time in microseconds.
    pub dwt97_batch_readback_us: u128,
    /// Logical device-to-host transfers during backend 9/7 batch output readback.
    pub dwt97_batch_readback_transfers: usize,
    /// Device-to-host bytes during backend 9/7 batch output readback.
    pub dwt97_batch_readback_bytes: u64,
    /// HTJ2K encode time in microseconds.
    pub htj2k_encode_us: u128,
    /// Encode-stage accelerator dispatches during HTJ2K encode.
    pub htj2k_encode_accelerator_dispatches: usize,
    /// HT cleanup code-block accelerator dispatches during HTJ2K encode.
    pub htj2k_encode_ht_code_block_dispatches: usize,
    /// Packetization accelerator dispatches during HTJ2K encode.
    pub htj2k_encode_packetization_dispatches: usize,
    /// Time spent writing compressed frames to a DICOM `PixelData` spool.
    pub dicom_spool_write_us: u128,
    /// Time spent writing final DICOM instances.
    pub dicom_final_write_us: u128,
    /// Number of source tiles represented by this timing report.
    pub tile_count: usize,
    /// Number of components transformed into wavelet bands.
    pub component_count: usize,
    /// Number of same-geometry transform batches offered to the accelerator.
    pub batch_count: usize,
    /// Number of component jobs in same-geometry transform batches.
    pub batch_jobs: usize,
    /// Number of accelerator hook calls.
    pub accelerator_attempts: usize,
    /// Number of component jobs offered through accelerator hook calls.
    pub accelerator_jobs: usize,
    /// Number of accelerator hook calls that returned an accelerated result.
    pub accelerator_dispatches: usize,
    /// Number of component jobs completed by accelerated results.
    pub accelerator_dispatched_jobs: usize,
    /// Number of component jobs completed by scalar CPU fallback transforms.
    pub cpu_fallback_jobs: usize,
}

impl TranscodeTimingReport {
    /// Returns true when the report contains evidence that accelerator-backed
    /// work executed for the transcode transform path.
    pub fn accelerator_work_observed(&self) -> bool {
        self.accelerator_dispatches > 0
            || self.dwt97_batch_pack_upload_transfers > 0
            || self.dwt97_batch_pack_upload_bytes > 0
            || self.dwt97_batch_resident_dct_handoff_count > 0
            || self.dwt97_batch_idct_row_lift_us > 0
            || self.dwt97_batch_column_lift_us > 0
            || self.dwt97_batch_resident_dwt_handoff_count > 0
            || self.dwt97_batch_quantize_codeblock_us > 0
            || self.dwt97_batch_ht_encode_us > 0
            || self.dwt97_batch_ht_kernel_us > 0
            || self.dwt97_batch_ht_compact_us > 0
            || self.dwt97_batch_ht_codeblock_dispatches > 0
            || self.dwt97_batch_readback_transfers > 0
            || self.dwt97_batch_readback_bytes > 0
            || self.dwt97_batch_ht_status_readback_transfers > 0
            || self.dwt97_batch_ht_status_readback_bytes > 0
            || self.dwt97_batch_ht_output_readback_transfers > 0
            || self.dwt97_batch_ht_output_readback_bytes > 0
    }

    pub(super) fn add_assign(&mut self, other: Self) {
        macro_rules! saturating_add_fields {
            ($($field:ident),+ $(,)?) => {
                $(
                    self.$field = self.$field.saturating_add(other.$field);
                )+
            };
        }

        saturating_add_fields!(
            source_raw_probe_us,
            read_region_decode_us,
            compose_pad_us,
            generated_jpeg_encode_us,
            jpeg_dct_extract_us,
            jpeg_dct_repack_us,
            dct_to_wavelet_total_us,
            dct_to_wavelet_accelerator_us,
            dct_to_wavelet_cpu_fallback_us,
            dwt_decompose_us,
            dwt97_batch_pack_upload_us,
            dwt97_batch_pack_upload_transfers,
            dwt97_batch_pack_upload_bytes,
            dwt97_batch_resident_dct_handoff_count,
            dwt97_batch_idct_row_lift_us,
            dwt97_batch_column_lift_us,
            dwt97_batch_resident_dwt_handoff_count,
            dwt97_batch_quantize_codeblock_us,
            dwt97_batch_ht_encode_us,
            dwt97_batch_ht_kernel_us,
            dwt97_batch_ht_status_readback_us,
            dwt97_batch_ht_status_readback_transfers,
            dwt97_batch_ht_status_readback_bytes,
            dwt97_batch_ht_compact_us,
            dwt97_batch_ht_output_readback_us,
            dwt97_batch_ht_output_readback_transfers,
            dwt97_batch_ht_output_readback_bytes,
            dwt97_batch_ht_codeblock_dispatches,
            dwt97_batch_readback_us,
            dwt97_batch_readback_transfers,
            dwt97_batch_readback_bytes,
            htj2k_encode_us,
            htj2k_encode_accelerator_dispatches,
            htj2k_encode_ht_code_block_dispatches,
            htj2k_encode_packetization_dispatches,
            dicom_spool_write_us,
            dicom_final_write_us,
            tile_count,
            component_count,
            batch_count,
            batch_jobs,
            accelerator_attempts,
            accelerator_jobs,
            accelerator_dispatches,
            accelerator_dispatched_jobs,
            cpu_fallback_jobs,
        );
    }
}

/// Per-component transcode geometry preserved in the generated codestream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscodeComponentReport {
    /// Component index in JPEG SOF order.
    pub component_index: usize,
    /// Native component width in samples before HTJ2K SIZ expansion.
    pub width: u32,
    /// Native component height in samples before HTJ2K SIZ expansion.
    pub height: u32,
    /// Number of DCT blocks per component row, including padded edge blocks.
    pub block_cols: u32,
    /// Number of DCT block rows, including padded edge blocks.
    pub block_rows: u32,
    /// HTJ2K SIZ horizontal sampling factor.
    pub x_rsiz: u8,
    /// HTJ2K SIZ vertical sampling factor.
    pub y_rsiz: u8,
}

/// Error metrics from an optional validation oracle.
pub type TranscodeValidationMetrics = ErrorMetrics;

/// Classification for optional coefficient-validation metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscodeValidationClassification {
    /// All compared coefficients match the selected oracle exactly.
    Exact,
    /// Coefficients satisfy the experimental one-LSB-bounded threshold:
    /// maximum absolute error is at most one LSB and at least 99.9% of
    /// coefficients match exactly.
    OneLsbBounded,
    /// Coefficients do not satisfy the exact or one-LSB-bounded thresholds.
    OutsideThreshold,
}

impl TranscodeValidationClassification {
    /// Classify validation metrics using the experimental acceptance
    /// thresholds documented for this coefficient-domain path.
    #[must_use]
    pub fn classify_metrics(metrics: &TranscodeValidationMetrics) -> Self {
        if metrics.exact_matches == metrics.total && metrics.max_abs_error == 0 {
            Self::Exact
        } else if metrics.is_one_lsb_bounded(0.999) {
            Self::OneLsbBounded
        } else {
            Self::OutsideThreshold
        }
    }
}

/// Transcode summary for validation and benchmarking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscodeReport {
    /// Source reference-grid width.
    pub width: u32,
    /// Source reference-grid height.
    pub height: u32,
    /// Number of transformed components.
    pub component_count: usize,
    /// Native transformed component geometry and SIZ sampling.
    pub components: Vec<TranscodeComponentReport>,
    /// Rounded coefficient metrics against the optional float IDCT-then-DWT
    /// oracle.
    pub float_reference_metrics: Option<TranscodeValidationMetrics>,
    /// Threshold classification for `float_reference_metrics`.
    pub float_reference_classification: Option<TranscodeValidationClassification>,
    /// Rounded direct coefficients compared with j2k-jpeg scalar
    /// ISLOW-IDCT-then-reversible-5/3 coefficients.
    pub integer_reference_metrics: Option<TranscodeValidationMetrics>,
    /// Threshold classification for `integer_reference_metrics`.
    pub integer_reference_classification: Option<TranscodeValidationClassification>,
    /// Number of DWT decomposition levels encoded.
    pub decomposition_levels: u8,
    /// Coefficient path used to generate the HTJ2K bands.
    pub coefficient_path: JpegToHtj2kCoefficientPath,
    /// Name of the experimental path used.
    pub path: &'static str,
    /// Wall-clock extraction time in microseconds.
    pub extract_us: u128,
    /// Wall-clock DCT-to-wavelet time in microseconds.
    pub transform_us: u128,
    /// Wall-clock HTJ2K encode time in microseconds.
    pub encode_us: u128,
    /// Detailed stage timings and accelerator/fallback counters.
    pub timings: TranscodeTimingReport,
}
