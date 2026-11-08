// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Instant;

use super::{
    component_sampling_for_jpeg, decomposition_levels_for_components, encode_component_batch,
    encoded_transcode_retained_bytes, extract_dct_blocks, precomputed_batch_retained_bytes,
    transcode_component_batch, transcode_path_name, try_vec_with_capacity,
    validate_jpeg_transcode_workspace, validate_transcode_options,
    validation_metrics_retained_bytes, ComponentBatchRequest, ComponentTranscodeBatch,
    DctExtractOptions, DctToWaveletStageAccelerator, EncodedTranscode, HostLiveBudget,
    J2kEncodeStageAccelerator, JpegDctImage, JpegToHtj2kError, JpegToHtj2kOptions,
    JpegToHtj2kScratch, TranscodeComponentReport, TranscodeReport, TranscodeTimingReport,
    TranscodeValidationClassification, TranscodeValidationMetrics,
};

pub(super) struct PreparedJpegTranscode {
    pub(super) jpeg: JpegDctImage,
    pub(super) component_sampling: Vec<(u8, u8)>,
    pub(super) decomposition_levels: u8,
    pub(super) all_unit_sampled: bool,
    pub(super) component_reports: Vec<TranscodeComponentReport>,
    pub(super) extract_us: u128,
}

struct CompletedSingleTranscode {
    codestream: Vec<u8>,
    float_reference_metrics: Option<TranscodeValidationMetrics>,
    integer_reference_metrics: Option<TranscodeValidationMetrics>,
    component_reports: Vec<TranscodeComponentReport>,
    width: u32,
    height: u32,
    component_count: usize,
    decomposition_levels: u8,
    all_unit_sampled: bool,
    extract_us: u128,
    transform_us: u128,
    encode_us: u128,
    timings: TranscodeTimingReport,
}

fn finish_single_transcode(
    completed: CompletedSingleTranscode,
    options: &JpegToHtj2kOptions,
    scratch: &JpegToHtj2kScratch,
    external_live_bytes: usize,
) -> Result<EncodedTranscode, JpegToHtj2kError> {
    let float_reference_classification = completed
        .float_reference_metrics
        .as_ref()
        .map(TranscodeValidationClassification::classify_metrics);
    let integer_reference_classification = completed
        .integer_reference_metrics
        .as_ref()
        .map(TranscodeValidationClassification::classify_metrics);
    let encoded = EncodedTranscode {
        codestream: completed.codestream,
        report: TranscodeReport {
            width: completed.width,
            height: completed.height,
            component_count: completed.component_count,
            components: completed.component_reports,
            float_reference_classification,
            float_reference_metrics: completed.float_reference_metrics,
            integer_reference_classification,
            integer_reference_metrics: completed.integer_reference_metrics,
            decomposition_levels: completed.decomposition_levels,
            coefficient_path: options.coefficient_path,
            path: transcode_path_name(completed.all_unit_sampled, options.coefficient_path),
            extract_us: completed.extract_us,
            transform_us: completed.transform_us,
            encode_us: completed.encode_us,
            timings: completed.timings,
        },
    };
    let mut output_budget = HostLiveBudget::process_cap();
    output_budget.add_bytes(external_live_bytes)?;
    output_budget.add_bytes(scratch.retained_bytes()?)?;
    output_budget.add_bytes(encoded_transcode_retained_bytes(&encoded)?)?;
    Ok(encoded)
}

fn prepare_single_transcode(
    bytes: &[u8],
    options: &JpegToHtj2kOptions,
    scratch: &JpegToHtj2kScratch,
    external_live_bytes: usize,
) -> Result<(PreparedJpegTranscode, HostLiveBudget), JpegToHtj2kError> {
    validate_transcode_options(options)?;
    let prepared = prepare_jpeg_transcode(bytes, options, |workspace_bytes| {
        let mut admission_budget = HostLiveBudget::process_cap();
        admission_budget.add_bytes(external_live_bytes)?;
        admission_budget.add_bytes(scratch.retained_bytes()?)?;
        admission_budget.add_bytes(workspace_bytes)?;
        Ok(())
    })?;
    let mut retained_pipeline = HostLiveBudget::process_cap();
    retained_pipeline.add_bytes(external_live_bytes)?;
    retained_pipeline.add_bytes(prepared.jpeg.retained_bytes()?)?;
    retained_pipeline.add_capacity::<(u8, u8)>(prepared.component_sampling.capacity())?;
    retained_pipeline
        .add_capacity::<TranscodeComponentReport>(prepared.component_reports.capacity())?;
    Ok((prepared, retained_pipeline))
}

/// Shared JPEG facts before single-image or grouped transform preparation.
/// Caller admission runs after workspace validation, before DCT allocation.
pub(super) fn prepare_jpeg_transcode(
    bytes: &[u8],
    options: &JpegToHtj2kOptions,
    admit_workspace: impl FnOnce(usize) -> Result<(), JpegToHtj2kError>,
) -> Result<PreparedJpegTranscode, JpegToHtj2kError> {
    let workspace = validate_jpeg_transcode_workspace(bytes, options)?;
    admit_workspace(workspace.peak_bytes())?;

    let extract_start = Instant::now();
    let jpeg = extract_dct_blocks(bytes, DctExtractOptions::dequantized_only())?;
    let extract_us = extract_start.elapsed().as_micros();
    if jpeg.components.is_empty() || jpeg.components.len() > 4 {
        return Err(JpegToHtj2kError::Unsupported(
            "unsupported JPEG component count for jpeg_to_htj2k",
        ));
    }

    let component_sampling =
        component_sampling_for_jpeg(&jpeg.components, jpeg.width, jpeg.height)?;
    let decomposition_levels = decomposition_levels_for_components(
        &jpeg.components,
        options.encode_options.num_decomposition_levels,
    )?;
    let all_unit_sampled = component_sampling
        .iter()
        .all(|&(x_rsiz, y_rsiz)| x_rsiz == 1 && y_rsiz == 1);
    let mut component_reports = try_vec_with_capacity(jpeg.components.len())?;
    for (component, (x_rsiz, y_rsiz)) in jpeg
        .components
        .iter()
        .zip(component_sampling.iter().copied())
    {
        component_reports.push(TranscodeComponentReport {
            component_index: component.component_index,
            width: component.width,
            height: component.height,
            block_cols: component.block_cols,
            block_rows: component.block_rows,
            x_rsiz,
            y_rsiz,
        });
    }

    Ok(PreparedJpegTranscode {
        jpeg,
        component_sampling,
        decomposition_levels,
        all_unit_sampled,
        component_reports,
        extract_us,
    })
}

pub(super) fn jpeg_to_htj2k_with_scratch<
    A: DctToWaveletStageAccelerator,
    E: J2kEncodeStageAccelerator,
>(
    bytes: &[u8],
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut A,
    encode_accelerator: &mut E,
    external_live_bytes: usize,
) -> Result<EncodedTranscode, JpegToHtj2kError> {
    let (
        PreparedJpegTranscode {
            jpeg,
            component_sampling,
            decomposition_levels,
            all_unit_sampled,
            component_reports,
            extract_us,
        },
        retained_pipeline,
    ) = prepare_single_transcode(bytes, options, scratch, external_live_bytes)?;
    let mut timings = TranscodeTimingReport {
        tile_count: 1,
        ..TranscodeTimingReport::default()
    };
    timings.jpeg_dct_extract_us = extract_us;

    let transform_start = Instant::now();
    let component_batch = transcode_component_batch(
        ComponentBatchRequest {
            components: &jpeg.components,
            component_sampling: &component_sampling,
            decomposition_levels,
            options,
            retained_pipeline_bytes: retained_pipeline.live_bytes(),
        },
        scratch,
        accelerator,
        &mut timings,
    )?;
    let transform_us = transform_start.elapsed().as_micros();
    timings.dct_to_wavelet_total_us = transform_us;

    let mut transformed_budget = retained_pipeline;
    transformed_budget.add_bytes(scratch.retained_bytes()?)?;
    transformed_budget.add_bytes(precomputed_batch_retained_bytes(
        &component_batch.precomputed_components,
    )?)?;
    transformed_budget.add_bytes(validation_metrics_retained_bytes(
        component_batch.float_reference_metrics.as_ref(),
    )?)?;
    transformed_budget.add_bytes(validation_metrics_retained_bytes(
        component_batch.integer_reference_metrics.as_ref(),
    )?)?;

    let image_width = jpeg.width;
    let image_height = jpeg.height;
    let component_count = jpeg.components.len();
    let mut encode_external = HostLiveBudget::process_cap();
    encode_external.add_bytes(external_live_bytes)?;
    encode_external.add_bytes(scratch.retained_bytes()?)?;
    encode_external.add_capacity::<TranscodeComponentReport>(component_reports.capacity())?;
    encode_external.add_bytes(validation_metrics_retained_bytes(
        component_batch.float_reference_metrics.as_ref(),
    )?)?;
    encode_external.add_bytes(validation_metrics_retained_bytes(
        component_batch.integer_reference_metrics.as_ref(),
    )?)?;
    let native_host_cap = encode_external.remaining_bytes()?;
    let ComponentTranscodeBatch {
        precomputed_components,
        float_reference_metrics,
        integer_reference_metrics,
    } = component_batch;
    drop(component_sampling);
    drop(jpeg);

    let (codestream, encode_us) = encode_component_batch(
        image_width,
        image_height,
        precomputed_components,
        options,
        encode_accelerator,
        &mut timings,
        native_host_cap,
    )?;

    finish_single_transcode(
        CompletedSingleTranscode {
            codestream,
            float_reference_metrics,
            integer_reference_metrics,
            component_reports,
            width: image_width,
            height: image_height,
            component_count,
            decomposition_levels,
            all_unit_sampled,
            extract_us,
            transform_us,
            encode_us,
            timings,
        },
        options,
        scratch,
        external_live_bytes,
    )
}

#[cfg(test)]
mod preparation_tests {
    use super::super::batch::{
        batch_component_groups, float97_batch_component_groups, prepare_float97_batch_tile,
        prepare_integer_batch_tile,
    };
    use super::*;
    use j2k_test_support::{
        progressive_8x8_jpeg, JPEG_BASELINE_420_16X16, JPEG_BASELINE_444_8X8, JPEG_GRAYSCALE_8X8,
    };

    #[test]
    fn single_integer_and_float_preparation_preserve_the_same_jpeg_facts() {
        let progressive = progressive_8x8_jpeg();
        for bytes in [
            JPEG_GRAYSCALE_8X8,
            JPEG_BASELINE_444_8X8,
            JPEG_BASELINE_420_16X16,
            progressive.as_slice(),
        ] {
            let mut options = JpegToHtj2kOptions::lossless_53();
            options.encode_options.num_decomposition_levels = 3;
            let (single, _) =
                prepare_single_transcode(bytes, &options, &JpegToHtj2kScratch::default(), 0)
                    .expect("single preparation");
            let integer =
                prepare_integer_batch_tile(0, bytes, &options).expect("integer batch preparation");
            let mut float_options = JpegToHtj2kOptions::lossy_97();
            float_options.encode_options.num_decomposition_levels = 3;
            let float = prepare_float97_batch_tile(0, bytes, &float_options)
                .expect("float batch preparation");
            for (jpeg, sampling, levels, unit, reports) in [
                (
                    &integer.jpeg,
                    &integer.component_sampling,
                    integer.decomposition_levels,
                    integer.all_unit_sampled,
                    &integer.component_reports,
                ),
                (
                    &float.jpeg,
                    &float.component_sampling,
                    float.decomposition_levels,
                    float.all_unit_sampled,
                    &float.component_reports,
                ),
            ] {
                assert_eq!(jpeg, &single.jpeg);
                assert_eq!(sampling, &single.component_sampling);
                assert_eq!(levels, single.decomposition_levels);
                assert_eq!(unit, single.all_unit_sampled);
                assert_eq!(reports, &single.component_reports);
                assert!(jpeg.components.iter().all(|component| {
                    component.quantized_blocks.is_empty()
                        && !component.dequantized_blocks.is_empty()
                }));
            }
            assert_eq!(integer.timings.tile_count, 1);
            assert_eq!(float.timings.tile_count, 1);
            if bytes == JPEG_BASELINE_444_8X8 {
                assert_eq!(
                    batch_component_groups(core::slice::from_ref(&integer))
                        .unwrap()
                        .len(),
                    3,
                    "integer groups keep component identity"
                );
                assert_eq!(
                    float97_batch_component_groups(core::slice::from_ref(&float))
                        .unwrap()
                        .len(),
                    1,
                    "float groups can share compatible color geometry"
                );
            }
        }
    }

    #[test]
    fn preparation_keeps_workspace_errors_and_single_external_admission() {
        let options = JpegToHtj2kOptions::lossless_53();
        let scratch = JpegToHtj2kScratch::default();
        let cap = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;
        let malformed = [0_u8];
        let single = prepare_single_transcode(&malformed, &options, &scratch, cap)
            .err()
            .expect("invalid JPEG precedes external admission");
        let integer = prepare_integer_batch_tile(0, &malformed, &options)
            .err()
            .expect("invalid integer input");
        let float = prepare_float97_batch_tile(0, &malformed, &options)
            .err()
            .expect("invalid float input");
        assert_eq!(single.to_string(), integer.to_string());
        assert_eq!(single.to_string(), float.to_string());
        assert!(matches!(
            prepare_single_transcode(JPEG_GRAYSCALE_8X8, &options, &scratch, cap),
            Err(JpegToHtj2kError::MemoryCapExceeded { requested, cap: limit })
                if requested > cap && limit == cap
        ));
    }
}
