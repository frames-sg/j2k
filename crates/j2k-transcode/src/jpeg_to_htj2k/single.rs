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

struct PreparedSingleTranscode {
    jpeg: JpegDctImage,
    component_sampling: Vec<(u8, u8)>,
    decomposition_levels: u8,
    all_unit_sampled: bool,
    component_reports: Vec<TranscodeComponentReport>,
    extract_us: u128,
    retained_pipeline: HostLiveBudget,
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
) -> Result<PreparedSingleTranscode, JpegToHtj2kError> {
    validate_transcode_options(options)?;
    let workspace = validate_jpeg_transcode_workspace(bytes, options)?;
    let mut admission_budget = HostLiveBudget::process_cap();
    admission_budget.add_bytes(external_live_bytes)?;
    admission_budget.add_bytes(scratch.retained_bytes()?)?;
    admission_budget.add_bytes(workspace.peak_bytes())?;

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

    let mut retained_pipeline = HostLiveBudget::process_cap();
    retained_pipeline.add_bytes(external_live_bytes)?;
    retained_pipeline.add_bytes(jpeg.retained_bytes()?)?;
    retained_pipeline.add_capacity::<(u8, u8)>(component_sampling.capacity())?;
    retained_pipeline.add_capacity::<TranscodeComponentReport>(component_reports.capacity())?;
    Ok(PreparedSingleTranscode {
        jpeg,
        component_sampling,
        decomposition_levels,
        all_unit_sampled,
        component_reports,
        extract_us,
        retained_pipeline,
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
    let PreparedSingleTranscode {
        jpeg,
        component_sampling,
        decomposition_levels,
        all_unit_sampled,
        component_reports,
        extract_us,
        retained_pipeline,
    } = prepare_single_transcode(bytes, options, scratch, external_live_bytes)?;
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
