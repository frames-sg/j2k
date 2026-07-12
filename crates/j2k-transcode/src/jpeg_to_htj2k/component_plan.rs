// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    error_metrics_i32_with_live_budget, flatten_integer_wavelet, float97_reference_coefficients,
    float_direct_97_wavelet_from_component, float_direct_wavelet_from_component,
    float_reference_coefficients, integer_direct_wavelet_from_component,
    integer_reference_coefficients, integer_wavelet_from_first_level, j2k_dwt97_from_wavelet,
    j2k_dwt_from_integer_wavelet, j2k_dwt_from_wavelet, precomputed_batch_retained_bytes,
    record_accelerator_dispatch, record_batch_attempt, rounded_wavelet97_i32, rounded_wavelet_i32,
    same_geometry_component_groups, validate_component_block_grid, DctGridToReversibleDwt53Job,
    DctToWaveletStageAccelerator, HostLiveBudget, Instant, IntegerWavelet, JpegDctComponent,
    JpegToHtj2kCoefficientPath, JpegToHtj2kError, JpegToHtj2kOptions, JpegToHtj2kScratch,
    PrecomputedHtj2k53Component, PrecomputedHtj2k97Component, TranscodeTimingReport,
    TranscodeValidationMetrics,
};
use crate::allocation::{try_extend_from_slice, try_vec_from_slice, try_vec_with_capacity};

pub(super) struct ComponentTranscodeBatch {
    pub(super) precomputed_components: PrecomputedComponentBatch,
    pub(super) float_reference_metrics: Option<TranscodeValidationMetrics>,
    pub(super) integer_reference_metrics: Option<TranscodeValidationMetrics>,
}

pub(super) enum PrecomputedComponentBatch {
    Dwt53(Vec<PrecomputedHtj2k53Component>),
    Dwt97(Vec<PrecomputedHtj2k97Component>),
}

pub(super) struct ComponentTranscodeResult {
    pub(super) precomputed: PrecomputedComponent,
    pub(super) float_validation_coefficients: Option<(Vec<i32>, Vec<i32>)>,
    pub(super) integer_validation_coefficients: Option<(Vec<i32>, Vec<i32>)>,
}

pub(super) enum PrecomputedComponent {
    Dwt53(PrecomputedHtj2k53Component),
    Dwt97(PrecomputedHtj2k97Component),
}

#[derive(Clone, Copy)]
pub(super) struct ComponentBatchRequest<'a> {
    pub(super) components: &'a [JpegDctComponent],
    pub(super) component_sampling: &'a [(u8, u8)],
    pub(super) decomposition_levels: u8,
    pub(super) options: &'a JpegToHtj2kOptions,
    pub(super) retained_pipeline_bytes: usize,
}

pub(super) fn transcode_component_batch(
    request: ComponentBatchRequest<'_>,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut impl DctToWaveletStageAccelerator,
    timings: &mut TranscodeTimingReport,
) -> Result<ComponentTranscodeBatch, JpegToHtj2kError> {
    if matches!(
        request.options.coefficient_path,
        JpegToHtj2kCoefficientPath::FloatDirectLinear97
    ) && request.options.validate_against_integer_reference
    {
        return Err(JpegToHtj2kError::Unsupported(
            "integer reversible validation is only defined for 5/3 coefficient paths",
        ));
    }

    if matches!(
        request.options.coefficient_path,
        JpegToHtj2kCoefficientPath::IntegerDirect53
    ) {
        return transcode_integer_component_batch(request, scratch, accelerator, timings);
    }

    let ComponentBatchRequest {
        components,
        component_sampling,
        decomposition_levels,
        options,
        retained_pipeline_bytes,
    } = request;
    let mut precomputed_53 = try_vec_with_capacity(components.len())?;
    let mut precomputed_97 = try_vec_with_capacity(components.len())?;
    let mut float_validation_actual = Vec::new();
    let mut float_validation_expected = Vec::new();
    let mut integer_validation_actual = Vec::new();
    let mut integer_validation_expected = Vec::new();

    for (component, (x_rsiz, y_rsiz)) in components.iter().zip(component_sampling.iter().copied()) {
        let component_result = component_to_precomputed_htj2k(
            ComponentTranscodePlan {
                component,
                x_rsiz,
                y_rsiz,
                decomposition_levels,
                options,
            },
            scratch,
            accelerator,
            timings,
        )?;
        match component_result.precomputed {
            PrecomputedComponent::Dwt53(precomputed) => precomputed_53.push(precomputed),
            PrecomputedComponent::Dwt97(precomputed) => precomputed_97.push(precomputed),
        }
        if let Some((actual, expected)) = component_result.float_validation_coefficients {
            try_extend_from_slice(&mut float_validation_actual, &actual)?;
            try_extend_from_slice(&mut float_validation_expected, &expected)?;
        }
        if let Some((actual, expected)) = component_result.integer_validation_coefficients {
            try_extend_from_slice(&mut integer_validation_actual, &actual)?;
            try_extend_from_slice(&mut integer_validation_expected, &expected)?;
        }
    }

    let precomputed_components = if matches!(
        options.coefficient_path,
        JpegToHtj2kCoefficientPath::FloatDirectLinear97
    ) {
        PrecomputedComponentBatch::Dwt97(precomputed_97)
    } else {
        PrecomputedComponentBatch::Dwt53(precomputed_53)
    };
    finish_component_batch(
        precomputed_components,
        ValidationCoefficientOwners {
            float_actual: float_validation_actual,
            float_expected: float_validation_expected,
            integer_actual: integer_validation_actual,
            integer_expected: integer_validation_expected,
        },
        options,
        scratch,
        retained_pipeline_bytes,
    )
}

pub(super) fn transcode_integer_component_batch(
    request: ComponentBatchRequest<'_>,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut impl DctToWaveletStageAccelerator,
    timings: &mut TranscodeTimingReport,
) -> Result<ComponentTranscodeBatch, JpegToHtj2kError> {
    let ComponentBatchRequest {
        components,
        component_sampling,
        decomposition_levels,
        options,
        retained_pipeline_bytes,
    } = request;
    let mut precomputed_53 = try_vec_with_capacity(components.len())?;
    precomputed_53.resize_with(components.len(), || None);
    let mut float_validation_actual = Vec::new();
    let mut float_validation_expected = Vec::new();
    let mut integer_validation_actual = Vec::new();
    let mut integer_validation_expected = Vec::new();

    for group in same_geometry_component_groups(components)? {
        let group_wavelets = integer_wavelets_for_component_group(
            &group,
            components,
            decomposition_levels,
            scratch,
            accelerator,
            timings,
        )?;
        for (component_index, wavelet) in group.into_iter().zip(group_wavelets) {
            let component = &components[component_index];
            let (x_rsiz, y_rsiz) = component_sampling[component_index];
            let actual_coefficients = flatten_integer_wavelet(&wavelet)?;
            precomputed_53[component_index] = Some(PrecomputedHtj2k53Component {
                x_rsiz,
                y_rsiz,
                dwt: j2k_dwt_from_integer_wavelet(&wavelet)?,
            });

            if options.validate_against_float_reference {
                try_extend_from_slice(&mut float_validation_actual, &actual_coefficients)?;
                let expected =
                    float_reference_coefficients(component, decomposition_levels, scratch)?;
                try_extend_from_slice(&mut float_validation_expected, &expected)?;
            }
            if options.validate_against_integer_reference {
                try_extend_from_slice(&mut integer_validation_actual, &actual_coefficients)?;
                let expected = integer_reference_coefficients(component, decomposition_levels)?;
                try_extend_from_slice(&mut integer_validation_expected, &expected)?;
            }
        }
    }

    let mut precomputed_components = try_vec_with_capacity(precomputed_53.len())?;
    for component in precomputed_53 {
        precomputed_components.push(component.ok_or(JpegToHtj2kError::Validation(
            "integer transcode did not produce all components",
        ))?);
    }

    finish_component_batch(
        PrecomputedComponentBatch::Dwt53(precomputed_components),
        ValidationCoefficientOwners {
            float_actual: float_validation_actual,
            float_expected: float_validation_expected,
            integer_actual: integer_validation_actual,
            integer_expected: integer_validation_expected,
        },
        options,
        scratch,
        retained_pipeline_bytes,
    )
}

struct ValidationCoefficientOwners {
    float_actual: Vec<i32>,
    float_expected: Vec<i32>,
    integer_actual: Vec<i32>,
    integer_expected: Vec<i32>,
}

fn finish_component_batch(
    precomputed_components: PrecomputedComponentBatch,
    owners: ValidationCoefficientOwners,
    options: &JpegToHtj2kOptions,
    scratch: &JpegToHtj2kScratch,
    retained_pipeline_bytes: usize,
) -> Result<ComponentTranscodeBatch, JpegToHtj2kError> {
    let ValidationCoefficientOwners {
        float_actual,
        float_expected,
        integer_actual,
        integer_expected,
    } = owners;

    let float_reference_metrics = if options.validate_against_float_reference {
        let mut budget =
            component_metrics_budget(&precomputed_components, scratch, retained_pipeline_bytes)?;
        for capacity in [
            float_actual.capacity(),
            float_expected.capacity(),
            integer_actual.capacity(),
            integer_expected.capacity(),
        ] {
            budget.add_capacity::<i32>(capacity)?;
        }
        Some(error_metrics_i32_with_live_budget(
            &float_actual,
            &float_expected,
            budget.live_bytes(),
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )?)
    } else {
        None
    };
    drop(float_actual);
    drop(float_expected);

    let integer_reference_metrics = if options.validate_against_integer_reference {
        let mut budget =
            component_metrics_budget(&precomputed_components, scratch, retained_pipeline_bytes)?;
        budget.add_capacity::<i32>(integer_actual.capacity())?;
        budget.add_capacity::<i32>(integer_expected.capacity())?;
        if let Some(metrics) = float_reference_metrics.as_ref() {
            budget.add_bytes(metrics.absolute_error_histogram.retained_bytes()?)?;
        }
        Some(error_metrics_i32_with_live_budget(
            &integer_actual,
            &integer_expected,
            budget.live_bytes(),
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )?)
    } else {
        None
    };

    Ok(ComponentTranscodeBatch {
        precomputed_components,
        float_reference_metrics,
        integer_reference_metrics,
    })
}

fn component_metrics_budget(
    precomputed_components: &PrecomputedComponentBatch,
    scratch: &JpegToHtj2kScratch,
    retained_pipeline_bytes: usize,
) -> Result<HostLiveBudget, JpegToHtj2kError> {
    let mut budget = HostLiveBudget::process_cap();
    budget.add_bytes(retained_pipeline_bytes)?;
    budget.add_bytes(scratch.retained_bytes()?)?;
    budget.add_bytes(precomputed_batch_retained_bytes(precomputed_components)?)?;
    Ok(budget)
}

pub(super) fn integer_wavelets_for_component_group(
    group: &[usize],
    components: &[JpegDctComponent],
    decomposition_levels: u8,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut impl DctToWaveletStageAccelerator,
    timings: &mut TranscodeTimingReport,
) -> Result<Vec<IntegerWavelet>, JpegToHtj2kError> {
    let mut jobs = try_vec_with_capacity(group.len())?;
    for &component_index in group {
        jobs.push(integer_dct_job_for_component(&components[component_index])?);
    }
    record_batch_attempt(timings, group.len());
    let accelerator_start = Instant::now();
    let accelerated_first_levels = accelerator
        .dct_grid_to_reversible_dwt53_batch(&jobs)
        .map_err(JpegToHtj2kError::Accelerator)?;
    timings.dct_to_wavelet_accelerator_us = timings
        .dct_to_wavelet_accelerator_us
        .saturating_add(accelerator_start.elapsed().as_micros());

    if let Some(first_levels) = accelerated_first_levels {
        if first_levels.len() != group.len() {
            return Err(JpegToHtj2kError::Validation(
                "reversible 5/3 batch accelerator returned wrong component count",
            ));
        }
        timings.component_count = timings.component_count.saturating_add(group.len());
        record_accelerator_dispatch(timings, group.len());
        let decompose_start = Instant::now();
        let mut wavelets = try_vec_with_capacity(first_levels.len())?;
        for first_level in first_levels {
            wavelets.push(integer_wavelet_from_first_level(
                first_level,
                decomposition_levels,
            )?);
        }
        timings.dwt_decompose_us = timings
            .dwt_decompose_us
            .saturating_add(decompose_start.elapsed().as_micros());
        return Ok(wavelets);
    }

    let mut wavelets = try_vec_with_capacity(group.len())?;
    for &component_index in group {
        wavelets.push(integer_direct_wavelet_from_component(
            &components[component_index],
            decomposition_levels,
            scratch,
            accelerator,
            timings,
        )?);
    }
    Ok(wavelets)
}

pub(super) fn integer_dct_job_for_component(
    component: &JpegDctComponent,
) -> Result<DctGridToReversibleDwt53Job<'_>, JpegToHtj2kError> {
    validate_component_block_grid(component)?;
    Ok(DctGridToReversibleDwt53Job {
        dequantized_blocks: &component.dequantized_blocks,
        block_cols: component.block_cols as usize,
        block_rows: component.block_rows as usize,
        width: component.width as usize,
        height: component.height as usize,
    })
}

#[derive(Clone, Copy)]
pub(super) struct ComponentTranscodePlan<'a> {
    pub(super) component: &'a JpegDctComponent,
    pub(super) x_rsiz: u8,
    pub(super) y_rsiz: u8,
    pub(super) decomposition_levels: u8,
    pub(super) options: &'a JpegToHtj2kOptions,
}

pub(super) fn component_to_precomputed_htj2k(
    plan: ComponentTranscodePlan<'_>,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut impl DctToWaveletStageAccelerator,
    timings: &mut TranscodeTimingReport,
) -> Result<ComponentTranscodeResult, JpegToHtj2kError> {
    let ComponentTranscodePlan {
        component,
        x_rsiz,
        y_rsiz,
        decomposition_levels,
        options,
    } = plan;
    let (dwt, actual_coefficients) = match options.coefficient_path {
        JpegToHtj2kCoefficientPath::IntegerDirect53 => {
            let wavelet = integer_direct_wavelet_from_component(
                component,
                decomposition_levels,
                scratch,
                accelerator,
                timings,
            )?;
            (
                PrecomputedComponent::Dwt53(PrecomputedHtj2k53Component {
                    x_rsiz,
                    y_rsiz,
                    dwt: j2k_dwt_from_integer_wavelet(&wavelet)?,
                }),
                flatten_integer_wavelet(&wavelet)?,
            )
        }
        JpegToHtj2kCoefficientPath::FloatDirectLinear53 => {
            let wavelet = float_direct_wavelet_from_component(
                component,
                decomposition_levels,
                scratch,
                accelerator,
                timings,
            )?;
            (
                PrecomputedComponent::Dwt53(PrecomputedHtj2k53Component {
                    x_rsiz,
                    y_rsiz,
                    dwt: j2k_dwt_from_wavelet(
                        &wavelet,
                        component.width as usize,
                        component.height as usize,
                    )?,
                }),
                rounded_wavelet_i32(&wavelet)?,
            )
        }
        JpegToHtj2kCoefficientPath::FloatDirectLinear97 => {
            let wavelet = float_direct_97_wavelet_from_component(
                component,
                decomposition_levels,
                scratch,
                accelerator,
                timings,
            )?;
            (
                PrecomputedComponent::Dwt97(PrecomputedHtj2k97Component {
                    x_rsiz,
                    y_rsiz,
                    dwt: j2k_dwt97_from_wavelet(
                        &wavelet,
                        component.width as usize,
                        component.height as usize,
                    )?,
                }),
                rounded_wavelet97_i32(&wavelet)?,
            )
        }
    };
    let float_validation_coefficients = if options.validate_against_float_reference {
        let expected = match options.coefficient_path {
            JpegToHtj2kCoefficientPath::FloatDirectLinear97 => {
                float97_reference_coefficients(component, decomposition_levels, scratch)?
            }
            JpegToHtj2kCoefficientPath::IntegerDirect53
            | JpegToHtj2kCoefficientPath::FloatDirectLinear53 => {
                float_reference_coefficients(component, decomposition_levels, scratch)?
            }
        };
        Some((try_vec_from_slice(&actual_coefficients)?, expected))
    } else {
        None
    };
    let integer_validation_coefficients = if options.validate_against_integer_reference {
        let expected = integer_reference_coefficients(component, decomposition_levels)?;
        Some((actual_coefficients, expected))
    } else {
        None
    };

    Ok(ComponentTranscodeResult {
        precomputed: dwt,
        float_validation_coefficients,
        integer_validation_coefficients,
    })
}

pub(super) fn transcode_path_name(
    all_unit_sampled: bool,
    coefficient_path: JpegToHtj2kCoefficientPath,
) -> &'static str {
    match (all_unit_sampled, coefficient_path) {
        (true, JpegToHtj2kCoefficientPath::IntegerDirect53) => {
            "full_resolution_components_integer_direct_53"
        }
        (false, JpegToHtj2kCoefficientPath::IntegerDirect53) => {
            "native_component_sampling_integer_direct_53"
        }
        (true, JpegToHtj2kCoefficientPath::FloatDirectLinear53) => {
            "full_resolution_components_float_direct_53"
        }
        (false, JpegToHtj2kCoefficientPath::FloatDirectLinear53) => {
            "native_component_sampling_float_direct_53"
        }
        (true, JpegToHtj2kCoefficientPath::FloatDirectLinear97) => {
            "full_resolution_components_float_direct_97"
        }
        (false, JpegToHtj2kCoefficientPath::FloatDirectLinear97) => {
            "native_component_sampling_float_direct_97"
        }
    }
}
