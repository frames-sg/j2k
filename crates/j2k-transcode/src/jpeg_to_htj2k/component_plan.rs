// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    error_metrics_i32, flatten_integer_wavelet, float97_reference_coefficients,
    float_direct_97_wavelet_from_component, float_direct_wavelet_from_component,
    float_reference_coefficients, integer_direct_wavelet_from_component,
    integer_reference_coefficients, integer_wavelet_from_first_level, j2k_dwt97_from_wavelet,
    j2k_dwt_from_integer_wavelet, j2k_dwt_from_wavelet, record_accelerator_dispatch,
    record_batch_attempt, rounded_wavelet97_i32, rounded_wavelet_i32,
    validate_component_block_grid, DctGridToReversibleDwt53Job, DctToWaveletStageAccelerator,
    Instant, IntegerWavelet, JpegDctComponent, JpegToHtj2kCoefficientPath, JpegToHtj2kError,
    JpegToHtj2kOptions, JpegToHtj2kScratch, PrecomputedHtj2k53Component,
    PrecomputedHtj2k97Component, TranscodeTimingReport, TranscodeValidationMetrics,
};

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

pub(super) fn transcode_component_batch(
    components: &[JpegDctComponent],
    component_sampling: &[(u8, u8)],
    decomposition_levels: u8,
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut impl DctToWaveletStageAccelerator,
    timings: &mut TranscodeTimingReport,
) -> Result<ComponentTranscodeBatch, JpegToHtj2kError> {
    if matches!(
        options.coefficient_path,
        JpegToHtj2kCoefficientPath::FloatDirectLinear97
    ) && options.validate_against_integer_reference
    {
        return Err(JpegToHtj2kError::Unsupported(
            "integer reversible validation is only defined for 5/3 coefficient paths",
        ));
    }

    if matches!(
        options.coefficient_path,
        JpegToHtj2kCoefficientPath::IntegerDirect53
    ) {
        return transcode_integer_component_batch(
            components,
            component_sampling,
            decomposition_levels,
            options,
            scratch,
            accelerator,
            timings,
        );
    }

    let mut precomputed_53 = Vec::with_capacity(components.len());
    let mut precomputed_97 = Vec::with_capacity(components.len());
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
            float_validation_actual.extend(actual);
            float_validation_expected.extend(expected);
        }
        if let Some((actual, expected)) = component_result.integer_validation_coefficients {
            integer_validation_actual.extend(actual);
            integer_validation_expected.extend(expected);
        }
    }

    let float_reference_metrics = if options.validate_against_float_reference {
        Some(error_metrics_i32(
            &float_validation_actual,
            &float_validation_expected,
        )?)
    } else {
        None
    };
    let integer_reference_metrics = if options.validate_against_integer_reference {
        Some(error_metrics_i32(
            &integer_validation_actual,
            &integer_validation_expected,
        )?)
    } else {
        None
    };

    let precomputed_components = if matches!(
        options.coefficient_path,
        JpegToHtj2kCoefficientPath::FloatDirectLinear97
    ) {
        PrecomputedComponentBatch::Dwt97(precomputed_97)
    } else {
        PrecomputedComponentBatch::Dwt53(precomputed_53)
    };

    Ok(ComponentTranscodeBatch {
        precomputed_components,
        float_reference_metrics,
        integer_reference_metrics,
    })
}

pub(super) fn transcode_integer_component_batch(
    components: &[JpegDctComponent],
    component_sampling: &[(u8, u8)],
    decomposition_levels: u8,
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut impl DctToWaveletStageAccelerator,
    timings: &mut TranscodeTimingReport,
) -> Result<ComponentTranscodeBatch, JpegToHtj2kError> {
    let mut precomputed_53: Vec<Option<PrecomputedHtj2k53Component>> =
        (0..components.len()).map(|_| None).collect();
    let mut float_validation_actual = Vec::new();
    let mut float_validation_expected = Vec::new();
    let mut integer_validation_actual = Vec::new();
    let mut integer_validation_expected = Vec::new();

    for group in same_geometry_component_groups(components) {
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
            let actual_coefficients = flatten_integer_wavelet(&wavelet);
            precomputed_53[component_index] = Some(PrecomputedHtj2k53Component {
                x_rsiz,
                y_rsiz,
                dwt: j2k_dwt_from_integer_wavelet(&wavelet),
            });

            if options.validate_against_float_reference {
                float_validation_actual.extend(actual_coefficients.clone());
                float_validation_expected.extend(float_reference_coefficients(
                    component,
                    decomposition_levels,
                    scratch,
                )?);
            }
            if options.validate_against_integer_reference {
                integer_validation_actual.extend(actual_coefficients);
                integer_validation_expected.extend(integer_reference_coefficients(
                    component,
                    decomposition_levels,
                )?);
            }
        }
    }

    let float_reference_metrics = if options.validate_against_float_reference {
        Some(error_metrics_i32(
            &float_validation_actual,
            &float_validation_expected,
        )?)
    } else {
        None
    };
    let integer_reference_metrics = if options.validate_against_integer_reference {
        Some(error_metrics_i32(
            &integer_validation_actual,
            &integer_validation_expected,
        )?)
    } else {
        None
    };
    let precomputed_components = precomputed_53
        .into_iter()
        .map(|component| {
            component.ok_or(JpegToHtj2kError::Validation(
                "integer transcode did not produce all components",
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(ComponentTranscodeBatch {
        precomputed_components: PrecomputedComponentBatch::Dwt53(precomputed_components),
        float_reference_metrics,
        integer_reference_metrics,
    })
}

pub(super) fn integer_wavelets_for_component_group(
    group: &[usize],
    components: &[JpegDctComponent],
    decomposition_levels: u8,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut impl DctToWaveletStageAccelerator,
    timings: &mut TranscodeTimingReport,
) -> Result<Vec<IntegerWavelet>, JpegToHtj2kError> {
    let jobs = group
        .iter()
        .map(|&component_index| integer_dct_job_for_component(&components[component_index]))
        .collect::<Result<Vec<_>, _>>()?;
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
        let wavelets = first_levels
            .into_iter()
            .map(|first_level| integer_wavelet_from_first_level(first_level, decomposition_levels))
            .collect();
        timings.dwt_decompose_us = timings
            .dwt_decompose_us
            .saturating_add(decompose_start.elapsed().as_micros());
        return Ok(wavelets);
    }

    group
        .iter()
        .map(|&component_index| {
            integer_direct_wavelet_from_component(
                &components[component_index],
                decomposition_levels,
                scratch,
                accelerator,
                timings,
            )
        })
        .collect()
}

pub(super) fn same_geometry_component_groups(components: &[JpegDctComponent]) -> Vec<Vec<usize>> {
    let mut assigned = vec![false; components.len()];
    let mut groups = Vec::new();

    for component_index in 0..components.len() {
        if assigned[component_index] {
            continue;
        }
        assigned[component_index] = true;
        let mut group = vec![component_index];
        for candidate_index in component_index + 1..components.len() {
            if !assigned[candidate_index]
                && same_component_geometry(
                    &components[component_index],
                    &components[candidate_index],
                )
            {
                assigned[candidate_index] = true;
                group.push(candidate_index);
            }
        }
        groups.push(group);
    }

    groups
}

pub(super) fn same_component_geometry(left: &JpegDctComponent, right: &JpegDctComponent) -> bool {
    left.width == right.width
        && left.height == right.height
        && left.block_cols == right.block_cols
        && left.block_rows == right.block_rows
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
                    dwt: j2k_dwt_from_integer_wavelet(&wavelet),
                }),
                flatten_integer_wavelet(&wavelet),
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
                    ),
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
                    ),
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
        Some((actual_coefficients.clone(), expected))
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
