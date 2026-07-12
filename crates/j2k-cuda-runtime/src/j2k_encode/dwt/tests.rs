// SPDX-License-Identifier: MIT OR Apache-2.0

mod launch_geometry;

use j2k_codec_math::dwt::max_decomposition_levels;

use crate::{
    error::CudaError,
    execution::CudaExecutionStats,
    j2k_encode::{CudaDwt53LevelShape, CudaJ2kResidentComponents},
    CudaContext,
};

use super::validation::{FORWARD_DWT_LEVELS_EXCEED_GEOMETRY, FORWARD_DWT_SAMPLES_EXCEED_INDEX_ABI};

fn samples(width: u32, height: u32) -> Vec<f32> {
    let count = usize::try_from(width)
        .expect("test width fits usize")
        .checked_mul(usize::try_from(height).expect("test height fits usize"))
        .expect("test sample count");
    (0..count)
        .map(|index| {
            let value =
                u16::try_from((index * 13 + index / 3 + 5) % 97).expect("test sample fits u16");
            f32::from(value) - 48.0
        })
        .collect()
}

fn resident_components(context: &CudaContext, samples: &[f32]) -> CudaJ2kResidentComponents {
    CudaJ2kResidentComponents {
        buffer: context.upload_f32(samples).expect("resident DWT samples"),
        num_pixels: samples.len(),
        num_components: 1,
        execution: CudaExecutionStats::default(),
    }
}

fn expected_ll_dimensions(mut width: u32, mut height: u32, levels: u8) -> (u32, u32) {
    for _ in 0..levels {
        width = width.div_ceil(2);
        height = height.div_ceil(2);
    }
    (width, height)
}

fn assert_level_shapes(levels: &[CudaDwt53LevelShape], width: u32, height: u32) {
    let mut current = (width, height);
    for level in levels {
        assert_eq!((level.width, level.height), current);
        assert_eq!(
            (level.low_width, level.low_height),
            (current.0.div_ceil(2), current.1.div_ceil(2))
        );
        assert_eq!(
            (level.high_width, level.high_height),
            (current.0 / 2, current.1 / 2)
        );
        current = (level.low_width, level.low_height);
    }
}

fn assert_invalid_levels<T>(result: Result<T, CudaError>, width: u32, height: u32, requested: u8) {
    let maximum = max_decomposition_levels(width, height);
    match result {
        Err(CudaError::InvalidArgument { message }) => assert_eq!(
            message,
            format!(
                "{FORWARD_DWT_LEVELS_EXCEED_GEOMETRY}: requested {requested}, maximum {maximum} for {width}x{height}"
            )
        ),
        Err(error) => panic!("expected invalid forward DWT levels, got {error}"),
        Ok(_) => panic!("expected invalid forward DWT levels"),
    }
}

fn assert_invalid_indexing<T>(result: Result<T, CudaError>, width: u32, height: u32) {
    let samples = u64::from(width) * u64::from(height);
    match result {
        Err(CudaError::InvalidArgument { message }) => assert_eq!(
            message,
            format!(
                "{FORWARD_DWT_SAMPLES_EXCEED_INDEX_ABI}: {samples} samples for {width}x{height}"
            )
        ),
        Err(error) => panic!("expected invalid forward DWT indexing, got {error}"),
        Ok(_) => panic!("expected invalid forward DWT indexing"),
    }
}

#[test]
fn safe_host_and_resident_dwt_apis_reject_degenerate_later_levels() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let context = CudaContext::system_default().expect("CUDA context");

    for (width, height, requested) in [
        (2, 8, 2),
        (8, 2, 2),
        (1, 8, 1),
        (1, 7, 1),
        (8, 1, 1),
        (7, 1, 1),
    ] {
        let input = samples(width, height);
        assert_invalid_levels(
            context.j2k_forward_dwt53(&input, width, height, requested),
            width,
            height,
            requested,
        );
        assert_invalid_levels(
            context.j2k_forward_dwt97(&input, width, height, requested),
            width,
            height,
            requested,
        );

        let components = resident_components(&context, &input);
        assert_invalid_levels(
            context.j2k_forward_dwt53_resident_component(&components, 0, width, height, requested),
            width,
            height,
            requested,
        );
        assert_invalid_levels(
            context.j2k_forward_dwt97_resident_component(&components, 0, width, height, requested),
            width,
            height,
            requested,
        );
    }
}

#[test]
fn resident_level_validation_precedes_component_copy() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let context = CudaContext::system_default().expect("CUDA context");
    let width = 1;
    let height = 7;
    let requested = 1;
    let input = samples(width, height);
    let components = resident_components(&context, &input);

    assert_invalid_levels(
        context.j2k_forward_dwt53_resident_component(
            &components,
            u8::MAX,
            width,
            height,
            requested,
        ),
        width,
        height,
        requested,
    );
    assert_invalid_levels(
        context.j2k_forward_dwt97_resident_component(
            &components,
            u8::MAX,
            width,
            height,
            requested,
        ),
        width,
        height,
        requested,
    );
}

#[test]
fn resident_index_validation_precedes_component_copy() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let context = CudaContext::system_default().expect("CUDA context");
    let width = 65_536;
    let height = 65_537;
    let num_pixels = usize::try_from(u64::from(width) * u64::from(height))
        .expect("CUDA host represents the test sample count");
    let components = CudaJ2kResidentComponents {
        buffer: context.allocate(0).expect("empty resident buffer"),
        num_pixels,
        num_components: 1,
        execution: CudaExecutionStats::default(),
    };

    assert_invalid_indexing(
        context.j2k_forward_dwt53_resident_component(&components, u8::MAX, width, height, 1),
        width,
        height,
    );
    assert_invalid_indexing(
        context.j2k_forward_dwt97_resident_component(&components, u8::MAX, width, height, 1),
        width,
        height,
    );
}

#[test]
fn boundary_valid_53_and_97_levels_produce_complete_host_and_resident_outputs() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let context = CudaContext::system_default().expect("CUDA context");

    for (width, height, levels) in [(2, 8, 1), (8, 2, 1), (7, 9, 2), (8, 8, 3)] {
        assert_eq!(levels, max_decomposition_levels(width, height));
        let input = samples(width, height);
        let expected_ll = expected_ll_dimensions(width, height, levels);
        let expected_dispatches = usize::from(levels) * 2;

        let host53 = context
            .j2k_forward_dwt53(&input, width, height, levels)
            .expect("boundary-valid host 5/3 DWT");
        let host97 = context
            .j2k_forward_dwt97(&input, width, height, levels)
            .expect("boundary-valid host 9/7 DWT");
        assert_eq!(host53.ll_dimensions(), expected_ll);
        assert_eq!(host97.ll_dimensions(), expected_ll);
        assert_eq!(host53.levels().len(), usize::from(levels));
        assert_eq!(host97.levels().len(), usize::from(levels));
        assert_level_shapes(host53.levels(), width, height);
        assert_level_shapes(host97.levels(), width, height);
        assert_eq!(host53.execution().kernel_dispatches(), expected_dispatches);
        assert_eq!(host97.execution().kernel_dispatches(), expected_dispatches);
        assert!(host53.transformed().iter().all(|sample| sample.is_finite()));
        assert!(host97.transformed().iter().all(|sample| sample.is_finite()));

        let components = resident_components(&context, &input);
        let resident53 = context
            .j2k_forward_dwt53_resident_component(&components, 0, width, height, levels)
            .expect("boundary-valid resident 5/3 DWT");
        let resident97 = context
            .j2k_forward_dwt97_resident_component(&components, 0, width, height, levels)
            .expect("boundary-valid resident 9/7 DWT");
        assert_eq!(resident53.ll_dimensions(), expected_ll);
        assert_eq!(resident97.ll_dimensions(), expected_ll);
        assert_eq!(resident53.levels(), host53.levels());
        assert_eq!(resident97.levels(), host97.levels());
        assert_eq!(
            resident53.execution().kernel_dispatches(),
            expected_dispatches + 1
        );
        assert_eq!(
            resident97.execution().kernel_dispatches(),
            expected_dispatches + 1
        );
        assert_eq!(resident53.execution().copy_kernel_dispatches(), 1);
        assert_eq!(resident97.execution().copy_kernel_dispatches(), 1);
        let resident53_samples = resident53
            .download_transformed()
            .expect("resident 5/3 readback");
        assert_eq!(resident53_samples.as_slice(), host53.transformed());
        let resident97_samples = resident97
            .download_transformed()
            .expect("resident 9/7 readback");
        for (resident, host) in resident97_samples.iter().zip(host97.transformed()) {
            assert!((resident - host).abs() < 0.000_001);
        }
    }
}
