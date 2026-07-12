// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    error::CudaError, execution::CudaExecutionStats, j2k_encode::CudaJ2kResidentComponents,
    CudaContext,
};

use super::super::validation::FORWARD_DWT_GEOMETRY_EXCEEDS_LAUNCH_LIMITS;

#[test]
fn resident_launch_geometry_validation_precedes_component_copy() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let context = CudaContext::system_default().expect("CUDA context");
    let width = 2;
    let height = 65_535 * 16 + 1;
    let num_pixels = usize::try_from(u64::from(width) * u64::from(height))
        .expect("CUDA host represents the test sample count");
    let components = CudaJ2kResidentComponents {
        buffer: context.allocate(0).expect("empty resident buffer"),
        num_pixels,
        num_components: 1,
        execution: CudaExecutionStats::default(),
    };
    let expected = format!("{FORWARD_DWT_GEOMETRY_EXCEEDS_LAUNCH_LIMITS}: {width}x{height}");

    for error in [
        context
            .j2k_forward_dwt53_resident_component(&components, u8::MAX, width, height, 1)
            .expect_err("5/3 launch geometry"),
        context
            .j2k_forward_dwt97_resident_component(&components, u8::MAX, width, height, 1)
            .expect_err("9/7 launch geometry"),
    ] {
        match error {
            CudaError::InvalidArgument { message } => assert_eq!(message, expected),
            other => panic!("expected invalid DWT launch geometry, got {other}"),
        }
    }
}
