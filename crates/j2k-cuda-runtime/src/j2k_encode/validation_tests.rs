// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    error::CudaError,
    j2k_encode::{validate_quantize_region, CudaJ2kQuantizeJob, CudaJ2kQuantizeSubbandRegionJob},
};

const QUANTIZATION: CudaJ2kQuantizeJob = CudaJ2kQuantizeJob {
    step_exponent: 0,
    step_mantissa: 0,
    range_bits: 8,
    reversible: true,
};

#[test]
fn empty_region_requires_no_source_storage() {
    let job = CudaJ2kQuantizeSubbandRegionJob {
        x0: u32::MAX,
        y0: u32::MAX,
        width: 0,
        height: 1,
        stride: 0,
        quantization: QUANTIZATION,
    };

    assert!(validate_quantize_region(job, 0).is_ok());
}

#[test]
fn region_validation_reports_exact_required_and_available_bytes() {
    let job = CudaJ2kQuantizeSubbandRegionJob {
        x0: 1,
        y0: 1,
        width: 2,
        height: 2,
        stride: 4,
        quantization: QUANTIZATION,
    };

    assert!(matches!(
        validate_quantize_region(job, 10),
        Err(CudaError::OutputTooSmall {
            required: 44,
            have: 40,
        })
    ));
}

#[test]
fn region_validation_rejects_rows_wider_than_the_stride() {
    let job = CudaJ2kQuantizeSubbandRegionJob {
        x0: 3,
        y0: 0,
        width: 2,
        height: 1,
        stride: 4,
        quantization: QUANTIZATION,
    };

    assert!(matches!(
        validate_quantize_region(job, 8),
        Err(CudaError::LengthTooLarge { len: 8 })
    ));
}
