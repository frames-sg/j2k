// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    error::CudaError,
    j2k_decode::types::{CudaJ2kIdwtJob, CudaJ2kRect},
};

use super::{validate_idwt_job_layout, ValidatedIdwtJob};

fn rect(x0: u32, y0: u32, x1: u32, y1: u32) -> CudaJ2kRect {
    CudaJ2kRect { x0, y0, x1, y1 }
}

fn four_by_four_job() -> CudaJ2kIdwtJob {
    CudaJ2kIdwtJob {
        rect: rect(0, 0, 4, 4),
        ll_rect: rect(0, 0, 2, 2),
        hl_rect: rect(0, 0, 2, 2),
        lh_rect: rect(0, 0, 2, 2),
        hh_rect: rect(0, 0, 2, 2),
        irreversible97: 0,
    }
}

fn assert_invalid_argument_contains<T>(result: Result<T, CudaError>, expected: &str) {
    match result {
        Err(CudaError::InvalidArgument { message }) => {
            assert!(message.contains(expected), "unexpected error: {message}");
        }
        Err(error) => panic!("expected invalid argument containing {expected:?}, got {error}"),
        Ok(_) => panic!("expected invalid argument containing {expected:?}"),
    }
}

#[test]
fn full_validator_rejects_each_undersized_input_band() {
    for (index, name) in ["LL", "HL", "LH", "HH"].into_iter().enumerate() {
        let mut band_bytes = [16; 4];
        band_bytes[index] = 15;
        assert_invalid_argument_contains(
            validate_idwt_job_layout(band_bytes, None, four_by_four_job()),
            &format!("{name} buffer is too small: required 16 bytes, have 15"),
        );
    }
}

#[test]
fn full_validator_reports_an_undersized_target_output() {
    assert!(matches!(
        validate_idwt_job_layout([16; 4], Some(63), four_by_four_job()),
        Err(CudaError::OutputTooSmall {
            required: 64,
            have: 63
        })
    ));
}

#[test]
fn full_validator_rejects_inverted_output_and_band_rectangles() {
    let mut job = four_by_four_job();
    job.rect.x0 = 5;
    assert_invalid_argument_contains(
        validate_idwt_job_layout([16; 4], None, job),
        "output rectangle has inverted x bounds",
    );

    let mut job = four_by_four_job();
    job.hh_rect.y0 = 3;
    assert_invalid_argument_contains(
        validate_idwt_job_layout([16; 4], None, job),
        "HH rectangle has inverted y bounds",
    );
}

#[test]
fn full_validator_rejects_band_geometry_that_cannot_reconstruct_the_output() {
    let mut job = four_by_four_job();
    job.hl_rect.x1 = 1;
    assert_invalid_argument_contains(
        validate_idwt_job_layout([16; 4], None, job),
        "HL geometry 1x2 does not match the 4x4 output geometry; expected 2x2",
    );
}

#[test]
fn full_validator_accepts_valid_two_by_eight_and_eight_by_two_jobs() {
    let two_by_eight = CudaJ2kIdwtJob {
        rect: rect(0, 0, 2, 8),
        ll_rect: rect(0, 0, 1, 4),
        hl_rect: rect(0, 0, 1, 4),
        lh_rect: rect(0, 0, 1, 4),
        hh_rect: rect(0, 0, 1, 4),
        irreversible97: 0,
    };
    let eight_by_two = CudaJ2kIdwtJob {
        rect: rect(0, 0, 8, 2),
        ll_rect: rect(0, 0, 4, 1),
        hl_rect: rect(0, 0, 4, 1),
        lh_rect: rect(0, 0, 4, 1),
        hh_rect: rect(0, 0, 4, 1),
        irreversible97: 1,
    };

    assert_eq!(
        validate_idwt_job_layout([16; 4], None, two_by_eight).expect("valid 2x8 IDWT job"),
        ValidatedIdwtJob {
            width: 2,
            height: 8,
            output_bytes: 64,
        }
    );
    assert_eq!(
        validate_idwt_job_layout([16; 4], None, eight_by_two).expect("valid 8x2 IDWT job"),
        ValidatedIdwtJob {
            width: 8,
            height: 2,
            output_bytes: 64,
        }
    );
}

#[test]
fn full_validator_accepts_one_by_n_jobs_for_even_and_odd_origins() {
    let even_origin = CudaJ2kIdwtJob {
        rect: rect(0, 0, 1, 7),
        ll_rect: rect(4, 2, 5, 6),
        hl_rect: rect(9, 2, 9, 6),
        lh_rect: rect(4, 7, 5, 10),
        hh_rect: rect(9, 7, 9, 10),
        irreversible97: 0,
    };
    let odd_origin = CudaJ2kIdwtJob {
        rect: rect(1, 1, 2, 8),
        ll_rect: rect(9, 4, 9, 7),
        hl_rect: rect(4, 4, 5, 7),
        lh_rect: rect(9, 8, 9, 12),
        hh_rect: rect(4, 8, 5, 12),
        irreversible97: 1,
    };

    assert!(validate_idwt_job_layout([16, 0, 12, 0], None, even_origin).is_ok());
    assert!(validate_idwt_job_layout([0, 12, 0, 16], None, odd_origin).is_ok());
}

#[test]
fn full_validator_rejects_linear_index_overflow() {
    let job = CudaJ2kIdwtJob {
        rect: rect(0, 0, 65_537, 65_536),
        ll_rect: rect(0, 0, 32_769, 32_768),
        hl_rect: rect(0, 0, 32_768, 32_768),
        lh_rect: rect(0, 0, 32_769, 32_768),
        hh_rect: rect(0, 0, 32_768, 32_768),
        irreversible97: 0,
    };
    assert_invalid_argument_contains(
        validate_idwt_job_layout([usize::MAX; 4], None, job),
        "output sample count 4295032832 exceeds the CUDA u32 indexing ABI",
    );
}

#[test]
fn full_validator_rejects_a_max_width_that_overflows_device_iteration() {
    let job = CudaJ2kIdwtJob {
        rect: rect(0, 0, u32::MAX, 1),
        ll_rect: CudaJ2kRect::default(),
        hl_rect: CudaJ2kRect::default(),
        lh_rect: CudaJ2kRect::default(),
        hh_rect: CudaJ2kRect::default(),
        irreversible97: 0,
    };
    assert_invalid_argument_contains(
        validate_idwt_job_layout([usize::MAX; 4], None, job),
        "exceeds the CUDA u32 iteration ABI",
    );
}

#[test]
fn full_validator_accepts_a_coherent_empty_job_without_storage() {
    assert_eq!(
        validate_idwt_job_layout(
            [0; 4],
            Some(0),
            CudaJ2kIdwtJob {
                rect: rect(7, 9, 7, 9),
                ll_rect: rect(3, 4, 3, 4),
                hl_rect: rect(3, 4, 3, 4),
                lh_rect: rect(3, 4, 3, 4),
                hh_rect: rect(3, 4, 3, 4),
                irreversible97: u32::MAX,
            },
        )
        .expect("coherent empty IDWT job"),
        ValidatedIdwtJob {
            width: 0,
            height: 0,
            output_bytes: 0,
        }
    );
}
