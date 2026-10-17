// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "parallel")]
use super::prepare_referenced_direct_scratch;
use super::{
    component_band_count, include_plan_allocations, DirectAllocationBudget, DirectWorkspaceBudget,
};
use crate::error::{DecodeError, ValidationError};
#[cfg(feature = "parallel")]
use crate::{
    HtCodeBlockDecodeWorkspace, HtOwnedCodeBlockBatchJob, J2kDirectCpuScratch,
    J2kReferencedHtj2kPlan, J2kReferencedPayloadRecordSpan, J2kReferencedTileGeometry,
    J2kReferencedTilePlan,
};
use crate::{
    HtOwnedSubBandPlan, J2kDirectGrayscalePlan, J2kDirectGrayscaleStep, J2kDirectIdwtStep,
    J2kDirectStoreStep, J2kOwnedSubBandPlan, J2kRect, J2kWaveletTransform,
    DEFAULT_MAX_DECODE_BYTES,
};

#[test]
fn aggregate_budget_has_an_exact_shared_cap_boundary() {
    let mut budget = DirectAllocationBudget {
        bytes: DEFAULT_MAX_DECODE_BYTES - 1,
    };
    budget.include_bytes(1).expect("exact boundary fits");
    assert_eq!(
        budget.include_bytes(1),
        Err(DecodeError::Validation(ValidationError::ImageTooLarge))
    );
}

#[test]
fn actual_scalar_workspace_uses_the_remaining_direct_budget() {
    let budget = DirectWorkspaceBudget {
        base_bytes: DEFAULT_MAX_DECODE_BYTES - 1,
        peak_bytes: DEFAULT_MAX_DECODE_BYTES - 1,
    };
    budget.validate_workspace(1).expect("exact boundary fits");
    assert_eq!(
        budget.validate_workspace(2),
        Err(DecodeError::Validation(ValidationError::ImageTooLarge))
    );
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one boundary regression compares serial and parallel IDWT accounting"
)]
fn direct_plan_peak_counts_serial_and_parallel_banded_idwt_scratch() {
    let rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 1_031,
        y1: 133,
    };
    let half = |high_x: bool, high_y: bool| J2kRect {
        x0: 0,
        y0: 0,
        x1: if high_x {
            rect.width() / 2
        } else {
            rect.width().div_ceil(2)
        },
        y1: if high_y {
            rect.height() / 2
        } else {
            rect.height().div_ceil(2)
        },
    };
    let plan = J2kDirectGrayscalePlan {
        dimensions: (rect.width(), rect.height()),
        bit_depth: 8,
        steps: vec![
            J2kDirectGrayscaleStep::Idwt(J2kDirectIdwtStep {
                output_band_id: 5,
                rect,
                transform: J2kWaveletTransform::Reversible53,
                ll_band_id: 1,
                ll: half(false, false),
                hl_band_id: 2,
                hl: half(true, false),
                lh_band_id: 3,
                lh: half(false, true),
                hh_band_id: 4,
                hh: half(true, true),
            }),
            J2kDirectGrayscaleStep::Store(J2kDirectStoreStep {
                input_band_id: 5,
                input_rect: rect,
                source_x: 0,
                source_y: 0,
                copy_width: rect.width(),
                copy_height: rect.height(),
                output_width: rect.width(),
                output_height: rect.height(),
                output_x: 0,
                output_y: 0,
                addend: 0.0,
            }),
        ],
    };
    // One serial stream holds the two band rows plus two ghosts per row.
    let serial_bytes = (1_031_usize + 4) * core::mem::size_of::<f32>();
    assert_eq!(serial_bytes, 4_140);

    let mut budget = DirectAllocationBudget::default();
    let temporary = include_plan_allocations(&mut budget, core::slice::from_ref(&plan), 0)
        .expect("plan allocation");

    assert_eq!(temporary, serial_bytes);
    let serial_boundary = DirectWorkspaceBudget {
        base_bytes: DEFAULT_MAX_DECODE_BYTES - serial_bytes,
        peak_bytes: DEFAULT_MAX_DECODE_BYTES,
    };
    serial_boundary
        .validate_workspace(serial_bytes)
        .expect("exact serial banded scratch boundary fits");
    assert_eq!(
        serial_boundary.validate_workspace(serial_bytes + 1),
        Err(DecodeError::Validation(ValidationError::ImageTooLarge))
    );

    #[cfg(feature = "parallel")]
    {
        let tile = J2kReferencedTilePlan::new(
            0,
            rect,
            rect,
            J2kReferencedPayloadRecordSpan {
                first_record: 0,
                record_count: 0,
            },
            Vec::new(),
            Vec::new(),
            J2kWaveletTransform::Reversible53,
            J2kReferencedTileGeometry::Grayscale(plan),
        );
        let referenced = J2kReferencedHtj2kPlan::grayscale(
            vec![tile],
            (rect.width(), rect.height()),
            rect,
            Vec::new(),
        );
        let mut serial_scratch = J2kDirectCpuScratch::new();
        let serial_peak = prepare_referenced_direct_scratch(&referenced, &mut serial_scratch)
            .expect("serial referenced budget")
            .peak_bytes();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("four-thread pool");
        // Four Rayon threads and 133 rows produce three concurrent strips.
        // Each owns six ring rows plus the serial horizontal scratch.
        let parallel_bytes = 3 * (6 * 1_031_usize + 1_031 + 4) * core::mem::size_of::<f32>();
        assert_eq!(parallel_bytes, 86_652);
        let mut parallel_scratch = J2kDirectCpuScratch::new();
        parallel_scratch.set_parallel_decode(true);
        let parallel_peak = pool
            .install(|| prepare_referenced_direct_scratch(&referenced, &mut parallel_scratch))
            .expect("parallel referenced budget")
            .peak_bytes();
        assert_eq!(parallel_peak - serial_peak, parallel_bytes - serial_bytes);

        let parallel_boundary = DirectWorkspaceBudget {
            base_bytes: DEFAULT_MAX_DECODE_BYTES - parallel_bytes,
            peak_bytes: DEFAULT_MAX_DECODE_BYTES,
        };
        parallel_boundary
            .validate_workspace(parallel_bytes)
            .expect("exact parallel banded scratch boundary fits");
        assert_eq!(
            parallel_boundary.validate_workspace(parallel_bytes + 1),
            Err(DecodeError::Validation(ValidationError::ImageTooLarge))
        );
    }
}

#[cfg(feature = "parallel")]
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one boundary regression covers dispatch and retained parallel owners together"
)]
fn referenced_parallel_peak_counts_ht_dispatch_and_retained_task_owners() {
    const JOB_COUNT: u32 = 16;
    let job_count = usize::try_from(JOB_COUNT).expect("job count fits usize");
    let rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: JOB_COUNT,
        y1: 1,
    };
    let jobs = (0..JOB_COUNT)
        .map(|output_x| HtOwnedCodeBlockBatchJob {
            output_x,
            output_y: 0,
            data: Vec::new(),
            cleanup_length: 0,
            refinement_length: 0,
            width: 1,
            height: 1,
            output_stride: job_count,
            missing_bit_planes: 0,
            number_of_coding_passes: 1,
            num_bitplanes: 1,
            roi_shift: 0,
            stripe_causal: false,
            strict: true,
            dequantization_step: 1.0,
        })
        .collect();
    let component = J2kDirectGrayscalePlan {
        dimensions: (JOB_COUNT, 1),
        bit_depth: 8,
        steps: vec![
            J2kDirectGrayscaleStep::HtSubBand(HtOwnedSubBandPlan {
                band_id: 1,
                rect,
                width: JOB_COUNT,
                height: 1,
                irreversible_midpoint: false,
                jobs,
            }),
            J2kDirectGrayscaleStep::Store(J2kDirectStoreStep {
                input_band_id: 1,
                input_rect: rect,
                source_x: 0,
                source_y: 0,
                copy_width: rect.width(),
                copy_height: rect.height(),
                output_width: rect.width(),
                output_height: rect.height(),
                output_x: 0,
                output_y: 0,
                addend: 0.0,
            }),
        ],
    };
    let tile = J2kReferencedTilePlan::new(
        0,
        rect,
        rect,
        J2kReferencedPayloadRecordSpan {
            first_record: 0,
            record_count: 0,
        },
        Vec::new(),
        Vec::new(),
        J2kWaveletTransform::Reversible53,
        J2kReferencedTileGeometry::Grayscale(component),
    );
    let referenced = J2kReferencedHtj2kPlan::grayscale(
        vec![tile],
        (rect.width(), rect.height()),
        rect,
        Vec::new(),
    );

    let mut serial_scratch = J2kDirectCpuScratch::new();
    let serial = prepare_referenced_direct_scratch(&referenced, &mut serial_scratch)
        .expect("serial referenced budget");
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("four-thread pool");
    let mut parallel_scratch = J2kDirectCpuScratch::new();
    parallel_scratch.set_parallel_decode(true);
    let parallel = pool
        .install(|| prepare_referenced_direct_scratch(&referenced, &mut parallel_scratch))
        .expect("parallel referenced budget");

    let dispatch_bytes =
        super::super::referenced::parallel_ht_decode_transient_bytes(job_count, job_count, 4)
            .expect("dispatch bytes");
    assert_eq!(parallel.peak_bytes - parallel.base_bytes, dispatch_bytes);
    assert!(dispatch_bytes > serial.peak_bytes - serial.base_bytes);

    let retained_tasks = parallel_scratch
        .ht_task_workspaces
        .iter()
        .try_fold(
            parallel_scratch.ht_task_workspaces.capacity()
                * core::mem::size_of::<HtCodeBlockDecodeWorkspace>(),
            |bytes, workspace| {
                workspace
                    .allocated_bytes()
                    .map(|workspace_bytes| bytes + workspace_bytes)
            },
        )
        .expect("retained task bytes");
    assert_eq!(parallel.base_bytes - serial.base_bytes, retained_tasks);

    let boundary = DirectWorkspaceBudget {
        base_bytes: DEFAULT_MAX_DECODE_BYTES - dispatch_bytes,
        peak_bytes: DEFAULT_MAX_DECODE_BYTES,
    };
    boundary
        .validate_workspace(dispatch_bytes)
        .expect("exact dispatch boundary fits");
    assert_eq!(
        boundary.validate_workspace(dispatch_bytes + 1),
        Err(DecodeError::Validation(ValidationError::ImageTooLarge))
    );
}

#[test]
fn mixed_entropy_steps_reserve_one_shared_band_owner() {
    let rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 2,
        y1: 2,
    };
    let plan = J2kDirectGrayscalePlan {
        dimensions: (2, 2),
        bit_depth: 8,
        steps: vec![
            J2kDirectGrayscaleStep::ClassicSubBand(J2kOwnedSubBandPlan {
                band_id: 7,
                rect,
                width: 2,
                height: 2,
                irreversible_midpoint: false,
                jobs: Vec::new(),
            }),
            J2kDirectGrayscaleStep::HtSubBand(HtOwnedSubBandPlan {
                band_id: 7,
                rect,
                width: 2,
                height: 2,
                irreversible_midpoint: false,
                jobs: Vec::new(),
            }),
        ],
    };

    assert_eq!(component_band_count(&plan).expect("band count"), 1);
}
