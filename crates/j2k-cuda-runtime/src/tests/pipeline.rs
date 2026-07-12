// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn idwt_cooperative_53_selection_requires_large_reversible_batches() {
    let mut kernel_job = CudaJ2kIdwtMultiKernelJob {
        ll_ptr: 0,
        hl_ptr: 0,
        lh_ptr: 0,
        hh_ptr: 0,
        output_ptr: 0,
        job: CudaJ2kIdwtJob {
            rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
            },
            ll_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
            },
            hl_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
            },
            lh_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
            },
            hh_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
            },
            irreversible97: 0,
        },
        reserved_tail: 0,
    };

    assert!(!idwt_batch_uses_cooperative_53(&[kernel_job], 127, 128));
    assert!(!idwt_batch_uses_cooperative_53(&[kernel_job], 128, 127));
    assert!(idwt_batch_uses_cooperative_53(&[kernel_job], 128, 128));
    assert!(idwt_batch_uses_cooperative_53(&[kernel_job], 512, 512));
    assert!(!idwt_batch_uses_cooperative_53(&[kernel_job], 513, 128));
    kernel_job.job.irreversible97 = 1;
    assert!(!idwt_batch_uses_cooperative_53(&[kernel_job], 128, 128));
}

#[test]
fn idwt_cooperative_97_selection_requires_large_irreversible_batches() {
    let mut kernel_job = CudaJ2kIdwtMultiKernelJob {
        ll_ptr: 0,
        hl_ptr: 0,
        lh_ptr: 0,
        hh_ptr: 0,
        output_ptr: 0,
        job: CudaJ2kIdwtJob {
            rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
            },
            ll_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
            },
            hl_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
            },
            lh_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
            },
            hh_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
            },
            irreversible97: 1,
        },
        reserved_tail: 0,
    };

    assert_eq!(
        idwt_batch_kernel_mode(&[kernel_job], 128, 128),
        CudaJ2kIdwtBatchKernelMode::Cooperative97
    );
    assert_eq!(
        idwt_batch_kernel_mode(&[kernel_job], 64, 64),
        CudaJ2kIdwtBatchKernelMode::Cooperative97
    );
    assert_eq!(
        idwt_batch_kernel_mode(&[kernel_job], 512, 512),
        CudaJ2kIdwtBatchKernelMode::Cooperative97
    );
    assert_eq!(
        idwt_batch_kernel_mode(&[kernel_job], 63, 64),
        CudaJ2kIdwtBatchKernelMode::Generic
    );
    assert_eq!(
        idwt_batch_kernel_mode(&[kernel_job], 513, 128),
        CudaJ2kIdwtBatchKernelMode::Generic
    );
    kernel_job.job.irreversible97 = 0;
    assert_ne!(
        idwt_batch_kernel_mode(&[kernel_job], 128, 128),
        CudaJ2kIdwtBatchKernelMode::Cooperative97
    );
}

#[test]
fn idwt_batch_trace_row_reports_stage_shape_and_mode() {
    let kernel_jobs = [
        CudaJ2kIdwtMultiKernelJob {
            ll_ptr: 0,
            hl_ptr: 0,
            lh_ptr: 0,
            hh_ptr: 0,
            output_ptr: 0,
            job: CudaJ2kIdwtJob {
                rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 128,
                    y1: 96,
                },
                ll_rect: CudaJ2kRect::default(),
                hl_rect: CudaJ2kRect::default(),
                lh_rect: CudaJ2kRect::default(),
                hh_rect: CudaJ2kRect::default(),
                irreversible97: 1,
            },
            reserved_tail: 0,
        },
        CudaJ2kIdwtMultiKernelJob {
            ll_ptr: 0,
            hl_ptr: 0,
            lh_ptr: 0,
            hh_ptr: 0,
            output_ptr: 0,
            job: CudaJ2kIdwtJob {
                rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 64,
                    y1: 48,
                },
                ll_rect: CudaJ2kRect::default(),
                hl_rect: CudaJ2kRect::default(),
                lh_rect: CudaJ2kRect::default(),
                hh_rect: CudaJ2kRect::default(),
                irreversible97: 1,
            },
            reserved_tail: 0,
        },
    ];

    let row = idwt_batch_trace_row(
        3,
        &kernel_jobs,
        128,
        96,
        CudaJ2kIdwtBatchKernelMode::Cooperative97,
        42,
    );

    assert_eq!(
            format_idwt_batch_trace_row(row),
            "j2k_profile codec=j2k op=cuda_idwt_batch path=decode stage_index=3 mode=Cooperative97 job_count=2 max_width=128 max_height=96 min_width=64 min_height=48 total_pixels=15360 irreversible_jobs=2 elapsed_us=42"
        );
}

#[test]
fn j2k_inverse_dwt_batch_empty_uses_no_dispatch_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let execution = context
        .j2k_inverse_dwt_batch_device_with_pool(&[] as &[CudaJ2kIdwtTarget<'_>], &pool)
        .expect("empty batched CUDA inverse DWT");

    assert_eq!(execution.kernel_dispatches(), 0);
    assert_eq!(execution.decode_kernel_dispatches(), 0);
}

#[test]
fn j2k_inverse_dwt_batch_matches_expected_outputs_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let ll = context
        .upload(super::super::f32_slice_as_bytes(&[10.0]))
        .expect("upload LL");
    let hl = context
        .upload(super::super::f32_slice_as_bytes(&[2.0]))
        .expect("upload HL");
    let lh = context
        .upload(super::super::f32_slice_as_bytes(&[4.0]))
        .expect("upload LH");
    let hh = context
        .upload(super::super::f32_slice_as_bytes(&[1.0]))
        .expect("upload HH");
    let first_output = pool
        .take(4 * std::mem::size_of::<f32>())
        .expect("first batched IDWT output");
    let second_output = pool
        .take(4 * std::mem::size_of::<f32>())
        .expect("second batched IDWT output");
    let job = CudaJ2kIdwtJob {
        rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 2,
            y1: 2,
        },
        ll_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        hl_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        lh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        hh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        irreversible97: 0,
    };

    let execution = context
        .j2k_inverse_dwt_batch_device_with_pool(
            &[
                CudaJ2kIdwtTarget {
                    ll: &ll,
                    hl: &hl,
                    lh: &lh,
                    hh: &hh,
                    output: first_output
                        .as_device_buffer()
                        .expect("first output device buffer"),
                    job,
                },
                CudaJ2kIdwtTarget {
                    ll: &ll,
                    hl: &hl,
                    lh: &lh,
                    hh: &hh,
                    output: second_output
                        .as_device_buffer()
                        .expect("second output device buffer"),
                    job,
                },
            ],
            &pool,
        )
        .expect("batched CUDA inverse DWT");
    assert_eq!(execution.kernel_dispatches(), 2);

    let mut first_actual = vec![0.0f32; 4];
    first_output
        .copy_to_host(super::super::f32_slice_as_bytes_mut(&mut first_actual))
        .expect("download first batched IDWT");
    assert_eq!(first_actual, vec![7.0, 9.0, 10.0, 13.0]);
    let mut second_actual = vec![0.0f32; 4];
    second_output
        .copy_to_host(super::super::f32_slice_as_bytes_mut(&mut second_actual))
        .expect("download second batched IDWT");
    assert_eq!(second_actual, vec![7.0, 9.0, 10.0, 13.0]);
}

#[test]
fn j2k_inverse_dwt_batch_odd_origin_matches_single_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let ll = context
        .upload(super::super::f32_slice_as_bytes(&[10.0]))
        .expect("upload odd LL");
    let hl = context
        .upload(super::super::f32_slice_as_bytes(&[2.0, 5.0]))
        .expect("upload odd HL");
    let lh = context
        .upload(super::super::f32_slice_as_bytes(&[4.0, 7.0]))
        .expect("upload odd LH");
    let hh = context
        .upload(super::super::f32_slice_as_bytes(&[1.0, 3.0, 6.0, 8.0]))
        .expect("upload odd HH");
    let job = CudaJ2kIdwtJob {
        rect: CudaJ2kRect {
            x0: 1,
            y0: 1,
            x1: 4,
            y1: 4,
        },
        ll_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        hl_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 2,
            y1: 1,
        },
        lh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 2,
        },
        hh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 2,
            y1: 2,
        },
        irreversible97: 0,
    };

    let single = context
        .j2k_inverse_dwt_single_device_with_pool(&ll, &hl, &lh, &hh, job, &pool)
        .expect("single CUDA inverse DWT");
    assert_eq!(single.execution().kernel_dispatches(), 3);
    let batch_output = pool
        .take(9 * std::mem::size_of::<f32>())
        .expect("odd batched IDWT output");
    let execution = context
        .j2k_inverse_dwt_batch_device_with_pool(
            &[CudaJ2kIdwtTarget {
                ll: &ll,
                hl: &hl,
                lh: &lh,
                hh: &hh,
                output: batch_output
                    .as_device_buffer()
                    .expect("odd batch output device buffer"),
                job,
            }],
            &pool,
        )
        .expect("odd-origin batched CUDA inverse DWT");
    assert_eq!(execution.kernel_dispatches(), 2);

    let mut single_actual = vec![0.0f32; 9];
    single
        .buffer()
        .expect("single odd output device buffer")
        .copy_to_host(super::super::f32_slice_as_bytes_mut(&mut single_actual))
        .expect("download single odd IDWT");
    let mut batch_actual = vec![0.0f32; 9];
    batch_output
        .copy_to_host(super::super::f32_slice_as_bytes_mut(&mut batch_actual))
        .expect("download batch odd IDWT");
    assert_eq!(batch_actual, single_actual);
}

#[test]
#[expect(
    clippy::cast_precision_loss,
    clippy::similar_names,
    reason = "small fixture coordinates and parallel plane names mirror the CUDA API"
)]
fn j2k_inverse_dwt_batch_large_reversible_matches_single_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let band_len = 64 * 64;
    let ll_values: Vec<f32> = (0..band_len).map(|idx| (idx % 19) as f32).collect();
    let hl_values: Vec<f32> = (0..band_len).map(|idx| ((idx * 3) % 23) as f32).collect();
    let lh_values: Vec<f32> = (0..band_len).map(|idx| ((idx * 5) % 29) as f32).collect();
    let hh_values: Vec<f32> = (0..band_len).map(|idx| ((idx * 7) % 31) as f32).collect();
    let ll = context
        .upload(super::super::f32_slice_as_bytes(&ll_values))
        .expect("upload large LL");
    let hl = context
        .upload(super::super::f32_slice_as_bytes(&hl_values))
        .expect("upload large HL");
    let lh = context
        .upload(super::super::f32_slice_as_bytes(&lh_values))
        .expect("upload large LH");
    let hh = context
        .upload(super::super::f32_slice_as_bytes(&hh_values))
        .expect("upload large HH");
    let job = CudaJ2kIdwtJob {
        rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 128,
            y1: 128,
        },
        ll_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 64,
            y1: 64,
        },
        hl_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 64,
            y1: 64,
        },
        lh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 64,
            y1: 64,
        },
        hh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 64,
            y1: 64,
        },
        irreversible97: 0,
    };

    let single = context
        .j2k_inverse_dwt_single_device_with_pool(&ll, &hl, &lh, &hh, job, &pool)
        .expect("large single CUDA inverse DWT");
    let batch_output = pool
        .take(128 * 128 * std::mem::size_of::<f32>())
        .expect("large batched IDWT output");
    let execution = context
        .j2k_inverse_dwt_batch_device_with_pool(
            &[CudaJ2kIdwtTarget {
                ll: &ll,
                hl: &hl,
                lh: &lh,
                hh: &hh,
                output: batch_output
                    .as_device_buffer()
                    .expect("large batch output device buffer"),
                job,
            }],
            &pool,
        )
        .expect("large batched CUDA inverse DWT");
    assert_eq!(execution.kernel_dispatches(), 2);

    let mut single_actual = vec![0.0f32; 128 * 128];
    single
        .buffer()
        .expect("large single output device buffer")
        .copy_to_host(super::super::f32_slice_as_bytes_mut(&mut single_actual))
        .expect("download large single IDWT");
    let mut batch_actual = vec![0.0f32; 128 * 128];
    batch_output
        .copy_to_host(super::super::f32_slice_as_bytes_mut(&mut batch_actual))
        .expect("download large batch IDWT");
    assert_eq!(batch_actual, single_actual);
}

#[test]
#[expect(
    clippy::cast_precision_loss,
    clippy::similar_names,
    reason = "small fixture coordinates and parallel plane names mirror the CUDA API"
)]
fn j2k_inverse_dwt_batch_large_irreversible_matches_single_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let band_len = 128 * 128;
    let ll_values: Vec<f32> = (0..band_len)
        .map(|idx| ((idx % 43) as f32) * 0.25)
        .collect();
    let hl_values: Vec<f32> = (0..band_len)
        .map(|idx| (((idx * 3) % 47) as f32) * 0.125)
        .collect();
    let lh_values: Vec<f32> = (0..band_len)
        .map(|idx| (((idx * 5) % 53) as f32) * 0.0625)
        .collect();
    let hh_values: Vec<f32> = (0..band_len)
        .map(|idx| (((idx * 7) % 59) as f32) * 0.03125)
        .collect();
    let ll = context
        .upload(super::super::f32_slice_as_bytes(&ll_values))
        .expect("upload large irreversible LL");
    let hl = context
        .upload(super::super::f32_slice_as_bytes(&hl_values))
        .expect("upload large irreversible HL");
    let lh = context
        .upload(super::super::f32_slice_as_bytes(&lh_values))
        .expect("upload large irreversible LH");
    let hh = context
        .upload(super::super::f32_slice_as_bytes(&hh_values))
        .expect("upload large irreversible HH");
    let job = CudaJ2kIdwtJob {
        rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 256,
            y1: 256,
        },
        ll_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 128,
            y1: 128,
        },
        hl_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 128,
            y1: 128,
        },
        lh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 128,
            y1: 128,
        },
        hh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 128,
            y1: 128,
        },
        irreversible97: 1,
    };

    let single = context
        .j2k_inverse_dwt_single_device_with_pool(&ll, &hl, &lh, &hh, job, &pool)
        .expect("large irreversible single CUDA inverse DWT");
    let batch_output = pool
        .take(256 * 256 * std::mem::size_of::<f32>())
        .expect("large irreversible batched IDWT output");
    let execution = context
        .j2k_inverse_dwt_batch_device_with_pool(
            &[CudaJ2kIdwtTarget {
                ll: &ll,
                hl: &hl,
                lh: &lh,
                hh: &hh,
                output: batch_output
                    .as_device_buffer()
                    .expect("large irreversible batch output device buffer"),
                job,
            }],
            &pool,
        )
        .expect("large irreversible batched CUDA inverse DWT");
    assert_eq!(execution.kernel_dispatches(), 2);

    let mut single_actual = vec![0.0f32; 256 * 256];
    single
        .buffer()
        .expect("large irreversible single output device buffer")
        .copy_to_host(super::super::f32_slice_as_bytes_mut(&mut single_actual))
        .expect("download large irreversible single IDWT");
    let mut batch_actual = vec![0.0f32; 256 * 256];
    batch_output
        .copy_to_host(super::super::f32_slice_as_bytes_mut(&mut batch_actual))
        .expect("download large irreversible batch IDWT");
    assert_eq!(batch_actual, single_actual);
}

#[test]
#[expect(
    clippy::cast_precision_loss,
    clippy::similar_names,
    reason = "small fixture coordinates and parallel plane names mirror the CUDA API"
)]
fn j2k_inverse_dwt_batch_512_reversible_matches_single_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let band_len = 256 * 256;
    let ll_values: Vec<f32> = (0..band_len).map(|idx| (idx % 43) as f32).collect();
    let hl_values: Vec<f32> = (0..band_len).map(|idx| ((idx * 3) % 47) as f32).collect();
    let lh_values: Vec<f32> = (0..band_len).map(|idx| ((idx * 5) % 53) as f32).collect();
    let hh_values: Vec<f32> = (0..band_len).map(|idx| ((idx * 7) % 59) as f32).collect();
    let ll = context
        .upload(super::super::f32_slice_as_bytes(&ll_values))
        .expect("upload 512 LL");
    let hl = context
        .upload(super::super::f32_slice_as_bytes(&hl_values))
        .expect("upload 512 HL");
    let lh = context
        .upload(super::super::f32_slice_as_bytes(&lh_values))
        .expect("upload 512 LH");
    let hh = context
        .upload(super::super::f32_slice_as_bytes(&hh_values))
        .expect("upload 512 HH");
    let job = CudaJ2kIdwtJob {
        rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 512,
            y1: 512,
        },
        ll_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 256,
            y1: 256,
        },
        hl_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 256,
            y1: 256,
        },
        lh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 256,
            y1: 256,
        },
        hh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 256,
            y1: 256,
        },
        irreversible97: 0,
    };

    let single = context
        .j2k_inverse_dwt_single_device_with_pool(&ll, &hl, &lh, &hh, job, &pool)
        .expect("512 single CUDA inverse DWT");
    let batch_output = pool
        .take(512 * 512 * std::mem::size_of::<f32>())
        .expect("512 batched IDWT output");
    let execution = context
        .j2k_inverse_dwt_batch_device_with_pool(
            &[CudaJ2kIdwtTarget {
                ll: &ll,
                hl: &hl,
                lh: &lh,
                hh: &hh,
                output: batch_output
                    .as_device_buffer()
                    .expect("512 batch output device buffer"),
                job,
            }],
            &pool,
        )
        .expect("512 batched CUDA inverse DWT");
    assert_eq!(execution.kernel_dispatches(), 2);

    let mut single_actual = vec![0.0f32; 512 * 512];
    single
        .buffer()
        .expect("512 single output device buffer")
        .copy_to_host(super::super::f32_slice_as_bytes_mut(&mut single_actual))
        .expect("download 512 single IDWT");
    let mut batch_actual = vec![0.0f32; 512 * 512];
    batch_output
        .copy_to_host(super::super::f32_slice_as_bytes_mut(&mut batch_actual))
        .expect("download 512 batch IDWT");
    assert_eq!(batch_actual, single_actual);
}

#[test]
fn j2k_inverse_dwt_batch_enqueue_matches_expected_outputs_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let ll = context
        .upload(super::super::f32_slice_as_bytes(&[10.0]))
        .expect("upload LL");
    let hl = context
        .upload(super::super::f32_slice_as_bytes(&[2.0]))
        .expect("upload HL");
    let lh = context
        .upload(super::super::f32_slice_as_bytes(&[4.0]))
        .expect("upload LH");
    let hh = context
        .upload(super::super::f32_slice_as_bytes(&[1.0]))
        .expect("upload HH");
    let output = pool
        .take(4 * std::mem::size_of::<f32>())
        .expect("batched IDWT output");
    let job = CudaJ2kIdwtJob {
        rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 2,
            y1: 2,
        },
        ll_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        hl_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        lh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        hh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        irreversible97: 0,
    };

    // SAFETY: every target buffer remains live and untouched through the
    // queued handle's explicit `finish` completion below.
    let queued = unsafe {
        context.j2k_inverse_dwt_batch_device_enqueue_with_pool(
            &[CudaJ2kIdwtTarget {
                ll: &ll,
                hl: &hl,
                lh: &lh,
                hh: &hh,
                output: output.as_device_buffer().expect("output device buffer"),
                job,
            }],
            &pool,
        )
    }
    .expect("enqueue batched CUDA inverse DWT");
    assert_eq!(queued.execution().kernel_dispatches(), 2);
    let completed = queued.finish().expect("queued IDWT completion");
    assert_eq!(completed.kernel_dispatches(), 2);

    let mut actual = vec![0.0f32; 4];
    output
        .copy_to_host(super::super::f32_slice_as_bytes_mut(&mut actual))
        .expect("download queued batched IDWT");
    assert_eq!(actual, vec![7.0, 9.0, 10.0, 13.0]);
}

#[test]
#[expect(
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "end-to-end reversible pipeline fixture keeps staged buffer assertions together"
)]
fn j2k_inverse_dwt_batch_sequence_enqueue_matches_two_stage_path_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let ll = context
        .upload(super::super::f32_slice_as_bytes(&[10.0]))
        .expect("upload LL");
    let hl = context
        .upload(super::super::f32_slice_as_bytes(&[2.0]))
        .expect("upload HL");
    let lh = context
        .upload(super::super::f32_slice_as_bytes(&[4.0]))
        .expect("upload LH");
    let hh = context
        .upload(super::super::f32_slice_as_bytes(&[1.0]))
        .expect("upload HH");
    let stage2_hl = context
        .upload(super::super::f32_slice_as_bytes(&[0.0, 1.0, 2.0, 3.0]))
        .expect("upload stage2 HL");
    let stage2_lh = context
        .upload(super::super::f32_slice_as_bytes(&[4.0, 5.0, 6.0, 7.0]))
        .expect("upload stage2 LH");
    let stage2_hh = context
        .upload(super::super::f32_slice_as_bytes(&[8.0, 9.0, 10.0, 11.0]))
        .expect("upload stage2 HH");
    let stage1_job = CudaJ2kIdwtJob {
        rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 2,
            y1: 2,
        },
        ll_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        hl_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        lh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        hh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        irreversible97: 0,
    };
    let stage2_job = CudaJ2kIdwtJob {
        rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 4,
            y1: 4,
        },
        ll_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 2,
            y1: 2,
        },
        hl_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 2,
            y1: 2,
        },
        lh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 2,
            y1: 2,
        },
        hh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 2,
            y1: 2,
        },
        irreversible97: 0,
    };
    let legacy_stage1 = pool
        .take(4 * std::mem::size_of::<f32>())
        .expect("legacy stage1 output");
    let legacy_stage2 = pool
        .take(16 * std::mem::size_of::<f32>())
        .expect("legacy stage2 output");
    let sequence_stage1 = pool
        .take(4 * std::mem::size_of::<f32>())
        .expect("sequence stage1 output");
    let sequence_stage2 = pool
        .take(16 * std::mem::size_of::<f32>())
        .expect("sequence stage2 output");

    context
        .j2k_inverse_dwt_batch_device_with_pool(
            &[CudaJ2kIdwtTarget {
                ll: &ll,
                hl: &hl,
                lh: &lh,
                hh: &hh,
                output: legacy_stage1
                    .as_device_buffer()
                    .expect("legacy stage1 device buffer"),
                job: stage1_job,
            }],
            &pool,
        )
        .expect("legacy stage1 IDWT");
    context
        .j2k_inverse_dwt_batch_device_with_pool(
            &[CudaJ2kIdwtTarget {
                ll: legacy_stage1
                    .as_device_buffer()
                    .expect("legacy stage1 device buffer"),
                hl: &stage2_hl,
                lh: &stage2_lh,
                hh: &stage2_hh,
                output: legacy_stage2
                    .as_device_buffer()
                    .expect("legacy stage2 device buffer"),
                job: stage2_job,
            }],
            &pool,
        )
        .expect("legacy stage2 IDWT");

    let sequence_stage1_targets = [CudaJ2kIdwtTarget {
        ll: &ll,
        hl: &hl,
        lh: &lh,
        hh: &hh,
        output: sequence_stage1
            .as_device_buffer()
            .expect("sequence stage1 device buffer"),
        job: stage1_job,
    }];
    let sequence_stage2_targets = [CudaJ2kIdwtTarget {
        ll: sequence_stage1
            .as_device_buffer()
            .expect("sequence stage1 device buffer"),
        hl: &stage2_hl,
        lh: &stage2_lh,
        hh: &stage2_hh,
        output: sequence_stage2
            .as_device_buffer()
            .expect("sequence stage2 device buffer"),
        job: stage2_job,
    }];
    // SAFETY: both stages' buffers remain live and untouched through the
    // queued handle's explicit `finish` completion below.
    let queued = unsafe {
        context.j2k_inverse_dwt_batch_sequence_enqueue_with_pool(
            &[&sequence_stage1_targets, &sequence_stage2_targets],
            &pool,
        )
    }
    .expect("queued IDWT sequence");
    assert_eq!(queued.execution().kernel_dispatches(), 4);
    assert_eq!(queued.resource_count(), 1);
    let completed = queued.finish().expect("queued IDWT sequence completion");
    assert_eq!(completed.kernel_dispatches(), 4);

    let mut legacy_actual = vec![0.0f32; 16];
    legacy_stage2
        .copy_to_host(super::super::f32_slice_as_bytes_mut(&mut legacy_actual))
        .expect("download legacy stage2 IDWT");
    let mut sequence_actual = vec![0.0f32; 16];
    sequence_stage2
        .copy_to_host(super::super::f32_slice_as_bytes_mut(&mut sequence_actual))
        .expect("download sequence stage2 IDWT");
    assert_eq!(sequence_actual, legacy_actual);
}

#[test]
fn j2k_store_rgb8_mct_matches_inverse_mct_plus_store_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let plane0 = [16.0f32, 18.0, 21.0, 24.0];
    let plane1 = [-3.0f32, 4.0, 5.0, -6.0];
    let plane2 = [2.0f32, -1.0, 7.0, 3.0];
    let legacy0 = context
        .upload(super::super::f32_slice_as_bytes(&plane0))
        .expect("upload legacy MCT plane 0");
    let legacy1 = context
        .upload(super::super::f32_slice_as_bytes(&plane1))
        .expect("upload legacy MCT plane 1");
    let legacy2 = context
        .upload(super::super::f32_slice_as_bytes(&plane2))
        .expect("upload legacy MCT plane 2");
    let fused0 = context
        .upload(super::super::f32_slice_as_bytes(&plane0))
        .expect("upload fused MCT plane 0");
    let fused1 = context
        .upload(super::super::f32_slice_as_bytes(&plane1))
        .expect("upload fused MCT plane 1");
    let fused2 = context
        .upload(super::super::f32_slice_as_bytes(&plane2))
        .expect("upload fused MCT plane 2");
    let addend = 128.0;

    let mct_stats = context
        .j2k_inverse_mct_device(
            &legacy0,
            &legacy1,
            &legacy2,
            super::super::CudaJ2kInverseMctJob {
                len: 4,
                irreversible97: 0,
                addend0: addend,
                addend1: addend,
                addend2: addend,
            },
        )
        .expect("legacy inverse MCT");
    assert_eq!(mct_stats.kernel_dispatches(), 1);
    let store_job = super::super::CudaJ2kStoreRgb8Job {
        input_width0: 2,
        input_width1: 2,
        input_width2: 2,
        source_x0: 0,
        source_y0: 0,
        source_x1: 0,
        source_y1: 0,
        source_x2: 0,
        source_y2: 0,
        copy_width: 2,
        copy_height: 2,
        output_width: 2,
        output_height: 2,
        output_x: 0,
        output_y: 0,
        addend0: 0.0,
        addend1: 0.0,
        addend2: 0.0,
        bit_depth0: 8,
        bit_depth1: 8,
        bit_depth2: 8,
        rgba: 1,
    };
    let legacy_output = context
        .j2k_store_rgb8_device(&legacy0, &legacy1, &legacy2, store_job)
        .expect("legacy RGB8 store");
    let fused_output = context
        .j2k_store_rgb8_mct_device(
            &fused0,
            &fused1,
            &fused2,
            super::super::CudaJ2kStoreRgb8MctJob {
                store: super::super::CudaJ2kStoreRgb8Job {
                    addend0: addend,
                    addend1: addend,
                    addend2: addend,
                    ..store_job
                },
                irreversible97: 0,
            },
        )
        .expect("fused RGB8 MCT store");

    assert_eq!(legacy_output.execution().kernel_dispatches(), 1);
    assert_eq!(fused_output.execution().kernel_dispatches(), 1);
    let mut legacy_bytes = vec![0u8; 16];
    legacy_output
        .buffer()
        .copy_to_host(&mut legacy_bytes)
        .expect("download legacy RGB8");
    let mut fused_bytes = vec![0u8; 16];
    fused_output
        .buffer()
        .copy_to_host(&mut fused_bytes)
        .expect("download fused RGB8");
    assert_eq!(fused_bytes, legacy_bytes);
}

#[test]
#[expect(
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "end-to-end irreversible pipeline fixture keeps staged buffer assertions together"
)]
fn j2k_store_rgb8_mct_batch_matches_separate_stores_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let plane0_a = [16.0f32, 18.0, 21.0, 24.0];
    let plane1_a = [-3.0f32, 4.0, 5.0, -6.0];
    let plane2_a = [2.0f32, -1.0, 7.0, 3.0];
    let plane0_b = [3.0f32, 7.0, 11.0, 13.0];
    let plane1_b = [5.0f32, -2.0, 9.0, 1.0];
    let plane2_b = [-4.0f32, 6.0, 0.0, 8.0];

    let plane0_a = context
        .upload(super::super::f32_slice_as_bytes(&plane0_a))
        .expect("upload plane 0 A");
    let plane1_a = context
        .upload(super::super::f32_slice_as_bytes(&plane1_a))
        .expect("upload plane 1 A");
    let plane2_a = context
        .upload(super::super::f32_slice_as_bytes(&plane2_a))
        .expect("upload plane 2 A");
    let plane0_b = context
        .upload(super::super::f32_slice_as_bytes(&plane0_b))
        .expect("upload plane 0 B");
    let plane1_b = context
        .upload(super::super::f32_slice_as_bytes(&plane1_b))
        .expect("upload plane 1 B");
    let plane2_b = context
        .upload(super::super::f32_slice_as_bytes(&plane2_b))
        .expect("upload plane 2 B");

    let store = super::super::CudaJ2kStoreRgb8Job {
        input_width0: 2,
        input_width1: 2,
        input_width2: 2,
        source_x0: 0,
        source_y0: 0,
        source_x1: 0,
        source_y1: 0,
        source_x2: 0,
        source_y2: 0,
        copy_width: 2,
        copy_height: 2,
        output_width: 2,
        output_height: 2,
        output_x: 0,
        output_y: 0,
        addend0: 128.0,
        addend1: 128.0,
        addend2: 128.0,
        bit_depth0: 8,
        bit_depth1: 8,
        bit_depth2: 8,
        rgba: 1,
    };
    let separate_a = context
        .j2k_store_rgb8_mct_device(
            &plane0_a,
            &plane1_a,
            &plane2_a,
            super::super::CudaJ2kStoreRgb8MctJob {
                store,
                irreversible97: 0,
            },
        )
        .expect("separate fused store A");
    let separate_b = context
        .j2k_store_rgb8_mct_device(
            &plane0_b,
            &plane1_b,
            &plane2_b,
            super::super::CudaJ2kStoreRgb8MctJob {
                store,
                irreversible97: 0,
            },
        )
        .expect("separate fused store B");

    let batched = context
        .j2k_store_rgb8_mct_batch_device(&[
            super::super::CudaJ2kStoreRgb8MctTarget {
                plane0: &plane0_a,
                plane1: &plane1_a,
                plane2: &plane2_a,
                job: super::super::CudaJ2kStoreRgb8MctJob {
                    store,
                    irreversible97: 0,
                },
            },
            super::super::CudaJ2kStoreRgb8MctTarget {
                plane0: &plane0_b,
                plane1: &plane1_b,
                plane2: &plane2_b,
                job: super::super::CudaJ2kStoreRgb8MctJob {
                    store,
                    irreversible97: 0,
                },
            },
        ])
        .expect("batched fused store");

    assert_eq!(batched.execution().kernel_dispatches(), 1);
    assert_eq!(batched.outputs().len(), 2);
    let mut separate_a_bytes = vec![0u8; 16];
    separate_a
        .buffer()
        .copy_to_host(&mut separate_a_bytes)
        .expect("download separate A");
    let mut separate_b_bytes = vec![0u8; 16];
    separate_b
        .buffer()
        .copy_to_host(&mut separate_b_bytes)
        .expect("download separate B");
    let mut batch_a_bytes = vec![0u8; 16];
    batched.outputs()[0]
        .copy_to_host(&mut batch_a_bytes)
        .expect("download batch A");
    let mut batch_b_bytes = vec![0u8; 16];
    batched.outputs()[1]
        .copy_to_host(&mut batch_b_bytes)
        .expect("download batch B");
    assert_eq!(batch_a_bytes, separate_a_bytes);
    assert_eq!(batch_b_bytes, separate_b_bytes);
}

#[test]
fn j2k_store_rgb8_mct_single_matches_one_item_batch_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let plane0 = [16.0f32, 18.0, 21.0, 24.0];
    let plane1 = [-3.0f32, 4.0, 5.0, -6.0];
    let plane2 = [2.0f32, -1.0, 7.0, 3.0];
    let single0 = context
        .upload(super::super::f32_slice_as_bytes(&plane0))
        .expect("upload single plane 0");
    let single1 = context
        .upload(super::super::f32_slice_as_bytes(&plane1))
        .expect("upload single plane 1");
    let single2 = context
        .upload(super::super::f32_slice_as_bytes(&plane2))
        .expect("upload single plane 2");
    let batch0 = context
        .upload(super::super::f32_slice_as_bytes(&plane0))
        .expect("upload batch plane 0");
    let batch1 = context
        .upload(super::super::f32_slice_as_bytes(&plane1))
        .expect("upload batch plane 1");
    let batch2 = context
        .upload(super::super::f32_slice_as_bytes(&plane2))
        .expect("upload batch plane 2");

    let store = super::super::CudaJ2kStoreRgb8Job {
        input_width0: 2,
        input_width1: 2,
        input_width2: 2,
        source_x0: 0,
        source_y0: 0,
        source_x1: 0,
        source_y1: 0,
        source_x2: 0,
        source_y2: 0,
        copy_width: 2,
        copy_height: 2,
        output_width: 2,
        output_height: 2,
        output_x: 0,
        output_y: 0,
        addend0: 128.0,
        addend1: 128.0,
        addend2: 128.0,
        bit_depth0: 8,
        bit_depth1: 8,
        bit_depth2: 8,
        rgba: 1,
    };
    let job = super::super::CudaJ2kStoreRgb8MctJob {
        store,
        irreversible97: 0,
    };
    let single = context
        .j2k_store_rgb8_mct_device(&single0, &single1, &single2, job)
        .expect("single RGB8 MCT store");
    let batch = context
        .j2k_store_rgb8_mct_batch_device(&[super::super::CudaJ2kStoreRgb8MctTarget {
            plane0: &batch0,
            plane1: &batch1,
            plane2: &batch2,
            job,
        }])
        .expect("one-item batch RGB8 MCT store");

    assert_eq!(single.execution().kernel_dispatches(), 1);
    assert_eq!(batch.execution().kernel_dispatches(), 1);
    let mut single_bytes = vec![0u8; 16];
    single
        .buffer()
        .copy_to_host(&mut single_bytes)
        .expect("download single RGB8 MCT store");
    let mut batch_bytes = vec![0u8; 16];
    batch.outputs()[0]
        .copy_to_host(&mut batch_bytes)
        .expect("download one-item batch RGB8 MCT store");
    assert_eq!(single_bytes, batch_bytes);
}

#[test]
fn j2k_store_rgb16_mct_matches_inverse_mct_plus_store_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let plane0 = [40.0f32, 44.0, 52.0, 55.0];
    let plane1 = [-3.5f32, 1.25, 2.75, -4.0];
    let plane2 = [5.0f32, -2.0, 1.5, 6.0];
    let legacy0 = context
        .upload(super::super::f32_slice_as_bytes(&plane0))
        .expect("upload legacy ICT plane 0");
    let legacy1 = context
        .upload(super::super::f32_slice_as_bytes(&plane1))
        .expect("upload legacy ICT plane 1");
    let legacy2 = context
        .upload(super::super::f32_slice_as_bytes(&plane2))
        .expect("upload legacy ICT plane 2");
    let fused0 = context
        .upload(super::super::f32_slice_as_bytes(&plane0))
        .expect("upload fused ICT plane 0");
    let fused1 = context
        .upload(super::super::f32_slice_as_bytes(&plane1))
        .expect("upload fused ICT plane 1");
    let fused2 = context
        .upload(super::super::f32_slice_as_bytes(&plane2))
        .expect("upload fused ICT plane 2");
    let addend = 32768.0;

    context
        .j2k_inverse_mct_device(
            &legacy0,
            &legacy1,
            &legacy2,
            super::super::CudaJ2kInverseMctJob {
                len: 4,
                irreversible97: 1,
                addend0: addend,
                addend1: addend,
                addend2: addend,
            },
        )
        .expect("legacy inverse ICT");
    let store_job = super::super::CudaJ2kStoreRgb16Job {
        input_width0: 2,
        input_width1: 2,
        input_width2: 2,
        source_x0: 0,
        source_y0: 0,
        source_x1: 0,
        source_y1: 0,
        source_x2: 0,
        source_y2: 0,
        copy_width: 2,
        copy_height: 2,
        output_width: 2,
        output_height: 2,
        output_x: 0,
        output_y: 0,
        addend0: 0.0,
        addend1: 0.0,
        addend2: 0.0,
        bit_depth0: 16,
        bit_depth1: 16,
        bit_depth2: 16,
        rgba: 0,
    };
    let legacy_output = context
        .j2k_store_rgb16_device(&legacy0, &legacy1, &legacy2, store_job)
        .expect("legacy RGB16 store");
    let fused_output = context
        .j2k_store_rgb16_mct_device(
            &fused0,
            &fused1,
            &fused2,
            super::super::CudaJ2kStoreRgb16MctJob {
                store: super::super::CudaJ2kStoreRgb16Job {
                    addend0: addend,
                    addend1: addend,
                    addend2: addend,
                    ..store_job
                },
                irreversible97: 1,
            },
        )
        .expect("fused RGB16 MCT store");

    assert_eq!(legacy_output.execution().kernel_dispatches(), 1);
    assert_eq!(fused_output.execution().kernel_dispatches(), 1);
    let mut legacy_bytes = vec![0u8; 24];
    legacy_output
        .buffer()
        .copy_to_host(&mut legacy_bytes)
        .expect("download legacy RGB16");
    let mut fused_bytes = vec![0u8; 24];
    fused_output
        .buffer()
        .copy_to_host(&mut fused_bytes)
        .expect("download fused RGB16");
    assert_eq!(fused_bytes, legacy_bytes);
}

#[test]
fn j2k_dequantize_htj2k_codeblocks_multi_uses_one_dispatch_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let first = context
        .upload(super::super::i32_slice_as_bytes(&[0, 0, 0, 0]))
        .expect("upload first coefficients");
    let second = context
        .upload(super::super::i32_slice_as_bytes(&[0, 0]))
        .expect("upload second coefficients");
    let first_jobs = [CudaHtj2kCodeBlockJob {
        payload_offset: 0,
        width: 2,
        height: 2,
        payload_len: 0,
        cleanup_length: 0,
        refinement_length: 0,
        missing_bit_planes: 0,
        num_bitplanes: 1,
        number_of_coding_passes: 1,
        output_stride: 2,
        output_offset: 0,
        dequantization_step: 1.0,
        stripe_causal: false,
    }];
    let second_jobs = [CudaHtj2kCodeBlockJob {
        payload_offset: 0,
        width: 2,
        height: 1,
        payload_len: 0,
        cleanup_length: 0,
        refinement_length: 0,
        missing_bit_planes: 0,
        num_bitplanes: 1,
        number_of_coding_passes: 1,
        output_stride: 2,
        output_offset: 0,
        dequantization_step: 1.0,
        stripe_causal: false,
    }];

    let pool = context.buffer_pool();
    let execution = context
        .j2k_dequantize_htj2k_codeblocks_multi_device_with_pool(
            &[
                CudaHtj2kDequantizeTarget {
                    coefficients: &first,
                    jobs: &first_jobs,
                    output_words: 4,
                },
                CudaHtj2kDequantizeTarget {
                    coefficients: &second,
                    jobs: &second_jobs,
                    output_words: 2,
                },
            ],
            &pool,
        )
        .expect("multi-buffer HTJ2K dequant");
    assert_eq!(execution.kernel_dispatches(), 1);

    let mut first_actual = vec![f32::NAN; 4];
    first
        .copy_to_host(super::super::f32_slice_as_bytes_mut(&mut first_actual))
        .expect("download first coefficients");
    assert_eq!(first_actual, vec![0.0; 4]);
    let mut second_actual = vec![f32::NAN; 2];
    second
        .copy_to_host(super::super::f32_slice_as_bytes_mut(&mut second_actual))
        .expect("download second coefficients");
    assert_eq!(second_actual, vec![0.0; 2]);
}

#[test]
fn queued_cleanup_metadata_dequantizes_without_second_job_upload_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let first = context
        .upload(super::super::i32_slice_as_bytes(&[1, i32::MIN + 2, 0, 3]))
        .expect("upload first coefficients");
    let second = context
        .upload(super::super::i32_slice_as_bytes(&[4, i32::MIN + 5]))
        .expect("upload second coefficients");
    let jobs = [
        CudaHtj2kCleanupMultiKernelJob {
            output_ptr: first.device_ptr(),
            coded_offset: 0,
            width: 2,
            height: 2,
            coded_len: 0,
            cleanup_length: 0,
            refinement_length: 0,
            missing_msbs: 0,
            num_bitplanes: 31,
            number_of_coding_passes: 1,
            output_stride: 2,
            output_offset: 0,
            dequantization_step: 0.5,
            stripe_causal: 0,
            reserved_tail: 0,
        },
        CudaHtj2kCleanupMultiKernelJob {
            output_ptr: second.device_ptr(),
            coded_offset: 0,
            width: 2,
            height: 1,
            coded_len: 0,
            cleanup_length: 0,
            refinement_length: 0,
            missing_msbs: 0,
            num_bitplanes: 31,
            number_of_coding_passes: 1,
            output_stride: 2,
            output_offset: 0,
            dequantization_step: 0.25,
            stripe_causal: 0,
            reserved_tail: 0,
        },
    ];
    let jobs_buffer = pool
        .upload(super::super::htj2k_cleanup_multi_jobs_as_bytes(&jobs))
        .expect("upload cleanup metadata");
    let queued = CudaQueuedHtj2kCleanup {
        context: context.clone(),
        resources: vec![jobs_buffer],
        status_buffer: None,
        status_count: jobs.len(),
        kernel_name: "j2k_htj2k_decode_codeblocks_multi",
        execution: CudaExecutionStats::default(),
        pool_reuse_guard: None,
        finish_host_live_bytes: 0,
    };

    let execution = context
        .j2k_dequantize_queued_htj2k_cleanup_with_pool(&queued)
        .expect("dequant from queued cleanup metadata");
    assert_eq!(execution.kernel_dispatches(), 1);

    let mut first_actual = vec![f32::NAN; 4];
    first
        .copy_to_host(super::super::f32_slice_as_bytes_mut(&mut first_actual))
        .expect("download first coefficients");
    assert_eq!(first_actual, vec![0.5, -1.0, 0.0, 1.5]);
    let mut second_actual = vec![f32::NAN; 2];
    second
        .copy_to_host(super::super::f32_slice_as_bytes_mut(&mut second_actual))
        .expect("download second coefficients");
    assert_eq!(second_actual, vec![1.0, -1.25]);
}

#[test]
fn htj2k_decode_multi_kernel_routes_cleanup_only_jobs() {
    let cleanup_job = CudaHtj2kCleanupMultiKernelJob {
        output_ptr: 0,
        coded_offset: 0,
        width: 64,
        height: 64,
        coded_len: 8,
        cleanup_length: 8,
        refinement_length: 0,
        missing_msbs: 0,
        num_bitplanes: 8,
        number_of_coding_passes: 1,
        output_stride: 64,
        output_offset: 0,
        dequantization_step: 1.0,
        stripe_causal: 0,
        reserved_tail: 0,
    };
    let (_, cleanup_kernel_name) = super::super::htj2k_decode_multi_kernel_for_jobs(&[cleanup_job]);
    assert_eq!(
        cleanup_kernel_name,
        "j2k_htj2k_decode_codeblocks_multi_cleanup_only"
    );

    let mut refinement_job = cleanup_job;
    refinement_job.refinement_length = 4;
    refinement_job.number_of_coding_passes = 2;
    let (_, generic_kernel_name) =
        super::super::htj2k_decode_multi_kernel_for_jobs(&[refinement_job]);
    assert_eq!(generic_kernel_name, "j2k_htj2k_decode_codeblocks_multi");
}

#[test]
fn htj2k_decode_multi_cleanup_dequant_kernel_accepts_cleanup_only_jobs() {
    let cleanup_job = CudaHtj2kCleanupMultiKernelJob {
        output_ptr: 0,
        coded_offset: 0,
        width: 64,
        height: 64,
        coded_len: 8,
        cleanup_length: 8,
        refinement_length: 0,
        missing_msbs: 0,
        num_bitplanes: 8,
        number_of_coding_passes: 1,
        output_stride: 64,
        output_offset: 0,
        dequantization_step: 1.0,
        stripe_causal: 0,
        reserved_tail: 0,
    };
    let (_, cleanup_dequant_kernel_name) =
        super::super::htj2k_decode_multi_cleanup_dequant_kernel_for_jobs(&[cleanup_job])
            .expect("cleanup-only jobs use fused cleanup/dequant kernel");
    assert_eq!(
        cleanup_dequant_kernel_name,
        "j2k_htj2k_decode_codeblocks_multi_cleanup_dequantize"
    );
}

#[test]
fn htj2k_decode_multi_cleanup_dequant_kernel_rejects_refinement_jobs() {
    let mut refinement_job = CudaHtj2kCleanupMultiKernelJob {
        output_ptr: 0,
        coded_offset: 0,
        width: 64,
        height: 64,
        coded_len: 12,
        cleanup_length: 8,
        refinement_length: 4,
        missing_msbs: 0,
        num_bitplanes: 8,
        number_of_coding_passes: 2,
        output_stride: 64,
        output_offset: 0,
        dequantization_step: 1.0,
        stripe_causal: 0,
        reserved_tail: 0,
    };
    assert!(
        super::super::htj2k_decode_multi_cleanup_dequant_kernel_for_jobs(&[refinement_job])
            .is_none()
    );

    refinement_job.refinement_length = 0;
    assert!(
        super::super::htj2k_decode_multi_cleanup_dequant_kernel_for_jobs(&[refinement_job])
            .is_none()
    );
}

#[test]
#[expect(
    clippy::similar_names,
    reason = "source/destination plane variables mirror the transcode stage contract"
)]
fn htj2k_cleanup_multi_empty_targets_use_no_dispatch_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let first_vlc = [0u16; 1024];
    let later_vlc = [0u16; 1024];
    let first_uvlc = [0u16; 320];
    let later_uvlc = [0u16; 256];
    let tables = context
        .upload_htj2k_decode_table_resources(CudaHtj2kDecodeTables {
            vlc_table0: &first_vlc,
            vlc_table1: &later_vlc,
            uvlc_table0: &first_uvlc,
            uvlc_table1: &later_uvlc,
        })
        .expect("decode tables");
    let resources = context
        .upload_htj2k_decode_resources_with_tables(&[], &tables)
        .expect("decode resources");

    let (execution, timings) = context
        .decode_htj2k_codeblocks_cleanup_multi_with_resources_and_pool_timed(
            &resources,
            &[] as &[CudaHtj2kCleanupTarget<'_>],
            &pool,
            false,
        )
        .expect("empty cleanup batch");

    assert_eq!(execution.kernel_dispatches(), 0);
    assert_eq!(execution.decode_kernel_dispatches(), 0);
    assert_eq!(timings.status_d2h_us, 0);
}

#[test]
#[expect(
    clippy::similar_names,
    reason = "source/destination plane variables mirror the transcode stage contract"
)]
fn htj2k_cleanup_multi_enqueue_empty_targets_finish_with_no_dispatch_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let first_vlc = [0u16; 1024];
    let later_vlc = [0u16; 1024];
    let first_uvlc = [0u16; 320];
    let later_uvlc = [0u16; 256];
    let tables = context
        .upload_htj2k_decode_table_resources(CudaHtj2kDecodeTables {
            vlc_table0: &first_vlc,
            vlc_table1: &later_vlc,
            uvlc_table0: &first_uvlc,
            uvlc_table1: &later_uvlc,
        })
        .expect("decode tables");
    let resources = context
        .upload_htj2k_decode_resources_with_tables(&[], &tables)
        .expect("decode resources");

    // SAFETY: the target set is empty, so no borrowed device allocation can
    // outlive the queued cleanup handle.
    let queued = unsafe {
        context.decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool(
            &resources,
            &[] as &[CudaHtj2kCleanupTarget<'_>],
            &pool,
        )
    }
    .expect("empty queued cleanup batch");
    assert_eq!(queued.execution().kernel_dispatches(), 0);
    assert_eq!(queued.execution().decode_kernel_dispatches(), 0);
    assert_eq!(queued.resource_count(), 0);

    let execution = queued.finish().expect("finish empty queued cleanup");
    assert_eq!(execution.kernel_dispatches(), 0);
    assert_eq!(execution.decode_kernel_dispatches(), 0);
}

#[test]
fn j2k_forward_rct_matches_cpu_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let mut plane0 = vec![10.0, 1.0, 0.0, 255.0, 128.0];
    let mut plane1 = vec![20.0, 2.0, 255.0, 0.0, 64.0];
    let mut plane2 = vec![30.0, 3.0, 128.0, 127.0, 32.0];
    let mut expected0 = plane0.clone();
    let mut expected1 = plane1.clone();
    let mut expected2 = plane2.clone();
    for ((r, g), b) in expected0
        .iter_mut()
        .zip(expected1.iter_mut())
        .zip(expected2.iter_mut())
    {
        let r0 = *r;
        let g0 = *g;
        let b0 = *b;
        *r = ((r0 + 2.0_f32 * g0 + b0) * 0.25_f32).floor();
        *g = b0 - g0;
        *b = r0 - g0;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let execution = context
        .j2k_forward_rct(&mut plane0, &mut plane1, &mut plane2)
        .expect("CUDA forward RCT");

    assert_eq!(execution.kernel_dispatches(), 1);
    assert_eq!(plane0, expected0);
    assert_eq!(plane1, expected1);
    assert_eq!(plane2, expected2);
}

#[test]
fn j2k_deinterleave_to_f32_matches_cpu_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let pixels = [0u8, 128, 255, 64, 32, 16];
    let context = CudaContext::system_default().expect("CUDA context");
    let output = context
        .j2k_deinterleave_to_f32(&pixels, 2, 3, 8, false)
        .expect("CUDA deinterleave");

    assert_eq!(output.execution().kernel_dispatches(), 1);
    assert_eq!(
        output.components(),
        &[vec![-128.0, -64.0], vec![0.0, -96.0], vec![127.0, -112.0],]
    );
}

#[test]
fn j2k_deinterleave_then_rct_can_stay_resident_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let pixels = [10u8, 20, 30, 40, 50, 60];
    let context = CudaContext::system_default().expect("CUDA context");
    let mut components = context
        .j2k_deinterleave_to_f32_resident(&pixels, 2, 3, 8, false)
        .expect("resident CUDA deinterleave");

    assert_eq!(components.num_components(), 3);
    assert_eq!(components.num_pixels(), 2);
    assert_eq!(components.execution().kernel_dispatches(), 1);

    let rct_execution = context
        .j2k_forward_rct_resident(&mut components)
        .expect("resident CUDA forward RCT");

    assert_eq!(rct_execution.kernel_dispatches(), 1);
    assert_eq!(
        components
            .download_components()
            .expect("download resident components"),
        vec![vec![-108.0, -78.0], vec![10.0, 10.0], vec![-10.0, -10.0]]
    );
}

#[test]
fn j2k_deinterleave_then_ict_can_stay_resident_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let pixels = [10u8, 20, 30, 40, 50, 60];
    let context = CudaContext::system_default().expect("CUDA context");
    let mut components = context
        .j2k_deinterleave_to_f32_resident(&pixels, 2, 3, 8, false)
        .expect("resident CUDA deinterleave");

    let ict_execution = context
        .j2k_forward_ict_resident(&mut components)
        .expect("resident CUDA forward ICT");

    assert_eq!(ict_execution.kernel_dispatches(), 1);
    let actual = components
        .download_components()
        .expect("download resident components");
    let expected = [[-118.0f32, -88.0], [-108.0, -78.0], [-98.0, -68.0]];
    for idx in 0..2 {
        let r = expected[0][idx];
        let g = expected[1][idx];
        let b = expected[2][idx];
        let expected_y = 0.299 * r + 0.587 * g + 0.114 * b;
        let blue_chroma = -0.16875 * r - 0.33126 * g + 0.5 * b;
        let red_chroma = 0.5 * r - 0.41869 * g - 0.08131 * b;
        assert!((actual[0][idx] - expected_y).abs() < 0.000_1);
        assert!((actual[1][idx] - blue_chroma).abs() < 0.000_1);
        assert!((actual[2][idx] - red_chroma).abs() < 0.000_1);
    }
}

#[test]
fn j2k_resident_deinterleave_can_feed_resident_dwt53_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let pixels = [0u8, 64, 128, 255];
    let context = CudaContext::system_default().expect("CUDA context");
    let components = context
        .j2k_deinterleave_to_f32_resident(&pixels, 4, 1, 8, false)
        .expect("resident CUDA deinterleave");
    let host_component = components
        .download_components()
        .expect("download source component")[0]
        .clone();
    let expected = context
        .j2k_forward_dwt53(&host_component, 2, 2, 1)
        .expect("host-staged CUDA DWT");

    let resident = context
        .j2k_forward_dwt53_resident_component(&components, 0, 2, 2, 1)
        .expect("resident CUDA DWT");

    assert_eq!(resident.levels(), expected.levels());
    assert_eq!(resident.ll_dimensions(), expected.ll_dimensions());
    assert_eq!(resident.execution().copy_kernel_dispatches, 1);
    assert_eq!(
        resident
            .download_transformed()
            .expect("download resident DWT"),
        expected.transformed()
    );
}

#[test]
fn j2k_resident_deinterleave_can_feed_resident_dwt97_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let pixels = [0u8, 64, 128, 255];
    let context = CudaContext::system_default().expect("CUDA context");
    let components = context
        .j2k_deinterleave_to_f32_resident(&pixels, 4, 1, 8, false)
        .expect("resident CUDA deinterleave");
    let host_component = components
        .download_components()
        .expect("download source component")[0]
        .clone();
    let expected = context
        .j2k_forward_dwt97(&host_component, 2, 2, 1)
        .expect("host-staged CUDA DWT");

    let resident = context
        .j2k_forward_dwt97_resident_component(&components, 0, 2, 2, 1)
        .expect("resident CUDA DWT");

    assert_eq!(resident.levels(), expected.levels());
    assert_eq!(resident.ll_dimensions(), expected.ll_dimensions());
    assert_eq!(resident.execution().copy_kernel_dispatches, 1);
    assert_eq!(
        resident
            .download_transformed()
            .expect("download resident DWT"),
        expected.transformed()
    );
}

#[test]
fn j2k_forward_ict_matches_cpu_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let mut plane0 = vec![10.0, 1.0, 0.0, 255.0, 128.0];
    let mut plane1 = vec![20.0, 2.0, 255.0, 0.0, 64.0];
    let mut plane2 = vec![30.0, 3.0, 128.0, 127.0, 32.0];
    let mut expected0 = plane0.clone();
    let mut expected1 = plane1.clone();
    let mut expected2 = plane2.clone();
    for ((r, g), b) in expected0
        .iter_mut()
        .zip(expected1.iter_mut())
        .zip(expected2.iter_mut())
    {
        let r0 = *r;
        let g0 = *g;
        let b0 = *b;
        *r = 0.299 * r0 + 0.587 * g0 + 0.114 * b0;
        *g = -0.16875 * r0 - 0.33126 * g0 + 0.5 * b0;
        *b = 0.5 * r0 - 0.41869 * g0 - 0.08131 * b0;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let execution = context
        .j2k_forward_ict(&mut plane0, &mut plane1, &mut plane2)
        .expect("CUDA forward ICT");

    assert_eq!(execution.kernel_dispatches(), 1);
    for (actual, expected) in plane0.iter().zip(expected0) {
        assert!((*actual - expected).abs() < 0.0001);
    }
    for (actual, expected) in plane1.iter().zip(expected1) {
        assert!((*actual - expected).abs() < 0.0001);
    }
    for (actual, expected) in plane2.iter().zip(expected2) {
        assert!((*actual - expected).abs() < 0.0001);
    }
}

#[test]
fn j2k_forward_dwt53_matches_cpu_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let width = 5usize;
    let height = 3usize;
    let samples: Vec<f32> = (0..width * height)
        .map(|value| {
            let sample = u16::try_from((value * 7 + 3) % 19).expect("sample fits in u16");
            f32::from(sample)
        })
        .collect();
    let expected = cpu_forward_dwt53_buffer(&samples, width, height, 1);

    let context = CudaContext::system_default().expect("CUDA context");
    let output = context
        .j2k_forward_dwt53(
            &samples,
            u32::try_from(width).expect("width fits in u32"),
            u32::try_from(height).expect("height fits in u32"),
            1,
        )
        .expect("CUDA forward 5/3 DWT");

    assert_eq!(output.execution().kernel_dispatches(), 2);
    assert_eq!(output.transformed(), expected.as_slice());
    assert_eq!(output.ll_dimensions(), (3, 2));
}

#[test]
fn j2k_forward_dwt97_matches_cpu_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let width = 5usize;
    let height = 3usize;
    let samples: Vec<f32> = (0..width * height)
        .map(|value| {
            let sample = u16::try_from((value * 11 + 5) % 31).expect("sample fits in u16");
            f32::from(sample) - 12.0
        })
        .collect();
    let expected = cpu_forward_dwt97_buffer(&samples, width, height, 1);

    let context = CudaContext::system_default().expect("CUDA context");
    let output = context
        .j2k_forward_dwt97(
            &samples,
            u32::try_from(width).expect("width fits in u32"),
            u32::try_from(height).expect("height fits in u32"),
            1,
        )
        .expect("CUDA forward 9/7 DWT");

    assert_eq!(output.execution().kernel_dispatches(), 2);
    for (actual, expected) in output.transformed().iter().zip(expected) {
        assert!((*actual - expected).abs() < 0.001);
    }
    assert_eq!(output.ll_dimensions(), (3, 2));
}

#[test]
fn j2k_quantize_subband_matches_cpu_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let samples = [-3.6f32, -2.5, -0.4, 0.0, 0.49, 1.5, 3.2, 9.9];
    let context = CudaContext::system_default().expect("CUDA context");
    let reversible = context
        .j2k_quantize_subband(
            &samples,
            CudaJ2kQuantizeJob {
                step_exponent: 8,
                step_mantissa: 0,
                range_bits: 8,
                reversible: true,
            },
        )
        .expect("CUDA reversible quantize");
    assert_eq!(reversible.execution().kernel_dispatches(), 1);
    assert_eq!(reversible.coefficients(), &[-4, -3, 0, 0, 0, 2, 3, 10]);

    let irreversible = context
        .j2k_quantize_subband(
            &samples,
            CudaJ2kQuantizeJob {
                step_exponent: 9,
                step_mantissa: 0,
                range_bits: 8,
                reversible: false,
            },
        )
        .expect("CUDA irreversible quantize");
    assert_eq!(irreversible.execution().kernel_dispatches(), 1);
    // delta = 2^(range_bits - step_exponent) = 2^(8 - 9) = 0.5, so q = sign*floor(|s|/0.5).
    // Matches native QuantStepSize::delta and JPEG 2000 T.800 Annex E.
    assert_eq!(irreversible.coefficients(), &[-7, -5, 0, 0, 0, 3, 6, 19]);
}

#[test]
fn j2k_quantize_strided_resident_subband_matches_contiguous_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let samples: Vec<f32> = (0u16..12).map(|value| f32::from(value) - 6.0).collect();
    let context = CudaContext::system_default().expect("CUDA context");
    let sample_buffer = context.upload_f32(&samples).expect("resident samples");
    let quantization = CudaJ2kQuantizeJob {
        step_exponent: 8,
        step_mantissa: 0,
        range_bits: 8,
        reversible: true,
    };
    let resident = context
        .j2k_quantize_subband_region_resident(
            &sample_buffer,
            CudaJ2kQuantizeSubbandRegionJob {
                x0: 1,
                y0: 1,
                width: 2,
                height: 2,
                stride: 4,
                quantization,
            },
        )
        .expect("resident strided quantize");
    let contiguous = [samples[5], samples[6], samples[9], samples[10]];
    let expected = context
        .j2k_quantize_subband(&contiguous, quantization)
        .expect("contiguous quantize");

    assert_eq!(resident.coefficient_count(), 4);
    assert_eq!(resident.execution().kernel_dispatches(), 1);
    assert_eq!(
        resident
            .download_coefficients()
            .expect("download resident quantized coefficients"),
        expected.coefficients()
    );
}

fn cpu_forward_dwt53_buffer(samples: &[f32], width: usize, height: usize, levels: u8) -> Vec<f32> {
    let mut buffer = samples.to_vec();
    let mut current_width = width;
    let mut current_height = height;

    for _ in 0..levels {
        if current_width < 2 && current_height < 2 {
            break;
        }
        if current_height >= 2 {
            let low_height = current_height.div_ceil(2);
            let mut col = vec![0.0; current_height];
            for x in 0..current_width {
                for y in 0..current_height {
                    col[y] = buffer[y * width + x];
                }
                forward_lift_53(&mut col);
                for y in 0..low_height {
                    buffer[y * width + x] = col[y * 2];
                }
                for y in 0..current_height / 2 {
                    buffer[(low_height + y) * width + x] = col[y * 2 + 1];
                }
            }
        }
        if current_width >= 2 {
            let mut row = vec![0.0; current_width];
            for y in 0..current_height {
                let row_start = y * width;
                row.copy_from_slice(&buffer[row_start..row_start + current_width]);
                forward_lift_53(&mut row);
                let low_width = current_width.div_ceil(2);
                for x in 0..low_width {
                    buffer[row_start + x] = row[x * 2];
                }
                for x in 0..current_width / 2 {
                    buffer[row_start + low_width + x] = row[x * 2 + 1];
                }
            }
        }
        current_width = current_width.div_ceil(2);
        current_height = current_height.div_ceil(2);
    }

    buffer
}

fn cpu_forward_dwt97_buffer(samples: &[f32], width: usize, height: usize, levels: u8) -> Vec<f32> {
    let mut buffer = samples.to_vec();
    let mut current_width = width;
    let mut current_height = height;

    for _ in 0..levels {
        if current_width < 2 && current_height < 2 {
            break;
        }
        if current_height >= 2 {
            let low_height = current_height.div_ceil(2);
            let mut col = vec![0.0; current_height];
            for x in 0..current_width {
                for y in 0..current_height {
                    col[y] = buffer[y * width + x];
                }
                forward_lift_97(&mut col);
                for y in 0..low_height {
                    buffer[y * width + x] = col[y * 2];
                }
                for y in 0..current_height / 2 {
                    buffer[(low_height + y) * width + x] = col[y * 2 + 1];
                }
            }
        }
        if current_width >= 2 {
            let mut row = vec![0.0; current_width];
            for y in 0..current_height {
                let row_start = y * width;
                row.copy_from_slice(&buffer[row_start..row_start + current_width]);
                forward_lift_97(&mut row);
                let low_width = current_width.div_ceil(2);
                for x in 0..low_width {
                    buffer[row_start + x] = row[x * 2];
                }
                for x in 0..current_width / 2 {
                    buffer[row_start + low_width + x] = row[x * 2 + 1];
                }
            }
        }
        current_width = current_width.div_ceil(2);
        current_height = current_height.div_ceil(2);
    }

    buffer
}

fn forward_lift_53(data: &mut [f32]) {
    let n = data.len();
    if n < 2 {
        return;
    }

    let last_even = if n.is_multiple_of(2) { n - 2 } else { n - 1 };
    for i in (1..n).step_by(2) {
        let left = data[i - 1];
        let right = if i + 1 < n {
            data[i + 1]
        } else {
            data[last_even]
        };
        data[i] -= ((left + right) * 0.5).floor();
    }

    for i in (0..n).step_by(2) {
        let left = if i > 0 { data[i - 1] } else { data[1] };
        let right = if i + 1 < n { data[i + 1] } else { left };
        data[i] += ((left + right) * 0.25 + 0.5).floor();
    }
}

fn forward_lift_97(data: &mut [f32]) {
    const ALPHA: f32 = -1.586_134_3;
    const BETA: f32 = -0.052_980_117;
    const GAMMA: f32 = 0.882_911_1;
    const DELTA: f32 = 0.443_506_87;
    const KAPPA: f32 = 1.230_174_1;
    const INV_KAPPA: f32 = 1.0 / KAPPA;

    let n = data.len();
    if n < 2 {
        return;
    }

    let last_even = if n.is_multiple_of(2) { n - 2 } else { n - 1 };
    for i in (1..n).step_by(2) {
        let left = data[i - 1];
        let right = if i + 1 < n {
            data[i + 1]
        } else {
            data[last_even]
        };
        data[i] += ALPHA * (left + right);
    }
    for i in (0..n).step_by(2) {
        let left = if i > 0 { data[i - 1] } else { data[1] };
        let right = if i + 1 < n { data[i + 1] } else { left };
        data[i] += BETA * (left + right);
    }
    for i in (1..n).step_by(2) {
        let left = data[i - 1];
        let right = if i + 1 < n {
            data[i + 1]
        } else {
            data[last_even]
        };
        data[i] += GAMMA * (left + right);
    }
    for i in (0..n).step_by(2) {
        let left = if i > 0 { data[i - 1] } else { data[1] };
        let right = if i + 1 < n { data[i + 1] } else { left };
        data[i] += DELTA * (left + right);
    }
    for i in (0..n).step_by(2) {
        data[i] *= INV_KAPPA;
    }
    for i in (1..n).step_by(2) {
        data[i] *= KAPPA;
    }
}
