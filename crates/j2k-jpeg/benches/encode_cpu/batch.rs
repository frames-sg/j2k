// SPDX-License-Identifier: MIT OR Apache-2.0

//! Opt-in matched-worker CPU encoder batch comparison.

use std::hint::black_box;

use criterion::{measurement::WallTime, BenchmarkGroup, Criterion};
use j2k_jpeg::{encode_jpeg_baseline, JpegBackend};

use super::turbo::TurboEncoder;
use super::{CaseKind, EncodeCase};

#[derive(Clone, Copy)]
struct BatchJob {
    case_index: usize,
}

struct NativeWorker;

struct TurboWorker {
    encoders: Vec<TurboEncoder>,
    job_encoder_indices: Vec<usize>,
}

// SAFETY: every TurboEncoder owns an independent compressor. The common
// scoped runner lends each worker exclusively to one thread, joins that thread
// before returning, and never shares a compressor between workers.
unsafe impl Send for TurboWorker {}

pub(super) fn bench(c: &mut Criterion, cases: &[EncodeCase]) {
    let eligible = cases
        .iter()
        .enumerate()
        .filter_map(|(index, case)| case.options.restart_interval.is_none().then_some(index))
        .collect::<Vec<_>>();
    assert!(
        !eligible.is_empty(),
        "matched encoder batch has eligible cases"
    );

    let mut group = c.benchmark_group("jpeg_cpu_encode_matched_batch");
    if std::env::var_os("J2K_ENCODE_BENCH_BATCH_MIXED").is_some() {
        for batch_size in [16usize, 64] {
            let jobs = eligible
                .iter()
                .copied()
                .cycle()
                .take(batch_size)
                .map(|case_index| BatchJob { case_index })
                .collect::<Vec<_>>();
            bench_jobs(
                &mut group,
                cases,
                &format!("mixed_{}_eligible_cases", eligible.len()),
                batch_size,
                &jobs,
            );
        }
    } else {
        for &case_index in &eligible {
            for batch_size in [16usize, 64] {
                let jobs = vec![BatchJob { case_index }; batch_size];
                bench_jobs(
                    &mut group,
                    cases,
                    &cases[case_index].name,
                    batch_size,
                    &jobs,
                );
            }
        }
    }
    group.finish();
}

fn bench_jobs(
    group: &mut BenchmarkGroup<'_, WallTime>,
    cases: &[EncodeCase],
    name: &str,
    batch_size: usize,
    jobs: &[BatchJob],
) {
    assert_eq!(jobs.len(), batch_size);
    for case_index in unique_case_indices(jobs) {
        verify_native_setup(&cases[case_index]);
    }

    for worker_count in [1usize, 2, 4] {
        let mut native_workers = (0..worker_count).map(|_| NativeWorker).collect::<Vec<_>>();
        let native_run =
            |_: &mut NativeWorker, chunk: &[BatchJob]| encode_native_chunk(cases, chunk);
        assert!(run_fixed_scoped_workers(&mut native_workers, jobs, &native_run,) != 0);
        group.bench_function(
            format!("j2k_workers_{worker_count}/{batch_size}/{name}"),
            |b| {
                b.iter(|| {
                    black_box(run_fixed_scoped_workers(
                        &mut native_workers,
                        jobs,
                        &native_run,
                    ))
                });
            },
        );

        let chunk_size = jobs.len().div_ceil(worker_count);
        let mut turbo_workers = jobs
            .chunks(chunk_size)
            .map(|chunk| TurboWorker::new(cases, chunk))
            .collect::<Vec<_>>();
        assert_eq!(turbo_workers.len(), worker_count);
        let turbo_run =
            |worker: &mut TurboWorker, chunk: &[BatchJob]| worker.encode_chunk(cases, chunk);
        assert!(run_fixed_scoped_workers(&mut turbo_workers, jobs, &turbo_run,) != 0);
        group.bench_function(
            format!("libjpeg-turbo_workers_{worker_count}/{batch_size}/{name}"),
            |b| {
                b.iter(|| {
                    black_box(run_fixed_scoped_workers(
                        &mut turbo_workers,
                        jobs,
                        &turbo_run,
                    ))
                });
            },
        );
    }
}

fn run_fixed_scoped_workers<W, F>(workers: &mut [W], jobs: &[BatchJob], run: &F) -> usize
where
    W: Send,
    F: Fn(&mut W, &[BatchJob]) -> usize + Sync,
{
    assert!(!workers.is_empty() && jobs.len() >= workers.len());
    if workers.len() == 1 {
        return run(&mut workers[0], jobs);
    }

    let worker_count = workers.len();
    let chunk_size = jobs.len().div_ceil(worker_count);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for (worker, chunk) in workers.iter_mut().zip(jobs.chunks(chunk_size)) {
            let run_one = run;
            handles.push(
                std::thread::Builder::new()
                    .spawn_scoped(scope, move || run_one(worker, chunk))
                    .expect("matched encoder batch worker"),
            );
        }
        assert_eq!(handles.len(), worker_count);
        handles
            .into_iter()
            .map(|handle| handle.join().expect("matched encoder batch worker"))
            .try_fold(0usize, usize::checked_add)
            .expect("matched encoder batch output bytes")
    })
}

fn encode_native_chunk(cases: &[EncodeCase], jobs: &[BatchJob]) -> usize {
    jobs.iter()
        .map(|job| cases[job.case_index].encode())
        .try_fold(0usize, usize::checked_add)
        .expect("native batch output bytes")
}

impl TurboWorker {
    fn new(cases: &[EncodeCase], jobs: &[BatchJob]) -> Self {
        let case_indices = unique_case_indices(jobs);
        let job_encoder_indices = jobs
            .iter()
            .map(|job| {
                case_indices
                    .binary_search(&job.case_index)
                    .expect("assigned case has a worker encoder")
            })
            .collect();
        let encoders = case_indices
            .into_iter()
            .map(|case_index| {
                let case = &cases[case_index];
                TurboEncoder::new(
                    case.width,
                    case.height,
                    matches!(case.kind, CaseKind::Gray),
                    case.options.subsampling,
                    case.options.quality,
                )
            })
            .collect();
        Self {
            encoders,
            job_encoder_indices,
        }
    }

    fn encode_chunk(&mut self, cases: &[EncodeCase], jobs: &[BatchJob]) -> usize {
        debug_assert_eq!(jobs.len(), self.job_encoder_indices.len());
        jobs.iter()
            .zip(&self.job_encoder_indices)
            .map(|(job, &encoder_index)| {
                self.encoders[encoder_index].encode(&cases[job.case_index].data)
            })
            .try_fold(0usize, usize::checked_add)
            .expect("TurboJPEG batch output bytes")
    }
}

fn unique_case_indices(jobs: &[BatchJob]) -> Vec<usize> {
    let mut indices = jobs.iter().map(|job| job.case_index).collect::<Vec<_>>();
    indices.sort_unstable();
    indices.dedup();
    indices
}

fn verify_native_setup(case: &EncodeCase) {
    let channels = match case.kind {
        CaseKind::Gray => 1usize,
        CaseKind::Rgb => 3,
    };
    let expected_samples = usize::try_from(case.width)
        .expect("benchmark width fits usize")
        .checked_mul(usize::try_from(case.height).expect("benchmark height fits usize"))
        .and_then(|pixels| pixels.checked_mul(channels))
        .expect("benchmark sample length");
    assert_eq!(
        case.data.len(),
        expected_samples,
        "benchmark input geometry"
    );

    let encoded = encode_jpeg_baseline(case.samples(), case.options)
        .expect("native matched batch setup encode");
    assert_eq!(encoded.backend, JpegBackend::Cpu);
    assert!(encoded.data.len() >= 4);
    assert_eq!(&encoded.data[..2], &[0xff, 0xd8]);
    assert_eq!(&encoded.data[encoded.data.len() - 2..], &[0xff, 0xd9]);
    black_box(encoded.data.len());
}
