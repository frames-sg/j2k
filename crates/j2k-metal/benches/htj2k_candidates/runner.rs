// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Duration;

use criterion::{Criterion, Throughput};
use j2k_metal::MetalEncodeStageAccelerator;

use super::case::{encode_cpu, encode_metal, preflight, workloads};

pub(crate) fn run() {
    let mut criterion = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .configure_from_args();

    for workload in workloads() {
        // Hold transform and packetization work on CPU so the measured routes
        // differ only at the HT code-block/candidate-set boundary.
        let mut preflight_accelerator = MetalEncodeStageAccelerator::for_ht_code_block_encode();
        let checked = preflight(&workload, &mut preflight_accelerator);
        eprintln!(
            "HTJ2K_CANDIDATE_PREFLIGHT case={} tile={}x{} codestream_bytes={} candidate_set_dispatches={} ht_dispatches={} psnr_db={}",
            workload.id,
            workload.side,
            workload.side,
            checked.codestream_bytes,
            checked.candidate_set_dispatches,
            checked.ht_dispatches,
            checked
                .psnr_db
                .map_or_else(|| "lossless".to_string(), |value| format!("{value:.3}")),
        );

        let mut group = criterion.benchmark_group(format!("htj2k-candidates/{}", workload.id));
        group.throughput(Throughput::Elements(1));
        group.bench_function("cpu", |bencher| {
            bencher.iter(|| {
                std::hint::black_box(
                    encode_cpu(&workload).expect("measured CPU HTJ2K candidate workload"),
                )
            });
        });

        let mut accelerator = MetalEncodeStageAccelerator::for_ht_code_block_encode();
        group.bench_function("metal", |bencher| {
            bencher.iter(|| {
                std::hint::black_box(
                    encode_metal(&workload, &mut accelerator)
                        .expect("measured Metal HTJ2K candidate workload"),
                )
            });
        });
        group.finish();
    }
    criterion.final_summary();
}
