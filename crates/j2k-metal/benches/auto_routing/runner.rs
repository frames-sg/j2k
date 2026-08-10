// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{path::PathBuf, time::Duration};

use criterion::Criterion;
use j2k_metal::MetalBackendSession;
use j2k_test_support::{
    load_auto_routing_manifest, load_auto_routing_pnm, write_auto_routing_evidence,
    AutoRoutingBackend, AutoRoutingEvidence, AutoRoutingOperation, AutoRoutingPlatform,
    AutoRoutingWorkloadKind,
};

use crate::{decode, encode};

const SAMPLE_SIZE: usize = 10;
const WARM_UP: Duration = Duration::from_secs(1);
const MEASUREMENT: Duration = Duration::from_secs(3);

pub(crate) fn run() {
    let manifest_path = required_path("J2K_AUTO_ROUTING_MANIFEST");
    let corpus_root = required_path("J2K_AUTO_ROUTING_ROOT");
    let evidence_path = required_path("J2K_AUTO_ROUTING_EVIDENCE");
    let workloads = load_auto_routing_manifest(&manifest_path, &corpus_root)
        .unwrap_or_else(|error| panic!("load Metal Auto-routing workloads: {error}"));
    let session = MetalBackendSession::system_default()
        .unwrap_or_else(|error| panic!("Metal Auto-routing benchmark needs a device: {error}"));
    let mut criterion = Criterion::default()
        .sample_size(SAMPLE_SIZE)
        .warm_up_time(WARM_UP)
        .measurement_time(MEASUREMENT)
        .configure_from_args();
    let mut cells = Vec::new();

    for workload in &workloads.workloads {
        match workload.kind {
            AutoRoutingWorkloadKind::Decode => {
                let decode = decode::DecodeCase::new(workload);
                for operation in [
                    AutoRoutingOperation::FullDecode,
                    AutoRoutingOperation::RoiDecode,
                    AutoRoutingOperation::ScaledDecode,
                    AutoRoutingOperation::BatchDecode,
                ] {
                    cells.push(decode::bench_cell(
                        &mut criterion,
                        &decode,
                        operation,
                        &session,
                    ));
                }
            }
            AutoRoutingWorkloadKind::Encode => {
                let encode = load_auto_routing_pnm(workload).unwrap_or_else(|error| {
                    panic!("load Metal encode workload {}: {error}", workload.id)
                });
                for operation in [
                    AutoRoutingOperation::LosslessEncode,
                    AutoRoutingOperation::LossyEncode,
                ] {
                    cells.push(encode::bench_cell(&mut criterion, &encode, operation));
                    if operation == AutoRoutingOperation::LosslessEncode
                        && encode.codec.is_high_throughput()
                    {
                        cells.push(encode::bench_batch_cell(&mut criterion, &encode));
                    }
                }
            }
        }
    }
    criterion.final_summary();

    let evidence = AutoRoutingEvidence {
        schema_version: workloads.manifest.schema_version,
        candidate_sha: required_env("J2K_AUTO_ROUTING_CANDIDATE_SHA"),
        backend: AutoRoutingBackend::Metal,
        platform: AutoRoutingPlatform {
            os: "macos".to_string(),
            arch: "aarch64".to_string(),
            hardware: required_env("J2K_AUTO_ROUTING_HARDWARE"),
            driver: required_env("J2K_AUTO_ROUTING_DRIVER"),
        },
        external_manifest_sha256: workloads.manifest_sha256,
        external_case_count: workloads.workloads.len(),
        cells,
    };
    write_auto_routing_evidence(&evidence_path, &evidence)
        .unwrap_or_else(|error| panic!("write Metal Auto-routing evidence: {error}"));
}

fn required_path(name: &str) -> PathBuf {
    PathBuf::from(required_env(name))
}

fn required_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set for Auto-routing benchmarks"))
}
