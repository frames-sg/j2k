// SPDX-License-Identifier: MIT OR Apache-2.0

use super::workload::WorkloadSpec;

pub(crate) const LOW_BATCH_SIZES: &[usize] = &[1, 8];
pub(crate) const BATCH_SIZES: &[usize] = &[LOW_BATCH_SIZES[0], LOW_BATCH_SIZES[1], 32, 64];

pub(crate) fn workload_specs() -> Vec<WorkloadSpec> {
    let mut workloads = accelerator_workload_specs();
    workloads.extend(color_workload_specs());
    workloads
}

fn accelerator_workload_specs() -> Vec<WorkloadSpec> {
    [
        ("gray12_512", 512, 512, 1, 12, false),
        ("gray12_1024", 1024, 1024, 1, 12, false),
        ("gray16_512", 512, 512, 1, 16, false),
        ("gray16_1024", 1024, 1024, 1, 16, false),
        ("gray_i12_512", 512, 512, 1, 12, true),
        ("gray_i12_1024", 1024, 1024, 1, 12, true),
        ("gray_i16_512", 512, 512, 1, 16, true),
        ("gray_i16_1024", 1024, 1024, 1, 16, true),
    ]
    .into_iter()
    .map(WorkloadSpec::from)
    .collect()
}

fn color_workload_specs() -> impl Iterator<Item = WorkloadSpec> {
    [
        ("rgb8_256", 256, 256, 3, 8, false),
        ("rgb8_512", 512, 512, 3, 8, false),
        ("rgb16_256", 256, 256, 3, 16, false),
        ("rgb16_512", 512, 512, 3, 16, false),
        ("rgba8_256", 256, 256, 4, 8, false),
        ("rgba8_512", 512, 512, 4, 8, false),
        ("rgba16_256", 256, 256, 4, 16, false),
        ("rgba16_512", 512, 512, 4, 16, false),
    ]
    .into_iter()
    .map(WorkloadSpec::from)
}
