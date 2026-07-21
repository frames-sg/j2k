// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{DecodeRequest, EncodedImage};

use super::{fixture::encode_ht_fixture, input_selection::InputMode, GENERATED_BATCH_SIZE};

#[derive(Clone, Copy)]
pub(crate) struct WorkloadSpec {
    pub(crate) name: &'static str,
    pub(crate) dimensions: (u32, u32),
    components: u16,
    precision: u8,
    signed: bool,
}

impl WorkloadSpec {
    pub(crate) const fn new(
        name: &'static str,
        width: u32,
        height: u32,
        components: u16,
        precision: u8,
        signed: bool,
    ) -> Self {
        Self {
            name,
            dimensions: (width, height),
            components,
            precision,
            signed,
        }
    }
}

pub(crate) struct Workload {
    pub(crate) name: &'static str,
    pub(crate) dimensions: (u32, u32),
    pub(crate) input_mode: InputMode,
    encoded: Vec<Arc<[u8]>>,
}

impl Workload {
    pub(crate) fn inputs(&self, request: DecodeRequest, count: usize) -> Vec<EncodedImage> {
        assert!(
            count <= GENERATED_BATCH_SIZE,
            "benchmark batch size exceeds generated input set"
        );
        match self.input_mode {
            InputMode::Distinct => self
                .encoded
                .iter()
                .take(count)
                .map(|bytes| EncodedImage::new(Arc::clone(bytes), request))
                .collect(),
            InputMode::Repeated => (0..count)
                .map(|_| EncodedImage::new(Arc::clone(&self.encoded[0]), request))
                .collect(),
        }
    }
}

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

impl From<(&'static str, u32, u32, u16, u8, bool)> for WorkloadSpec {
    fn from(
        (name, width, height, components, precision, signed): (
            &'static str,
            u32,
            u32,
            u16,
            u8,
            bool,
        ),
    ) -> Self {
        Self::new(name, width, height, components, precision, signed)
    }
}

pub(crate) fn materialize_workload(spec: WorkloadSpec, input_mode: InputMode) -> Workload {
    let encoded_count = match input_mode {
        InputMode::Distinct => GENERATED_BATCH_SIZE,
        InputMode::Repeated => 1,
    };
    let encoded = (0..encoded_count)
        .map(|variant| {
            Arc::from(encode_ht_fixture(
                spec.dimensions.0,
                spec.dimensions.1,
                spec.components,
                spec.precision,
                spec.signed,
                variant,
            ))
        })
        .collect();
    Workload {
        name: spec.name,
        dimensions: spec.dimensions,
        input_mode,
        encoded,
    }
}
