// SPDX-License-Identifier: MIT OR Apache-2.0

use super::classification::{CorpusInputClass, DecodeMode};
use super::input_loader::load_inputs;

#[derive(Clone)]
pub(crate) struct BenchInput {
    pub(crate) name: String,
    pub(crate) bytes: Vec<u8>,
    pub(crate) dimensions: (u32, u32),
    pub(crate) mode: DecodeMode,
    pub(crate) input_class: CorpusInputClass,
}

pub(crate) fn load_bench_inputs() -> Vec<BenchInput> {
    load_inputs(|name, bytes, dimensions, mode, input_class| BenchInput {
        name,
        bytes,
        dimensions,
        mode,
        input_class,
    })
}
