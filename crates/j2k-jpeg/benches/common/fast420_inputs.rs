// SPDX-License-Identifier: MIT OR Apache-2.0

use super::input_loader::load_inputs;

pub(crate) struct BenchInput {
    pub(crate) name: String,
    pub(crate) bytes: Vec<u8>,
}

pub(crate) fn load_bench_inputs() -> Vec<BenchInput> {
    load_inputs(|name, bytes, _, _, _| BenchInput { name, bytes })
}
