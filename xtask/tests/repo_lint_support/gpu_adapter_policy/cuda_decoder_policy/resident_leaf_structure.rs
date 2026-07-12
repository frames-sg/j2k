// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structural ownership and dependency direction for shared CUDA resident leaves.

use std::fs;

use super::super::{assert_pattern_checks, repo_root, PatternCheck};

#[test]
fn resident_decode_shared_primitives_keep_downward_dependencies() {
    let root = repo_root();
    let read = |relative: &str| {
        fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"))
    };
    let allocation = read("crates/j2k-cuda/src/allocation.rs");
    let buffer = read("crates/j2k-cuda/src/decoder/resident/buffer_access.rs");
    let cleanup = read("crates/j2k-cuda/src/decoder/resident/cleanup_dequant.rs");
    let component = read("crates/j2k-cuda/src/decoder/resident/component.rs");
    let idwt = read("crates/j2k-cuda/src/decoder/resident/idwt.rs");
    let surface = read("crates/j2k-cuda/src/decoder/resident/surface.rs");
    let color_store = read("crates/j2k-cuda/src/decoder/color_batch/store.rs");

    assert_pattern_checks(&[
        PatternCheck::new("CUDA resident pooled-buffer access leaf", &buffer)
            .required(&["fn pooled_cuda_buffer(", "as_device_buffer()"])
            .forbidden(&[
                "decode_cuda_component_plan",
                "run_cuda_component_idwt_steps",
                "Surface {",
            ]),
        PatternCheck::new("CUDA allocation dimension primitive", &allocation)
            .required(&["fn checked_cuda_element_count(", "checked_mul"]),
        PatternCheck::new("CUDA resident component owner", &component)
            .required(&["checked_cuda_element_count("])
            .forbidden(&["fn checked_component_area(", "super::surface"]),
        PatternCheck::new("CUDA resident cleanup dependency direction", &cleanup)
            .required(&["super::buffer_access::pooled_cuda_buffer"])
            .forbidden(&["super::surface"]),
        PatternCheck::new("CUDA resident IDWT dependency direction", &idwt)
            .required(&["super::buffer_access::pooled_cuda_buffer"])
            .forbidden(&["super::surface"]),
        PatternCheck::new("CUDA resident surface owner", &surface)
            .required(&["super::buffer_access::pooled_cuda_buffer"])
            .forbidden(&["fn pooled_cuda_buffer("]),
        PatternCheck::new("CUDA color-store area consumer", &color_store)
            .required(&["checked_cuda_element_count("])
            .forbidden(&["fn checked_color_store_area("]),
    ]);

    let resident_sources = [&buffer, &cleanup, &component, &idwt, &surface];
    let pooled_buffer_implementations = resident_sources
        .into_iter()
        .map(|source| source.matches("fn pooled_cuda_buffer(").count())
        .sum::<usize>();
    assert_eq!(
        pooled_buffer_implementations, 1,
        "CUDA resident decode must keep one pooled-buffer access implementation"
    );
    let checked_area_implementations = [&allocation, &component, &color_store]
        .into_iter()
        .map(|source| source.matches("fn checked_cuda_element_count(").count())
        .sum::<usize>();
    assert_eq!(
        checked_area_implementations, 1,
        "CUDA decode must keep one checked two-dimensional element-count implementation"
    );
}
