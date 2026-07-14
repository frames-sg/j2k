// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::{
    assert_file_pattern_checks, assert_pattern_checks, repo_root, FilePatternCheck, PatternCheck,
};

#[test]
fn j2k_ml_stays_independent_experimental_and_explicitly_feature_gated() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("crates/j2k-ml/Cargo.toml")
                .named("j2k-ml manifest")
                .required(&[
                    "name = \"j2k-ml\"",
                    "publish = false",
                    "default = []",
                    "cpu = []",
                    "cuda = [",
                    "metal = [",
                ]),
            FilePatternCheck::new("crates/j2k-ml/README.md")
                .named("j2k-ml independence notice")
                .required(&[
                    "independent integration",
                    "not an official Tracel or Burn crate",
                ]),
        ],
    );
}

#[test]
fn j2k_ml_accelerator_transfer_contracts_are_source_enforced() {
    let root = repo_root();
    let cuda =
        fs::read_to_string(root.join("crates/j2k-ml/src/cuda.rs")).expect("read j2k-ml CUDA route");
    let metal = fs::read_to_string(root.join("crates/j2k-ml/src/metal.rs"))
        .expect("read j2k-ml Metal route");
    let readback = fs::read_to_string(root.join("crates/j2k-metal/src/surface/readback.rs"))
        .expect("read packed Metal surface readback");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-ml CUDA direct allocation bridge", &cuda)
            .required(&[
                "empty_device_contiguous_dtype",
                ".get_resource(cube.handle.clone())",
                "CudaExternalDeviceBufferViewMut::from_raw_parts(",
                ".j2k_ml_convert_into_external(",
                "register_tensor_handle(handle)",
            ])
            .forbidden(&[
                "TensorData",
                "copy_to_host(",
                "copy_range_to_host(",
                "Tensor::from_data(",
            ]),
        PatternCheck::new("j2k-ml Metal compact staged bridge", &metal)
            .required(&[
                "download_surfaces_packed(session, &surface_refs)",
                "integer_tensor_4_from_bytes::<Wgpu>(",
                ".cast(FloatDType::F32)",
                "normalize_4(",
            ])
            .forbidden(&["TensorData", "Vec<f32>", "Tensor::from_data("]),
        PatternCheck::new("packed Metal batch readback", &readback).required(&[
            "checked_shared_buffer(session.device(), total)",
            "checked_blit_command_encoder(&command)",
            "commit_and_wait(&command)",
            "checked_buffer_read_vec::<u8>(&staging, 0, total)",
        ]),
    ]);

    assert_eq!(
        readback
            .matches("checked_shared_buffer(session.device(), total)")
            .count(),
        1,
        "a Metal tensor batch must allocate exactly one packed staging buffer"
    );
    assert_eq!(
        readback.matches("commit_and_wait(&command)").count(),
        1,
        "a Metal tensor batch must submit and complete exactly one packed readback command"
    );
    assert_eq!(
        readback
            .matches("checked_buffer_read_vec::<u8>(&staging, 0, total)")
            .count(),
        1,
        "a Metal tensor batch must perform exactly one packed host read"
    );
}
