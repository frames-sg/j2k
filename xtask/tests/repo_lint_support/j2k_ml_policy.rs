// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, process::Command};

use super::{
    assert_file_pattern_checks, assert_pattern_checks, repo_root, FilePatternCheck, PatternCheck,
};

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative)).unwrap_or_else(|error| {
        panic!("read {relative}: {error}");
    })
}

fn assert_below(path: &str, source: &str, limit: usize) {
    let lines = source.lines().count();
    assert!(
        lines < limit,
        "{path} has {lines} lines; expected fewer than {limit}"
    );
}

#[test]
fn j2k_ml_cpu_and_cuda_owners_stay_split_and_focused() {
    let cpu_path = "crates/j2k-ml/src/cpu.rs";
    let materialization_path = "crates/j2k-ml/src/cpu/materialization.rs";
    let cuda_path = "crates/j2k-ml/src/cuda.rs";
    let interop_path = "crates/j2k-ml/src/cuda/interop.rs";
    let config_path = "crates/j2k-ml/src/cuda/config.rs";
    let cpu = read(cpu_path);
    let materialization = read(materialization_path);
    let cuda = read(cuda_path);
    let interop = read(interop_path);
    let config = read(config_path);

    assert_below(cpu_path, &cpu, 450);
    assert_below(materialization_path, &materialization, 300);
    assert_below(cuda_path, &cuda, 450);
    assert_below(interop_path, &interop, 240);
    assert_below(config_path, &config, 120);
    assert_pattern_checks(&[
        PatternCheck::new("CPU decode orchestration", &cpu)
            .required(&[
                "mod materialization;",
                "fn decode_packed",
                "fn decode_batch",
            ])
            .forbidden(&[
                "TensorData",
                "fn normalize_3",
                "fn integer_tensor_4_from_bytes",
            ]),
        PatternCheck::new("CPU tensor materialization", &materialization).required(&[
            "TensorData",
            "fn integer_tensor_3_from_bytes",
            "fn integer_tensor_4_from_bytes",
            "fn normalize_3",
            "fn normalize_4",
        ]),
        PatternCheck::new("CUDA decode orchestration", &cuda)
            .required(&["mod config;", "mod interop;", "fn decode_plans_into"])
            .forbidden(&[
                "empty_device_contiguous_dtype",
                "CudaExternalDeviceBufferViewMut::from_raw_parts(",
                "fn kernel_config",
            ]),
        PatternCheck::new("CUDA allocation and context interop", &interop).required(&[
            "empty_device_contiguous_dtype",
            "CudaExternalDeviceBufferViewMut::from_raw_parts(",
            "register_tensor_handle(handle)",
        ]),
        PatternCheck::new("CUDA conversion configuration", &config).required(&[
            "fn kernel_config",
            "CudaJ2kMlKernelConfig",
            "CudaJ2kMlNormalization::MeanStd",
        ]),
    ]);
}

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
fn j2k_ml_quick_metal_feature_graph_excludes_cuda() {
    let output = Command::new("cargo")
        .args([
            "tree",
            "--locked",
            "-p",
            "j2k-ml",
            "--no-default-features",
            "--features",
            "metal",
            "-e",
            "features",
            "--prefix",
            "none",
        ])
        .current_dir(repo_root())
        .output()
        .expect("resolve the j2k-ml Metal feature graph");
    assert!(
        output.status.success(),
        "j2k-ml Metal feature resolution failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let graph = String::from_utf8(output.stdout).expect("Cargo feature graph is UTF-8");
    assert!(
        graph.starts_with("j2k-ml v"),
        "Cargo feature graph must be rooted at j2k-ml:\n{graph}"
    );
    for forbidden in ["j2k-cuda", "j2k-cuda-runtime", "burn-cuda"] {
        assert!(
            !graph.contains(forbidden),
            "j2k-ml's Metal-only feature graph unexpectedly contains {forbidden}:\n{graph}"
        );
    }

    let manifest = fs::read_to_string(repo_root().join("crates/j2k-ml/Cargo.toml"))
        .expect("read j2k-ml manifest");
    let metal_feature = manifest
        .split("metal = [")
        .nth(1)
        .and_then(|rest| rest.split(']').next())
        .expect("j2k-ml metal feature declaration");
    assert!(
        !metal_feature.contains("cuda"),
        "j2k-ml's Metal feature must not enable CUDA: {metal_feature}"
    );
}

#[test]
fn j2k_ml_uses_a_portable_arm_linux_test_backend() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("crates/j2k-ml/Cargo.toml")
                .named("j2k-ml target-specific test backends")
                .required(&[
                    "target.'cfg(all(target_arch = \"aarch64\", target_os = \"linux\"))'.dev-dependencies",
                    "burn-ndarray = { workspace = true }",
                    "target.'cfg(not(all(target_arch = \"aarch64\", target_os = \"linux\")))'.dev-dependencies",
                    "burn-flex = { workspace = true }",
                ]),
            FilePatternCheck::new("docs/j2k-ml.md")
                .named("j2k-ml ARM backend rationale")
                .required(&[
                    "Linux AArch64",
                    "https://github.com/sarah-quinones/gemm/issues/31",
                    "https://github.com/sarah-quinones/gemm/pull/43",
                ]),
        ],
    );
}

#[test]
fn j2k_ml_accelerator_transfer_contracts_are_source_enforced() {
    let root = repo_root();
    let cuda =
        fs::read_to_string(root.join("crates/j2k-ml/src/cuda.rs")).expect("read j2k-ml CUDA route");
    let cuda_interop = fs::read_to_string(root.join("crates/j2k-ml/src/cuda/interop.rs"))
        .expect("read j2k-ml CUDA allocation interop");
    let cuda_owners = format!("{cuda}\n{cuda_interop}");
    let metal = fs::read_to_string(root.join("crates/j2k-ml/src/metal.rs"))
        .expect("read j2k-ml Metal route");
    let readback = fs::read_to_string(root.join("crates/j2k-metal/src/surface/readback.rs"))
        .expect("read packed Metal surface readback");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-ml CUDA direct allocation bridge", &cuda_owners)
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
