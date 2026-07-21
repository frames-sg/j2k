// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::{
    assert_file_pattern_checks, assert_pattern_checks, repo_root, FilePatternCheck, PatternCheck,
};

mod benchmark_support_structure;

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
fn j2k_ml_is_a_thin_persistent_batch_adapter() {
    let library_path = "crates/j2k-ml/src/lib.rs";
    let cpu_module_path = "crates/j2k-ml/src/cpu.rs";
    let cpu_batch_path = "crates/j2k-ml/src/cpu/batch.rs";
    let cuda_module_path = "crates/j2k-ml/src/cuda.rs";
    let cuda_batch_path = "crates/j2k-ml/src/cuda/batch.rs";
    let metal_module_path = "crates/j2k-ml/src/metal.rs";
    let metal_batch_path = "crates/j2k-ml/src/metal/batch.rs";
    let library = read(library_path);
    let cpu_module = read(cpu_module_path);
    let cpu_batch = read(cpu_batch_path);
    let cuda_module = read(cuda_module_path);
    let cuda_batch = read(cuda_batch_path);
    let metal_module = read(metal_module_path);
    let metal_batch = read(metal_batch_path);
    let all_adapter_sources = format!(
        "{library}\n{cpu_module}\n{cpu_batch}\n{cuda_module}\n{cuda_batch}\n{metal_module}\n{metal_batch}"
    );

    assert_below(library_path, &library, 180);
    assert_below(cpu_module_path, &cpu_module, 30);
    assert_below(cuda_module_path, &cuda_module, 40);
    assert_below(metal_module_path, &metal_module, 40);
    assert_pattern_checks(&[
        PatternCheck::new("shared Burn batch output", &library).required(&[
            "pub enum BurnBatchTensor",
            "pub struct BurnBatchGroup",
            "pub struct BurnBatchDecode",
        ]),
        PatternCheck::new("CPU session adapter", &cpu_batch)
            .required(&[
                "pub struct CpuBurnDecoder",
                "j2k::CpuBatchDecoder",
                "pub fn decode_prepared",
                "TensorData::new",
            ])
            .forbidden(&["J2kDecoder::new", "decode_components_with_context"]),
        PatternCheck::new("CUDA session adapter", &cuda_batch)
            .required(&[
                "pub struct CudaBurnDecoder",
                "CudaBatchDecoder as CodecDecoder",
                ".context_for_device_interop(self.device.index)",
                "submit_batch_into",
            ])
            .forbidden(&[
                "context: Option<CudaContext>",
                "fn ensure_context(",
                "J2kDecoder::new",
                "decode_request_to_device_with_session",
            ]),
        PatternCheck::new("Metal session adapter", &metal_batch)
            .required(&[
                "pub struct MetalBurnDecoder",
                "MetalBatchDecoder as CodecDecoder",
                "submit_prepared_group_into_for_consumer_queue(",
            ])
            .forbidden(&["J2kDecoder::new", "decode_request_to_device_with_session"]),
        PatternCheck::new("training policy stays outside j2k-ml", &all_adapter_sources).forbidden(
            &[
                "FloatNormalization",
                "PanicOnDecodeError",
                "decode_float",
                "MeanStd",
                "Dataset",
                "augmentation",
                "prefetch",
            ],
        ),
    ]);
}

#[test]
fn metal_burn_decoder_keeps_batch_options_in_the_codec_session_only() {
    let metal_batch = read("crates/j2k-ml/src/metal/batch.rs");

    assert!(
        !metal_batch.contains("\n    options: BatchDecodeOptions,\n"),
        "MetalBurnDecoder must not duplicate options already retained by CodecDecoder"
    );
    assert!(
        metal_batch.contains(".field(\"options\", &self.codec.options())"),
        "MetalBurnDecoder Debug must read the codec-owned options"
    );
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
fn j2k_ml_accelerator_zero_copy_contracts_are_source_enforced() {
    let cuda_batch = read("crates/j2k-ml/src/cuda/batch.rs");
    let cuda_interop = read("crates/j2k-ml/src/cuda/interop.rs");
    let cuda_owners = format!("{cuda_batch}\n{cuda_interop}");
    let metal_batch = read("crates/j2k-ml/src/metal/batch.rs");
    let metal_interop = read("crates/j2k-ml/src/metal/interop.rs");
    let metal_owners = format!("{metal_batch}\n{metal_interop}");

    assert_pattern_checks(&[
        PatternCheck::new("CUDA Burn-owned destination", &cuda_owners)
            .required(&[
                "empty_device_contiguous_dtype",
                "external_write_stream",
                "with_primary_stream_ordering",
                "CudaExternalDeviceBufferViewMut::from_raw_parts(",
                "submit_batch_into",
                "register_tensor_handle(handle)",
            ])
            .forbidden(&[
                ".sync()",
                "TensorData",
                "copy_to_host(",
                "copy_range_to_host(",
                "Tensor::from_data(",
                "j2k_ml_convert_into_external",
            ]),
        PatternCheck::new("Metal Burn-owned destination", &metal_owners)
            .required(&[
                "checked_next_multiple_of(4)",
                "client.empty(tracked_len)",
                "CubeTensor::new_contiguous",
                "tracked_external_write_range(",
                "mark_external_write_initialized(initialized_range)",
                ".as_hal::<wgpu_hal::api::Metal>()",
                ".retained_raw_handle()",
                "MetalImageDestination::from_exclusive_buffer",
                "MetalBackendSession::with_command_queue",
                "submit_prepared_group_into_for_consumer_queue(",
                "register_tensor_handle(handle)",
            ])
            .forbidden(&[
                "download_surfaces_packed",
                "TensorData",
                "Tensor::from_data(",
                "integer_tensor_4_from_bytes",
                "empty_device_contiguous_dtype",
                ".enqueue_consumer_wait(",
            ]),
    ]);
}

#[test]
fn j2k_ml_batch_benchmarks_cover_native_medical_outputs_and_all_requests() {
    let support = [
        read("crates/j2k-ml/benches/support/mod.rs"),
        read("crates/j2k-ml/benches/support/decode_case.rs"),
        read("crates/j2k-ml/benches/support/fixture.rs"),
        read("crates/j2k-ml/benches/support/input_selection.rs"),
        read("crates/j2k-ml/benches/support/process_policy.rs"),
        read("crates/j2k-ml/benches/support/workload.rs"),
    ]
    .join("\n");
    let cpu = read("crates/j2k-ml/benches/batch_decode.rs");
    let cuda_main = read("crates/j2k-ml/benches/batch_decode_cuda.rs");
    let cuda_telemetry = read("crates/j2k-ml/benches/cuda_telemetry.rs");
    let cuda = format!("{cuda_main}\n{cuda_telemetry}");
    let metal_main = read("crates/j2k-ml/benches/batch_decode_metal.rs");
    let metal_telemetry = read("crates/j2k-ml/benches/metal_telemetry.rs");
    let metal = format!("{metal_main}\n{metal_telemetry}");
    let benchmark_docs = read("docs/j2k-ml.md");

    assert_below(
        "crates/j2k-ml/benches/batch_decode_cuda.rs",
        &cuda_main,
        500,
    );
    assert_below(
        "crates/j2k-ml/benches/batch_decode_metal.rs",
        &metal_main,
        500,
    );
    assert_below(
        "crates/j2k-ml/benches/cuda_telemetry.rs",
        &cuda_telemetry,
        300,
    );
    assert_below(
        "crates/j2k-ml/benches/metal_telemetry.rs",
        &metal_telemetry,
        300,
    );

    assert_pattern_checks(&[
        PatternCheck::new("native benchmark workload matrix", &support).required(&[
            "gray12_512",
            "gray_i12_512",
            "gray_i16_1024",
            "rgb8_256",
            "rgba8_256",
            "rgba16_512",
            "signed: bool",
            "wrap_benchmark_rgba_jph",
            "J2kChannelType::Opacity",
        ]),
        PatternCheck::new("CPU preparation benchmark", &cpu).required(&[
            "prepare_images",
            "Throughput::Elements",
            "require_prepared_success",
            "require_codec_success",
            "require_burn_success",
        ]),
        PatternCheck::new("CUDA RegionReduced benchmark", &cuda).required(&[
            "requests(workload.dimensions, true)",
            "staged_cpu_upload_pixels",
            "CudaSessionDiagnostics",
            "host_to_device_operations",
            "kernel_launches",
            "device_allocation_operations",
            "session_peak_live_device_bytes_before",
            "session_peak_live_device_bytes_after",
            "session_pool_peak_retained_bytes_before",
            "session_pool_peak_retained_bytes_after",
            "codec_resident",
            "burn_direct",
            "require_prepared_success",
            "require_codec_success",
            "require_burn_success",
        ]),
        PatternCheck::new("Metal RegionReduced benchmark", &metal).required(&[
            "requests(workload.dimensions, true)",
            "staged_cpu_upload_pixels",
            "metal_telemetry_v2",
            "input_mode",
            "asserted_decoded_host_uploads",
            "asserted_final_output_allocations",
            "measured_codec_group_launches",
            "asserted_codec_group_waits",
            "asserted_consumer_host_syncs",
            "session_pool_peak_cached_bytes_before",
            "session_pool_peak_cached_bytes_after",
            "require_prepared_success",
            "require_codec_success",
            "require_burn_success",
        ]),
        PatternCheck::new("CUDA high-water benchmark disclosure", &benchmark_docs)
            .required(&["session-cumulative", "rather than per-case peak"]),
    ]);
}
