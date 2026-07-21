// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, fs};

use syn::visit::{self, Visit};

use super::{
    assert_file_pattern_checks, assert_pattern_checks, repo_root, FilePatternCheck, PatternCheck,
};

mod benchmark_support_structure;

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative)).unwrap_or_else(|error| {
        panic!("read {relative}: {error}");
    })
}

#[derive(Default)]
struct RustEvidence {
    identifiers: BTreeSet<String>,
    string_literals: Vec<String>,
}

impl<'ast> Visit<'ast> for RustEvidence {
    fn visit_ident(&mut self, identifier: &'ast proc_macro2::Ident) {
        self.identifiers.insert(identifier.to_string());
        visit::visit_ident(self, identifier);
    }

    fn visit_lit_str(&mut self, literal: &'ast syn::LitStr) {
        self.string_literals.push(literal.value());
    }

    fn visit_member(&mut self, member: &'ast syn::Member) {
        if let syn::Member::Named(identifier) = member {
            self.identifiers.insert(identifier.to_string());
        }
        visit::visit_member(self, member);
    }

    fn visit_expr_field(&mut self, expression: &'ast syn::ExprField) {
        if let syn::Member::Named(identifier) = &expression.member {
            self.identifiers.insert(identifier.to_string());
        }
        visit::visit_expr_field(self, expression);
    }

    fn visit_macro(&mut self, expression: &'ast syn::Macro) {
        collect_macro_string_literals(expression.tokens.clone(), &mut self.string_literals);
        visit::visit_macro(self, expression);
    }
}

fn collect_macro_string_literals(
    tokens: proc_macro2::TokenStream,
    string_literals: &mut Vec<String>,
) {
    for token in tokens {
        match token {
            proc_macro2::TokenTree::Group(group) => {
                collect_macro_string_literals(group.stream(), string_literals);
            }
            proc_macro2::TokenTree::Literal(literal) => {
                if let Ok(literal) = syn::parse_str::<syn::LitStr>(&literal.to_string()) {
                    string_literals.push(literal.value());
                }
            }
            proc_macro2::TokenTree::Ident(_) | proc_macro2::TokenTree::Punct(_) => {}
        }
    }
}

fn rust_evidence(relative_paths: &[&str]) -> RustEvidence {
    let mut evidence = RustEvidence::default();
    for relative in relative_paths {
        let syntax = syn::parse_file(&read(relative))
            .unwrap_or_else(|error| panic!("parse {relative} as Rust: {error}"));
        evidence.visit_file(&syntax);
    }
    evidence
}

fn assert_identifier_evidence(label: &str, evidence: &RustEvidence, required: &[&str]) {
    let missing = required
        .iter()
        .copied()
        .filter(|identifier| !evidence.identifiers.contains(*identifier))
        .collect::<Vec<_>>();
    assert!(
        missing.is_empty(),
        "{label} missing identifiers: {missing:?}"
    );
}

fn assert_string_literal_evidence(label: &str, evidence: &RustEvidence, required: &[&str]) {
    let joined = evidence.string_literals.join("\n");
    let missing = required
        .iter()
        .copied()
        .filter(|value| !joined.contains(value))
        .collect::<Vec<_>>();
    assert!(
        missing.is_empty(),
        "{label} missing string-literal evidence: {missing:?}"
    );
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
    let catalog = rust_evidence(&["crates/j2k-ml/benches/support/workload_catalog.rs"]);
    assert_string_literal_evidence(
        "native benchmark workload catalog",
        &catalog,
        &[
            "gray12_512",
            "gray12_1024",
            "gray16_512",
            "gray16_1024",
            "gray_i12_512",
            "gray_i12_1024",
            "gray_i16_512",
            "gray_i16_1024",
            "rgb8_256",
            "rgb8_512",
            "rgb16_256",
            "rgb16_512",
            "rgba8_256",
            "rgba8_512",
            "rgba16_256",
            "rgba16_512",
        ],
    );

    let workload = rust_evidence(&["crates/j2k-ml/benches/support/workload.rs"]);
    assert_identifier_evidence(
        "reusable benchmark workload construction",
        &workload,
        &["Workload", "WorkloadSpec", "signed", "materialize_workload"],
    );
    let fixture = rust_evidence(&["crates/j2k-ml/benches/support/fixture.rs"]);
    assert_identifier_evidence(
        "native benchmark fixture construction",
        &fixture,
        &["wrap_benchmark_rgba_jph", "J2kChannelType", "Opacity"],
    );

    let cpu = rust_evidence(&["crates/j2k-ml/benches/batch_decode.rs"]);
    assert_identifier_evidence(
        "CPU batch benchmark",
        &cpu,
        &[
            "Throughput",
            "require_prepared_success",
            "require_codec_success",
            "require_burn_success",
        ],
    );
    assert_string_literal_evidence("CPU preparation benchmark", &cpu, &["prepare_images"]);

    let cuda = rust_evidence(&[
        "crates/j2k-ml/benches/batch_decode_cuda.rs",
        "crates/j2k-ml/benches/cuda_telemetry.rs",
    ]);
    assert_identifier_evidence(
        "CUDA benchmark telemetry",
        &cuda,
        &[
            "requests",
            "CudaSessionDiagnostics",
            "require_prepared_success",
            "require_codec_success",
            "require_burn_success",
        ],
    );
    assert_string_literal_evidence(
        "CUDA benchmark telemetry",
        &cuda,
        &[
            "staged_cpu_upload_pixels",
            "h2d_ops",
            "codec_kernel_launches",
            "runtime_device_allocations",
            "session_peak_live_device_bytes_before",
            "session_peak_live_device_bytes_after",
            "session_pool_peak_retained_bytes_before",
            "session_pool_peak_retained_bytes_after",
            "codec_resident",
            "burn_direct",
        ],
    );

    let metal = rust_evidence(&[
        "crates/j2k-ml/benches/batch_decode_metal.rs",
        "crates/j2k-ml/benches/metal_telemetry.rs",
    ]);
    assert_identifier_evidence(
        "Metal benchmark telemetry",
        &metal,
        &[
            "requests",
            "input_mode",
            "asserted_decoded_host_uploads",
            "asserted_codec_group_waits",
            "require_prepared_success",
            "require_codec_success",
            "require_burn_success",
        ],
    );
    assert_string_literal_evidence(
        "Metal benchmark telemetry",
        &metal,
        &[
            "staged_cpu_upload_pixels",
            "metal_telemetry_v2",
            "asserted_final_output_allocations",
            "measured_codec_group_launches",
            "asserted_consumer_host_syncs",
            "session_pool_peak_cached_bytes_before",
            "session_pool_peak_cached_bytes_after",
        ],
    );

    let benchmark_docs = read("docs/j2k-ml.md");
    assert!(benchmark_docs.contains("session-cumulative"));
    assert!(benchmark_docs.contains("rather than per-case peak"));
}
