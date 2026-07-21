// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use syn::visit::{self, Visit};

use super::read;

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

fn assert_workload_catalog_evidence() {
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
}

fn assert_cpu_benchmark_evidence() {
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
}

fn assert_cuda_benchmark_evidence() {
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
}

fn assert_metal_benchmark_evidence() {
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
}

#[test]
fn j2k_ml_batch_benchmarks_cover_native_medical_outputs_and_all_requests() {
    assert_workload_catalog_evidence();
    assert_cpu_benchmark_evidence();
    assert_cuda_benchmark_evidence();
    assert_metal_benchmark_evidence();

    let benchmark_docs = read("docs/j2k-ml.md");
    assert!(benchmark_docs.contains("session-cumulative"));
    assert!(benchmark_docs.contains("rather than per-case peak"));
}
