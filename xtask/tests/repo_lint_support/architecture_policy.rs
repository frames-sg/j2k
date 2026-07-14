// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::{
    architecture_doc_dependency_edges, assert_file_pattern_checks, assert_pattern_checks,
    cargo_metadata_workspace_edges, cargo_public_api_required, const_array_block, format_edge,
    repo_root, rust_sources, stable_api_snapshot_sources, xtask_sources, FilePatternCheck,
    PatternCheck,
};

#[test]
fn workspace_contains_public_j2k_crate() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("crates/j2k/Cargo.toml")
                .named("j2k manifest")
                .required(&["name = \"j2k\"", "j2k-core", "j2k-native", "j2k-types"]),
            FilePatternCheck::new("Cargo.toml")
                .named("workspace manifest")
                .required(&["\"crates/j2k\""]),
        ],
    );
}

#[test]
fn j2k_public_crate_uses_explicit_upstream_reexports() {
    let root = repo_root();
    let public_api_path = root.join("crates/j2k/src/lib.rs");
    let public_api = fs::read_to_string(&public_api_path).unwrap_or_else(|err| {
        panic!("read {}: {err}", public_api_path.display());
    });

    let glob_reexports = public_api
        .lines()
        .enumerate()
        .filter_map(|(idx, line)| {
            let trimmed = line.trim();
            (trimmed.starts_with("pub use j2k_") && trimmed.ends_with("::*;"))
                .then(|| format!("{}:{}", idx + 1, trimmed))
        })
        .collect::<Vec<_>>();
    assert!(
        glob_reexports.is_empty(),
        "j2k public crate must explicitly list upstream reexports:\n{}",
        glob_reexports.join("\n")
    );
}

#[test]
fn architecture_dependency_graph_matches_cargo_metadata() {
    let root = repo_root();
    let metadata_edges = cargo_metadata_workspace_edges(root);
    let docs =
        fs::read_to_string(root.join("docs/architecture.md")).expect("read architecture docs");
    let docs_edges = architecture_doc_dependency_edges(&docs);

    let missing = metadata_edges
        .difference(&docs_edges)
        .map(format_edge)
        .collect::<Vec<_>>();
    let extra = docs_edges
        .difference(&metadata_edges)
        .map(format_edge)
        .collect::<Vec<_>>();

    assert!(
        missing.is_empty() && extra.is_empty(),
        "docs/architecture.md crate dependency graph drifted from cargo metadata\n\
         missing from docs: {missing:#?}\n\
         not in cargo metadata: {extra:#?}"
    );
}

#[test]
fn architecture_docs_classify_workspace_and_in_repo_tool_crates() {
    assert_file_pattern_checks(
        repo_root(),
        &[FilePatternCheck::new("docs/architecture.md")
            .named("architecture docs")
            .required(&[
                "`j2k-test-support`",
                "`j2k-transcode-test-support`",
                "dev helper",
                "`xtask`",
                "workspace tool",
                "`xtask/`",
            ])
            .forbidden(&["All crates live under `crates/`"])],
    );
}

#[test]
fn tooling_and_validation_crates_stay_unpublished() {
    let root = repo_root();
    let xtask = xtask_sources(root);
    let publishable = const_array_block(&xtask, "PUBLISHABLE_PACKAGES");

    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("crates/j2k-test-support/Cargo.toml")
                .named("j2k-test-support manifest")
                .required(&["publish = false"]),
            FilePatternCheck::new("crates/j2k-transcode-test-support/Cargo.toml")
                .named("j2k-transcode-test-support manifest")
                .required(&["publish = false"]),
            FilePatternCheck::new("xtask/Cargo.toml")
                .named("xtask manifest")
                .required(&["publish = false"]),
            FilePatternCheck::new("Cargo.toml")
                .named("workspace manifest")
                .required(&[
                    "\"crates/j2k-test-support\"",
                    "\"crates/j2k-transcode-test-support\"",
                    "\"xtask\"",
                ])
                .forbidden(&["\"tests/nvidia-baseline\""]),
        ],
    );

    for package in ["j2k-test-support", "j2k-transcode-test-support", "xtask"] {
        assert!(
            !publishable.contains(&format!("\"{package}\"")),
            "xtask publishable package gate must not include {package}"
        );
    }
}

#[test]
fn public_crates_do_not_reexport_j2k_native() {
    let root = repo_root();
    let mut offenders = Vec::new();

    for crate_dir in [
        "crates/j2k/src",
        "crates/j2k-jpeg/src",
        "crates/j2k-transcode/src",
        "crates/j2k-metal/src",
        "crates/j2k-cuda/src",
        "crates/j2k-transcode-metal/src",
        "crates/j2k-transcode-cuda/src",
    ] {
        for path in rust_sources(&root.join(crate_dir)) {
            let source = fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
            for (line_idx, line) in source.lines().enumerate() {
                let trimmed = line.trim_start();
                if trimmed.starts_with("pub use j2k_native")
                    || trimmed.starts_with("pub type ") && trimmed.contains("j2k_native")
                {
                    offenders.push(format!(
                        "{}:{}:{}",
                        path.strip_prefix(root).unwrap_or(&path).display(),
                        line_idx + 1,
                        line.trim()
                    ));
                }
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "public crates must not re-export native J2K implementation types:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn native_decode_error_boundaries_preserve_owned_sources_and_shared_classification() {
    let forbidden_native_matches = [
        "NativeDecodeError::Format",
        "NativeDecodeError::Marker",
        "NativeDecodeError::Decoding",
        "native_direct_plan_unsupported_what",
    ];

    let root = repo_root();
    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("crates/j2k-native/src/error.rs")
                .named("native decode error classification")
                .required(&[
                    "pub enum DecodeErrorClass",
                    "pub const fn classify(&self)",
                    "direct_plan_unsupported_what(reason)",
                ]),
            FilePatternCheck::new("crates/j2k/src/error.rs")
                .named("facade native decode error mapper")
                .required(&[
                    "pub(crate) fn from_native_decode_error(error: NativeDecodeError)",
                    "match error.classify()",
                    "Self::NativeDecode",
                    "source: NativeBackendError::decode(error)",
                ]),
        ],
    );

    let facade = fs::read_to_string(root.join("crates/j2k/src/error.rs"))
        .expect("read j2k facade error mapper");
    let facade_production = facade.split("\n#[cfg(test)]").next().unwrap_or(&facade);
    for forbidden in forbidden_native_matches {
        assert!(
            !facade_production.contains(forbidden),
            "j2k facade mapper must use DecodeErrorClass instead of native internals: {forbidden}"
        );
    }

    for (path, native_source_path) in [
        (
            "crates/j2k-cuda/src/error.rs",
            "crates/j2k-cuda/src/error/native_source.rs",
        ),
        (
            "crates/j2k-metal/src/error.rs",
            "crates/j2k-metal/src/error/native_source.rs",
        ),
    ] {
        let source =
            fs::read_to_string(root.join(path)).unwrap_or_else(|err| panic!("read {path}: {err}"));
        let production = source.split("\n#[cfg(test)]").next().unwrap_or(&source);
        let native_source = fs::read_to_string(root.join(native_source_path))
            .unwrap_or_else(|err| panic!("read {native_source_path}: {err}"));
        assert!(
            production.contains("NativeDecode {")
                && production.contains("source: NativeBackendError")
                && production.contains("source: NativeBackendError::decode(error)")
                && native_source.contains("pub struct NativeBackendError")
                && native_source.contains("source.classify()")
                && native_source.contains("DecodeErrorClass::InputTooShort")
                && native_source.contains("DecodeErrorClass::InputTruncatedAt")
                && native_source.contains("DecodeErrorClass::Unsupported")
                && native_source.contains("impl core::error::Error for NativeBackendError"),
            "{path} must own opaque native sources and classify them through DecodeErrorClass"
        );
        assert!(
            source.contains("fn native_decode_resource_errors_preserve_typed_sources()"),
            "{path} must retain resource-category parity coverage"
        );
        let adapter_boundary = format!("{production}\n{native_source}");
        for forbidden in ["J2kError::from_native_decode_error", "error.to_string()"] {
            assert!(
                !adapter_boundary.contains(forbidden),
                "{path} must not cross the facade conversion boundary or erase native sources: {forbidden}"
            );
        }
        for forbidden in forbidden_native_matches {
            assert!(
                !adapter_boundary.contains(forbidden),
                "{path} must not match native error internals directly: {forbidden}"
            );
        }
    }
}

#[test]
fn facade_backend_errors_require_explicit_classification() {
    assert_file_pattern_checks(
        repo_root(),
        &[FilePatternCheck::new("crates/j2k/src/error.rs")
            .named("facade backend error classification")
            .required(&[
                "pub fn new(kind: BackendErrorKind, message: impl Into<String>)",
                "pub fn truncated(message: impl Into<String>)",
                "pub fn not_implemented(message: impl Into<String>)",
                "pub fn unsupported(message: impl Into<String>)",
                "pub fn buffer(message: impl Into<String>)",
                "fn backend_error_kind_drives_codec_classification()",
            ])
            .forbidden(&[
                "impl From<String> for BackendError",
                "impl From<&str> for BackendError",
            ])],
    );
}

#[test]
#[ignore = "strict repo-lint: shells out to cargo-public-api for all public API packages"]
fn rendered_public_api_does_not_expose_j2k_native() {
    let root = repo_root();

    for package in [
        "j2k",
        "j2k-jpeg",
        "j2k-transcode",
        "j2k-metal",
        "j2k-cuda",
        "j2k-transcode-metal",
        "j2k-transcode-cuda",
    ] {
        let api = cargo_public_api_required(root, package);
        assert!(
            !api.contains("j2k_native"),
            "public API for package {package} exposes j2k_native:\n{api}"
        );
    }
}

#[test]
fn obsolete_adaptive_route_policy_model_cannot_return() {
    let root = repo_root();

    for relative in [
        "crates/j2k/src/adapter/adaptive_route.rs",
        "crates/j2k/src/adapter/adaptive_route_tests.rs",
    ] {
        assert!(
            !root.join(relative).exists(),
            "obsolete test-only adaptive route policy must not return: {relative}"
        );
    }

    let mut model_offenders = Vec::new();
    for path in rust_sources(&root.join("crates/j2k/src")) {
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        for symbol in [
            "J2kAdaptiveBenchmarkEvidence",
            "J2kAdaptiveRcaFinding",
            "J2kAdaptiveRoutePlanner",
        ] {
            if source.contains(symbol) {
                model_offenders.push(format!(
                    "{}: {symbol}",
                    path.strip_prefix(root).unwrap_or(&path).display()
                ));
            }
        }
    }
    assert!(
        model_offenders.is_empty(),
        "obsolete synthetic adaptive route model must not return:\n{}",
        model_offenders.join("\n")
    );

    assert_file_pattern_checks(
        root,
        &[FilePatternCheck::new("crates/j2k/src/adapter/mod.rs")
            .named("j2k adapter module")
            .forbidden(&["mod adaptive_route;", "pub mod adaptive_route;"])],
    );
    let stable_api = stable_api_snapshot_sources(root);
    assert_pattern_checks(
        &[PatternCheck::new("stable API snapshot union", &stable_api)
            .forbidden(&["j2k::adapter::adaptive_route"])],
    );
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the public-API denylist is one fail-closed architecture contract and is reviewed as a whole"
)]
fn accidental_test_and_adapter_internals_stay_out_of_public_api() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("crates/j2k-cuda/src/lib.rs")
                .named("j2k-cuda public exports")
                .forbidden(&[
                    "cuda_dwt53_output_to_j2k_for_test",
                    "pub use direct_plan::{",
                ]),
            FilePatternCheck::new("crates/j2k-jpeg/src/adapter/mod.rs")
                .named("j2k-jpeg adapter module")
                .required(&["mod fast_packet;", "pub use fast_packet::{"])
                .forbidden(&["pub mod fast_packet;"]),
            FilePatternCheck::new("crates/j2k-jpeg/src/lib.rs")
                .named("j2k-jpeg decoder root facade")
                .required(&[
                    "mod info;",
                    "pub use info::{",
                    "mod context;",
                    "pub use context::DecoderContext;",
                    "mod batch_session;",
                    "pub use batch_session::JpegBatchSession;",
                    "mod capabilities;",
                    "pub use capabilities::{",
                    "mod output_buffer;",
                    "pub use output_buffer::JpegOutputBuffer;",
                    "mod segment;",
                    "pub use segment::{",
                    "mod error;",
                    "pub use error::{",
                    "mod encoder;",
                    "pub use encoder::{",
                    "mod decoder;",
                    "pub use decoder::{",
                ])
                .forbidden(&[
                    "pub mod info;",
                    "pub mod context;",
                    "pub mod batch_session;",
                    "pub mod capabilities;",
                    "pub mod output_buffer;",
                    "pub mod segment;",
                    "pub mod error;",
                    "pub mod encoder;",
                    "pub mod decoder;",
                ]),
            FilePatternCheck::new("crates/j2k/src/lib.rs")
                .named("j2k view root facade")
                .required(&[
                    "mod adapter;",
                    "mod context;",
                    "mod error;",
                    "mod scratch;",
                    "mod view;",
                    "pub use adapter::encode_stage::{",
                    "pub use context::J2kContext;",
                    "pub use error::{BackendError, BackendErrorKind, J2kError, NativeBackendError};",
                    "pub use scratch::J2kScratchPool;",
                    "pub use view::{",
                ])
                .forbidden(&[
                    "pub mod adapter;",
                    "pub mod context;",
                    "pub mod error;",
                    "pub mod scratch;",
                    "pub mod view;",
                ]),
            FilePatternCheck::new("crates/j2k-native/src/lib.rs")
                .named("j2k-native error root facade")
                .required(&["mod error;", "pub use error::{"])
                .forbidden(&["pub mod error;"]),
            FilePatternCheck::new("crates/j2k-core/src/lib.rs")
                .named("j2k-core passthrough root facade")
                .required(&[
                    "mod passthrough;",
                    "pub use passthrough::{",
                    "PassthroughCandidate",
                    "PassthroughDecision",
                    "PassthroughRejectReason",
                    "PassthroughRequirements",
                ])
                .forbidden(&["pub mod passthrough;"]),
            FilePatternCheck::new("crates/j2k-core/src/lib.rs")
                .named("j2k-core error root facade")
                .required(&[
                    "mod error;",
                    "pub use error::{",
                    "AdapterErrorKind",
                    "AdapterErrorParts",
                    "BufferError",
                    "CodecError",
                    "InputError",
                ])
                .forbidden(&["pub mod error;"]),
            FilePatternCheck::new("crates/j2k-core/src/lib.rs")
                .named("j2k-core shared-contract root facades")
                .required(&[
                    "mod backend;",
                    "mod batch;",
                    "mod context;",
                    "mod device;",
                    "mod traits;",
                    "mod types;",
                    "pub use backend::{BackendCapabilities, BackendKind, BackendRequest, CpuFeatures};",
                    "pub use context::{CacheStats, CodecContext, DecoderContext};",
                    "pub use device::validate_cuda_surface_backend_request;",
                    "pub use traits::{",
                    "pub use types::{CodedUnitLayout, Colorspace, DecodeOutcome, Info, Rect, TileLayout};",
                ])
                .forbidden(&[
                    "pub mod backend;",
                    "pub mod batch;",
                    "pub mod context;",
                    "pub mod device;",
                    "pub mod traits;",
                    "pub mod types;",
                ]),
            FilePatternCheck::new("crates/j2k-core/src/lib.rs")
                .named("j2k-core row/scratch root facades")
                .required(&[
                    "mod row_sink;",
                    "mod scratch;",
                    "pub use row_sink::RowSink;",
                    "pub use scratch::ScratchPool;",
                ])
                .forbidden(&["pub mod row_sink;", "pub mod scratch;"]),
            FilePatternCheck::new("crates/j2k-core/src/lib.rs")
                .named("j2k-core pixel/sample/scale root facades")
                .required(&[
                    "mod pixel;",
                    "mod sample;",
                    "mod scale;",
                    "pub use pixel::{PixelFormat, PixelLayout};",
                    "pub use sample::{Sample, SampleType};",
                    "pub use scale::Downscale;",
                ])
                .forbidden(&["pub mod pixel;", "pub mod sample;", "pub mod scale;"]),
            FilePatternCheck::new("crates/j2k-transcode/src/lib.rs")
                .named("j2k-transcode transform root facade")
                .required(&[
                    "mod dct53_2d;",
                    "mod dct97_2d;",
                    "mod htj2k97_codeblock_oracle;",
                    "pub use dct53_2d::{",
                    "pub use dct97_2d::dct8x8_blocks_then_dwt97_float;",
                    "pub use htj2k97_codeblock_oracle::{",
                ])
                .forbidden(&[
                    "pub mod dct53_2d;",
                    "pub mod dct97_2d;",
                    "pub mod htj2k97_codeblock_oracle;",
                ]),
            FilePatternCheck::new("crates/j2k-transcode/src/accelerator_contracts.rs")
                .named("j2k-transcode counter facade")
                .required(&[
                    "pub enum DctToWaveletStageCounterEvent",
                    "pub fn record(&mut self, event: DctToWaveletStageCounterEvent, count: usize)",
                ])
                .forbidden(&[
                    "pub fn record_reversible_dwt53_",
                    "pub fn record_dwt53_",
                    "pub fn record_dwt97_",
                    "pub fn record_htj2k97_",
                    "pub fn reversible_dwt53_first_level_from_block_samples",
                ]),
            FilePatternCheck::new("crates/j2k-transcode/src/dct53_2d.rs")
                .named("j2k-transcode 5/3 scratch API")
                .forbidden(&[
                    "Dct53GridError",
                    "pub struct Dct53GridScratch",
                    "pub fn max_abs_diff",
                    "pub fn weight_row_capacity",
                    "pub fn dct8x8_to_dwt53_float_linear",
                    "pub fn idct8x8_then_dwt53_float",
                    "pub fn dct8x8_blocks_to_dwt53_float_linear_with_scratch",
                ]),
            FilePatternCheck::new("crates/j2k-transcode/src/dct97_2d.rs")
                .named("j2k-transcode 9/7 scratch API")
                .required(&[
                    "pub struct Dct97GridScratch",
                    "pub fn dct8x8_blocks_then_dwt97_float_with_scratch",
                ])
                .forbidden(&[
                    "Dct97GridError",
                    "pub fn max_abs_diff",
                    "pub fn spatial_sample_capacity",
                ]),
            FilePatternCheck::new("crates/j2k-transcode/src/jpeg_to_htj2k.rs")
                .named("j2k-transcode transcoder scratch API")
                .forbidden(&[
                    "pub fn dct_block_scratch_capacity",
                    "pub fn integer_idct_block_scratch_capacity",
                ]),
            FilePatternCheck::new("crates/j2k-jpeg-metal/src/lib.rs")
                .named("j2k-jpeg-metal viewport facade")
                .required(&["mod viewport;", "pub use viewport::{"])
                .forbidden(&["pub mod viewport;"]),
            FilePatternCheck::new("docs/stable-api-1.0.public-api.txt")
                .named("ordinary stable API snapshot")
                .required(&[
                    "pub struct j2k_transcode::Dct97GridScratch",
                    "pub fn j2k_transcode::dct8x8_blocks_then_dwt97_float_with_scratch",
                ])
                .forbidden(&[
                    "cuda_dwt53_output_to_j2k_for_test",
                    "crate::adapter::DeviceBatchSummary",
                    "pub mod j2k_jpeg::transcode",
                    "j2k_jpeg::transcode::",
                    "pub mod j2k_jpeg::batch_session",
                    "pub mod j2k_jpeg::capabilities",
                    "pub mod j2k_jpeg::context",
                    "pub mod j2k_jpeg::encoder",
                    "pub mod j2k_jpeg::error",
                    "pub mod j2k_jpeg::info",
                    "pub mod j2k_jpeg::output_buffer",
                    "pub mod j2k_jpeg::segment",
                    "pub mod j2k_native::error",
                    "pub mod j2k::context",
                    "pub mod j2k::error",
                    "pub mod j2k::scratch",
                    "pub mod j2k::view",
                    "pub mod j2k::adapter",
                    "j2k::adapter::encode_stage::",
                    "pub mod j2k::adapter::device_plan",
                    "j2k::adapter::device_plan::",
                    "pub use j2k::CpuOnlyJ2kEncodeStageAccelerator",
                    "pub use j2k::EncodedHtJ2kCodeBlock",
                    "pub use j2k::EncodedJ2kCodeBlock",
                    "pub use j2k::IrreversibleQuantizationStep",
                    "pub use j2k::IrreversibleQuantizationSubbandScales",
                    "pub use j2k::J2kCodeBlockSegment",
                    "pub use j2k::J2kCodeBlockStyle",
                    "pub use j2k::J2kDeinterleaveToF32Job",
                    "pub use j2k::J2kEncodeDispatchReport",
                    "pub use j2k::J2kEncodeStageAccelerator",
                    "pub use j2k::J2kForwardDwt53Job",
                    "pub use j2k::J2kForwardDwt53Level",
                    "pub use j2k::J2kForwardDwt53Output",
                    "pub use j2k::J2kForwardDwt97Job",
                    "pub use j2k::J2kForwardDwt97Level",
                    "pub use j2k::J2kForwardDwt97Output",
                    "pub use j2k::J2kForwardIctJob",
                    "pub use j2k::J2kForwardRctJob",
                    "pub use j2k::J2kHtCodeBlockEncodeJob",
                    "pub use j2k::J2kHtSubbandEncodeJob",
                    "pub use j2k::J2kHtj2kTileEncodeJob",
                    "pub use j2k::J2kPacketizationBlockCodingMode",
                    "pub use j2k::J2kPacketizationCodeBlock",
                    "pub use j2k::J2kPacketizationEncodeJob",
                    "pub use j2k::J2kPacketizationPacketDescriptor",
                    "pub use j2k::J2kPacketizationProgressionOrder",
                    "pub use j2k::J2kPacketizationResolution",
                    "pub use j2k::J2kPacketizationSubband",
                    "pub use j2k::J2kQuantizeSubbandJob",
                    "pub use j2k::J2kSubBandType",
                    "pub use j2k::J2kTier1CodeBlockEncodeJob",
                    "pub use j2k::PrecomputedHtj2k53Component",
                    "pub use j2k::PrecomputedHtj2k53Image",
                    "pub use j2k::PrecomputedHtj2k97Component",
                    "pub use j2k::PrecomputedHtj2k97Image",
                    "pub use j2k::PreencodedHtj2k97CodeBlock",
                    "pub use j2k::PreencodedHtj2k97CompactCodeBlock",
                    "pub use j2k::PreencodedHtj2k97CompactComponent",
                    "pub use j2k::PreencodedHtj2k97CompactImage",
                    "pub use j2k::PreencodedHtj2k97CompactResolution",
                    "pub use j2k::PreencodedHtj2k97CompactSubband",
                    "pub use j2k::PreencodedHtj2k97Component",
                    "pub use j2k::PreencodedHtj2k97Image",
                    "pub use j2k::PreencodedHtj2k97Resolution",
                    "pub use j2k::PreencodedHtj2k97Subband",
                    "pub use j2k::PrequantizedHtj2k97CodeBlock",
                    "pub use j2k::PrequantizedHtj2k97Component",
                    "pub use j2k::PrequantizedHtj2k97Image",
                    "pub use j2k::PrequantizedHtj2k97Resolution",
                    "pub use j2k::PrequantizedHtj2k97Subband",
                    "pub fn j2k::J2kCodec::decode_tile(",
                    "pub fn j2k::J2kCodec::decode_tile_region(",
                    "pub fn j2k::J2kCodec::decode_tile_region_scaled(",
                    "pub fn j2k::J2kCodec::decode_tile_scaled(",
                    "pub fn j2k_jpeg::JpegCodec::decode_tile(",
                    "pub fn j2k_jpeg::JpegCodec::decode_tile_region(",
                    "pub fn j2k_jpeg::JpegCodec::decode_tile_region_scaled(",
                    "pub fn j2k_jpeg::JpegCodec::decode_tile_scaled(",
                    "pub fn j2k_types::CpuOnlyJ2kEncodeStageAccelerator::encode_deinterleave(",
                    "pub fn j2k_transcode::CpuOnlyDctToWaveletStageAccelerator::dct_grid_to_dwt97(",
                    "pub fn j2k_transcode::RayonReversibleDwt53Accelerator::dct_grid_to_reversible_dwt53(",
                    "pub fn j2k_cuda::CudaEncodeStageAccelerator::encode_deinterleave(",
                    "pub fn j2k_metal::MetalEncodeStageAccelerator::encode_deinterleave(",
                    "pub fn j2k_transcode_cuda::CudaDctToWaveletStageAccelerator::dct_grid_to_dwt97(",
                    "pub fn j2k_transcode_metal::MetalDctToWaveletStageAccelerator::dct_grid_to_dwt97(",
                    "pub fn j2k_native::Image<'a>::build_direct_color_plan_with_context(",
                    "pub fn j2k_native::Image<'a>::decode_components_with_ht_decoder(",
                    "pub fn j2k_native::Image<'a>::decode_reversible_53_coefficients(",
                    "pub fn j2k_native::Image<'a>::supports_direct_device_plane_reuse(",
                    "pub fn j2k_native::encode_precomputed_htj2k_53(",
                    "pub fn j2k_native::encode_precomputed_htj2k_97_with_accelerator(",
                    "pub fn j2k_native::encode_preencoded_htj2k_97_owned_with_accelerator(",
                    "pub fn j2k_native::encode_prequantized_htj2k_97(",
                    "pub fn j2k_native::encode_with_accelerator(",
                    "pub fn j2k_native::encode_with_accelerator_and_roi_regions(",
                    "pub fn j2k_native::irreversible_quantization_step_for_subband(",
                    "pub fn j2k_jpeg_metal::decode_viewport_to_surface",
                    "pub fn j2k_jpeg::decode_tile_into_in_context_with_options",
                    "pub fn j2k_jpeg::decode_tile_region_into_in_context_with_options",
                    "pub fn j2k_jpeg::decode_tile_region_scaled_into_in_context_with_options",
                    "pub fn j2k_jpeg::decode_tile_scaled_into_in_context_with_options",
                    "pub fn j2k_jpeg::decode_tiles_into_with_options",
                    "pub fn j2k_jpeg::decode_tiles_region_scaled_into_with_options",
                    "pub fn j2k_jpeg::decode_tiles_scaled_into_with_options",
                    "j2k_jpeg::internal::scratch::ScratchPool",
                    "j2k::scratch::J2kScratchPool",
                    "pub mod j2k_core::passthrough",
                    "pub fn j2k_core::passthrough::",
                    "pub mod j2k_core::row_sink",
                    "pub fn j2k_core::row_sink::",
                    "pub mod j2k_core::scratch",
                    "pub fn j2k_core::scratch::",
                    "pub mod j2k_core::pixel",
                    "pub mod j2k_core::sample",
                    "pub mod j2k_core::scale",
                    "pub mod j2k_core::error",
                    "pub mod j2k_core::backend",
                    "pub mod j2k_core::batch",
                    "pub mod j2k_core::context",
                    "pub mod j2k_core::device",
                    "pub mod j2k_core::traits",
                    "pub mod j2k_core::types",
                    "pub fn j2k_core::error::",
                    "pub fn j2k_core::backend::",
                    "pub fn j2k_core::batch::",
                    "pub fn j2k_core::context::",
                    "pub fn j2k_core::device::",
                    "pub fn j2k_core::traits::",
                    "pub fn j2k_core::pixel::",
                    "pub fn j2k_core::sample::",
                    "pub fn j2k_core::scale::",
                    "pub mod j2k_jpeg::decoder",
                    "pub mod j2k_jpeg_metal::viewport",
                    "pub mod j2k_transcode::dct53_2d",
                    "pub mod j2k_transcode::dct97_2d",
                    "pub mod j2k_transcode::htj2k97_codeblock_oracle",
                    "pub mod j2k_transcode::accelerator",
                    "j2k_jpeg_metal::viewport::",
                    "j2k_transcode::dct53_2d::",
                    "j2k_transcode::dct97_2d::",
                    "j2k_transcode::htj2k97_codeblock_oracle::",
                    "j2k_transcode::accelerator::",
                    "j2k_transcode::idct_blocks_to_signed_samples_rayon",
                    "DctToWaveletStageCounters::record_reversible_dwt53_",
                    "DctToWaveletStageCounters::record_dwt53_",
                    "DctToWaveletStageCounters::record_dwt97_",
                    "DctToWaveletStageCounters::record_htj2k97_",
                    "j2k_transcode::accelerator::reversible_dwt53_first_level_from_block_samples",
                    "j2k_transcode::Dct53GridScratch::weight_row_capacity",
                    "j2k_transcode::Dct97GridScratch::spatial_sample_capacity",
                    "j2k_transcode::Dct53GridScratch",
                    "j2k_transcode::Dwt53TwoDimensional<f64>::max_abs_diff",
                    "j2k_transcode::Dwt97TwoDimensional<f64>::max_abs_diff",
                    "j2k_transcode::Dct53GridError",
                    "j2k_transcode::Dct97GridError",
                    "j2k_transcode::TranscodePipelineMap::debug_report",
                    "j2k_transcode::dct8x8_to_dwt53_float_linear",
                    "j2k_transcode::idct8x8_then_dwt53_float",
                    "j2k_transcode::dct8x8_blocks_to_dwt53_float_linear_with_scratch",
                    "JpegToHtj2kTranscoder::dct_block_scratch_capacity",
                    "JpegToHtj2kTranscoder::integer_idct_block_scratch_capacity",
                    "pub fn j2k::J2kDecoder<'a>::parse(&'a [u8]) -> core::result::Result<Self::View, Self::Error>",
                    "pub fn j2k::J2kDecoder<'a>::decode_into(&mut self, &mut [u8], usize, j2k_core::pixel::PixelFormat) -> core::result::Result<j2k_core::types::DecodeOutcome<Self::Warning>, Self::Error>",
                    "pub fn j2k::J2kDecoder<'a>::decode_into_with_scratch(&mut self, &mut Self::Pool",
                    "pub fn j2k::J2kDecoder<'a>::decode_rows<R: j2k_core::row_sink::RowSink<u8>>",
                    "pub fn j2k::J2kDecoder<'a>::decode_rows<R: j2k_core::row_sink::RowSink<u16>>",
                    "pub fn j2k_jpeg::Decoder<'a>::parse(&'a [u8]) -> core::result::Result<Self::View, Self::Error>",
                    "pub fn j2k_jpeg::Decoder<'a>::decode_into(&mut self, &mut [u8], usize, j2k_core::pixel::PixelFormat) -> core::result::Result<j2k_core::types::DecodeOutcome<Self::Warning>, Self::Error>",
                    "pub fn j2k_jpeg::Decoder<'a>::decode_into_with_scratch(&mut self, &mut Self::Pool",
                    "pub fn j2k_jpeg::Decoder<'a>::decode_rows<R: j2k_core::row_sink::RowSink<u8>>",
                    "impl j2k_core::traits::ImageCodec for j2k::J2kDecoder<'_>",
                    "pub type j2k::J2kDecoder<'_>::Error",
                    "pub type j2k::J2kDecoder<'_>::Pool",
                    "pub type j2k::J2kDecoder<'_>::Warning",
                    "impl j2k_core::traits::ImageCodec for j2k_jpeg::Decoder<'_>",
                    "pub type j2k_jpeg::Decoder<'_>::Error",
                    "pub type j2k_jpeg::Decoder<'_>::Pool",
                    "pub type j2k_jpeg::Decoder<'_>::Warning",
                    "impl j2k_core::context::CodecContext for j2k::J2kContext",
                    "pub fn j2k::J2kContext::cache_stats",
                    "pub fn j2k::J2kContext::clear",
                    "impl j2k_core::scratch::ScratchPool for j2k::J2kScratchPool",
                    "pub fn j2k::J2kScratchPool::bytes_allocated",
                    "pub fn j2k::J2kScratchPool::reset",
                    "impl j2k_core::context::CodecContext for j2k_jpeg::DecoderContext",
                    "pub fn j2k_jpeg::DecoderContext::cache_stats",
                    "pub fn j2k_jpeg::DecoderContext::clear",
                    "impl j2k_core::scratch::ScratchPool for j2k_jpeg::ScratchPool",
                    "pub fn j2k_jpeg::ScratchPool::bytes_allocated",
                    "pub fn j2k_jpeg::ScratchPool::reset",
                    "impl j2k_core::scratch::ScratchPool for j2k_tilecodec::DeflatePool",
                    "pub fn j2k_tilecodec::DeflatePool::bytes_allocated",
                    "pub fn j2k_tilecodec::DeflatePool::reset",
                    "impl j2k_core::scratch::ScratchPool for j2k_tilecodec::LzwPool",
                    "pub fn j2k_tilecodec::LzwPool::bytes_allocated",
                    "pub fn j2k_tilecodec::LzwPool::reset",
                    "impl j2k_core::scratch::ScratchPool for j2k_tilecodec::NoPool",
                    "pub fn j2k_tilecodec::NoPool::bytes_allocated",
                    "pub fn j2k_tilecodec::NoPool::reset",
                    "impl j2k_core::scratch::ScratchPool for j2k_tilecodec::ZstdPool",
                    "pub fn j2k_tilecodec::ZstdPool::bytes_allocated",
                    "pub fn j2k_tilecodec::ZstdPool::reset",
                    "pub fn j2k_transcode::TranscodeStageDispatchMode::recover",
                    "pub fn j2k_transcode::TranscodeStageDispatchMode::unavailable",
                    "pub fn j2k_jpeg::JpegOutputBuffer::new_with_max_bytes",
                    "pub fn j2k_jpeg::JpegOutputBuffer::resize_with_max_bytes",
                    "pub fn j2k_jpeg::JpegOutputBuffer::resize_with_stride_with_max_bytes",
                    "pub fn j2k_jpeg::JpegOutputBuffer::with_stride_with_max_bytes",
                    "j2k_jpeg::Decoder::decode_tile",
                    "j2k_jpeg::Decoder::inspect_with_options",
                    "j2k_jpeg::Decoder::new_with_options",
                    "j2k_jpeg::Decoder::decode_request_with_scratch",
                    "j2k_jpeg::Decoder::decode_region_into_with_scratch",
                    "j2k_jpeg::Decoder::decode_rgba8_into_with_alpha",
                    "j2k_jpeg::Decoder::decode_region_rgba8_into_with_alpha",
                    "j2k_jpeg::Decoder::decode_rgba8_into_with_alpha_with_scratch",
                    "j2k_jpeg::Decoder::decode_region_rgba8_into_with_alpha_with_scratch",
                    "impl j2k_core::error::CodecError for j2k::J2kError",
                    "impl j2k_core::error::CodecError for j2k_jpeg::JpegError",
                    "impl j2k_core::error::CodecError for j2k_cuda::Error",
                    "impl j2k_core::error::CodecError for j2k_metal::Error",
                    "impl j2k_core::error::CodecError for j2k_jpeg_cuda::Error",
                    "impl j2k_core::error::CodecError for j2k_jpeg_metal::Error",
                    "pub fn j2k_core::adapter_error_is_buffer_error",
                    "pub fn j2k_core::adapter_error_is_not_implemented",
                    "pub fn j2k_core::adapter_error_is_truncated",
                    "pub fn j2k_core::adapter_error_is_unsupported",
                    "pub fn j2k_core::checked_surface_len",
                    "pub fn j2k_core::collect_indexed_batch_results",
                    "pub fn j2k_core::copy_tight_pixels_to_strided_output",
                    "pub fn j2k_core::ensure_allocation_within_cap",
                    "pub fn j2k_core::strided_output_len",
                    "pub fn j2k_core::strided_output_len_capped",
                    "pub fn j2k_core::tile_batch_worker_count",
                    "pub fn j2k_core::validate_cuda_surface_backend_request",
                    "pub fn j2k_core::validate_strided_output_buffer",
                    "pub type j2k_core::IndexedBatchResult",
                    "pub fn j2k::decode_tile_into_in_context",
                    "pub fn j2k::decode_tile_region_into_in_context",
                    "pub fn j2k::decode_tile_region_scaled_into_in_context",
                    "pub fn j2k::decode_tile_scaled_into_in_context",
                    "pub fn j2k_jpeg::decode_tile_into_in_context",
                    "pub fn j2k_jpeg::decode_tile_region_into_in_context",
                    "pub fn j2k_jpeg::decode_tile_region_scaled_into_in_context",
                    "pub fn j2k_jpeg::decode_tile_scaled_into_in_context",
                    "pub fn j2k_jpeg::JpegBatchSession::retained_worker_slots",
                    "pub fn j2k_jpeg::JpegBatchSession::worker_count",
                    "impl j2k_core::error::AdapterErrorParts for j2k_cuda::Error",
                    "pub fn j2k_cuda::Error::adapter_error_kind",
                    "pub fn j2k_cuda::Error::source_codec_error",
                    "impl j2k_core::error::AdapterErrorParts for j2k_metal::Error",
                    "pub fn j2k_metal::Error::adapter_error_kind",
                    "pub fn j2k_metal::Error::source_codec_error",
                    "impl j2k_core::error::AdapterErrorParts for j2k_jpeg_cuda::Error",
                    "pub fn j2k_jpeg_cuda::Error::adapter_error_kind",
                    "pub fn j2k_jpeg_cuda::Error::source_codec_error",
                    "impl j2k_core::error::AdapterErrorParts for j2k_jpeg_metal::Error",
                    "pub fn j2k_jpeg_metal::Error::adapter_error_kind",
                    "pub fn j2k_jpeg_metal::Error::source_codec_error",
                    "impl j2k_core::traits::TileDecompress for j2k_tilecodec::DeflateCodec",
                    "impl j2k_core::traits::TileDecompress for j2k_tilecodec::LzwCodec",
                    "impl j2k_core::traits::TileDecompress for j2k_tilecodec::UncompressedCodec",
                    "impl j2k_core::traits::TileDecompress for j2k_tilecodec::ZstdCodec",
                    "pub fn j2k_tilecodec::DeflateCodec::decompress_into",
                    "pub fn j2k_tilecodec::DeflateCodec::expected_size",
                    "pub fn j2k_tilecodec::LzwCodec::decompress_into",
                    "pub fn j2k_tilecodec::LzwCodec::expected_size",
                    "pub fn j2k_tilecodec::UncompressedCodec::decompress_into",
                    "pub fn j2k_tilecodec::UncompressedCodec::expected_size",
                    "pub fn j2k_tilecodec::ZstdCodec::decompress_into",
                    "pub fn j2k_tilecodec::ZstdCodec::expected_size",
                    "j2k_jpeg::adapter::jpeg_baseline_entropy_capacity_bytes",
                    "j2k_jpeg::adapter::jpeg_baseline_gpu_encode_batch_plan",
                    "j2k_jpeg::adapter::jpeg_baseline_gpu_encode_params",
                    "j2k_jpeg::adapter::jpeg_baseline_gpu_encode_tile_plan",
                    "j2k_jpeg::adapter::jpeg_baseline_gpu_entropy_capacity_bytes",
                    "j2k_jpeg::adapter::jpeg_baseline_sampling_for",
                    "j2k_jpeg::adapter::same_source_buffer_batch_end",
                    "j2k_jpeg::adapter::validate_jpeg_baseline_dimensions",
                    "j2k_jpeg::adapter::validate_jpeg_baseline_gpu_encode_tile",
                    "j2k_jpeg::adapter::validate_jpeg_baseline_restart_interval",
                    "j2k_jpeg::adapter::build_fast420_packet_for_decoder",
                    "j2k_jpeg::adapter::build_fast422_packet_for_decoder",
                    "j2k_jpeg::adapter::build_fast444_packet_for_decoder",
                    "j2k_jpeg::adapter::build_gray_packet_for_decoder",
                    "j2k_jpeg::adapter::build_device_plan",
                    "j2k_jpeg::adapter::DeviceCheckpoint",
                    "j2k_jpeg::adapter::DeviceComponentPlan",
                    "j2k_jpeg::adapter::DeviceDecodePlan",
                    "j2k_jpeg::adapter::summarize_device_batch",
                    "j2k_jpeg::adapter::FastPacketError",
                    "j2k_jpeg::adapter::TableKind",
                    "j2k_jpeg::adapter::JpegCanonicalHuffmanTable",
                    "j2k_jpeg::adapter::JpegEntropyCheckpointV1",
                    "j2k_jpeg::adapter::JpegFast420PacketV1",
                    "j2k_jpeg::adapter::JpegFast422PacketV1",
                    "j2k_jpeg::adapter::JpegFast444PacketV1",
                    "j2k_jpeg::adapter::JpegGrayPacketV1",
                    "j2k_jpeg::adapter::JpegHuffmanTable",
                    "j2k_jpeg::adapter::build_fast420_packet",
                    "j2k_jpeg::adapter::build_fast422_packet",
                    "j2k_jpeg::adapter::build_fast444_packet",
                    "j2k_jpeg::adapter::build_gray_packet",
                    "j2k::J2kDecoder::bytes",
                    "j2k::J2kDecoder::support_info",
                    "j2k::J2kDecoder::passthrough_candidate",
                    "j2k_jpeg::Decoder::passthrough_candidate",
                    "j2k_jpeg::Decoder::restart_index",
                    "j2k_cuda::CudaEncodeStageAccelerator::with_profile_collection",
                    "j2k_cuda::CudaHtj2kBandId",
                    "j2k_cuda::CudaHtj2kCodeBlock",
                    "j2k_cuda::CudaHtj2kDecodePlan",
                    "j2k_cuda::CudaHtj2kDecodeProfileDetail",
                    "j2k_cuda::CudaHtj2kEncodeProfileReport",
                    "j2k_cuda::CudaHtj2kIdwtStep",
                    "j2k_cuda::CudaHtj2kProfileReport",
                    "j2k_cuda::CudaHtj2kRect",
                    "j2k_cuda::CudaHtj2kEncodeProfileReport::emit",
                    "j2k_cuda::CudaHtj2kProfileReport::emit",
                    "j2k_cuda::CudaHtj2kStoreStep",
                    "j2k_cuda::CudaHtj2kSubband",
                    "j2k_cuda::CudaHtj2kTransform",
                    "j2k_cuda::J2kDecoder::decode_batch_to_device_with_session_and_profile",
                    "j2k_cuda::J2kDecoder::decode_to_device_with_session_and_profile",
                    "j2k_cuda::CudaLosslessBufferEncodeOutcome",
                    "j2k_cuda::CudaLosslessEncodeOutcome",
                    "j2k_cuda::encode_lossless_from_cuda_buffer_to_cuda_buffer_with_report",
                    "j2k_cuda::encode_lossless_from_cuda_buffer_with_report",
                    "j2k_cuda::encode_lossless_from_cuda_buffers_to_cuda_buffers_with_report",
                    "j2k_cuda::encode_lossless_from_cuda_buffers_with_report",
                    "j2k_cuda::encode_j2k_lossless_with_cuda_and_profile",
                    "compose_viewport_cpu",
                    "decode_viewport_region_cpu",
                    "decode_viewport_to_resizable_metal",
                    "j2k_jpeg_metal::Codec::submit_tile_request_to_device",
                    "j2k_jpeg_metal::Codec::inspect_rgb8_decoder_batch_metal_output",
                    "j2k_jpeg_metal::JpegMetalResidentBatchReport",
                    "j2k_jpeg_metal::MetalBatchOutputBuffer::ensure_rgb8_batch_report",
                    "j2k_jpeg_metal::MetalBatchTextureOutput::ensure_rgba8_batch_report",
                    "j2k_jpeg_cuda::Codec::decode_tile_rgb8_into_cuda_buffer_with_session",
                    "j2k_jpeg_cuda::Codec::decode_tiles_rgb8_into_cuda_buffers_with_session",
                    "j2k_jpeg_cuda::CudaJpegDecodePath",
                    "j2k_jpeg_cuda::CudaSession::owned_cuda_packet_cache_len",
                    "j2k_jpeg_cuda::CudaSession::recycle_owned_cuda_output_buffer",
                    "j2k_jpeg_cuda::CudaSession::retained_owned_cuda_output_buffers",
                    "j2k_jpeg_cuda::CudaSession::take_owned_cuda_output_buffer",
                    "j2k_jpeg_cuda::CudaSurfaceStats",
                    "j2k_jpeg_cuda::CudaSurface::stats",
                    "j2k_cuda::CudaSurfaceStats",
                    "j2k_cuda::CudaSurface::stats",
                    "impl j2k_core::traits::TileBatchDecodeDevice for j2k_jpeg_metal::Codec",
                    "impl j2k_core::traits::TileBatchDecodeManyDevice for j2k_jpeg_metal::Codec",
                    "impl j2k_core::traits::TileBatchDecodeSubmit for j2k_jpeg_metal::Codec",
                    "impl j2k_core::traits::ImageDecodeDevice<'a> for j2k_jpeg_metal::Decoder",
                    "impl j2k_core::traits::ImageDecodeSubmit<'a> for j2k_jpeg_metal::Decoder",
                    "impl j2k_core::traits::DeviceSurface for j2k_jpeg_metal::Surface",
                    "impl j2k_core::accelerator::AcceleratorSession for j2k_jpeg_metal::MetalBackendSession",
                    "pub fn j2k_jpeg_metal::Codec::decode_tiles_to_device(&mut j2k_core::context::DecoderContext",
                    "pub fn j2k_jpeg_metal::Codec::submit_tile_to_device(&mut j2k_core::context::DecoderContext",
                    "pub fn j2k_jpeg_metal::Decoder<'a>::submit_to_device",
                    "pub fn j2k_jpeg_metal::Surface::backend_kind",
                    "impl j2k_core::traits::TileBatchDecodeDevice for j2k_metal::Codec",
                    "impl j2k_core::traits::TileBatchDecodeManyDevice for j2k_metal::Codec",
                    "impl j2k_core::traits::TileBatchDecodeSubmit for j2k_metal::Codec",
                    "impl j2k_core::traits::ImageDecodeDevice<'a> for j2k_metal::J2kDecoder",
                    "impl j2k_core::traits::DeviceSurface for j2k_metal::Surface",
                    "impl j2k_core::traits::DeviceSubmission for j2k_metal::SubmittedJ2kLosslessMetal",
                    "impl j2k_core::accelerator::AcceleratorSession for j2k_metal::MetalBackendSession",
                    "pub fn j2k_metal::Codec::decode_tiles_to_device(&mut j2k_core::context::DecoderContext",
                    "pub fn j2k_metal::Codec::submit_tile_to_device(&mut j2k_core::context::DecoderContext",
                    "pub fn j2k_metal::Surface::backend_kind",
                    "pub fn j2k_metal::SubmittedJ2kLosslessMetal",
                    "impl j2k_core::traits::TileBatchDecodeDevice for j2k_jpeg_cuda::Codec",
                    "impl j2k_core::traits::TileBatchDecodeManyDevice for j2k_jpeg_cuda::Codec",
                    "impl j2k_core::traits::ImageDecodeDevice<'a> for j2k_jpeg_cuda::Decoder",
                    "impl j2k_core::traits::DeviceSurface for j2k_jpeg_cuda::Surface",
                    "impl j2k_core::accelerator::AcceleratorSession for j2k_jpeg_cuda::CudaSession",
                    "pub fn j2k_jpeg_cuda::Codec::decode_tiles_to_device(&mut j2k_core::context::DecoderContext",
                    "pub fn j2k_jpeg_cuda::Surface::backend_kind",
                    "impl j2k_core::traits::TileBatchDecodeDevice for j2k_cuda::Codec",
                    "impl j2k_core::traits::TileBatchDecodeManyDevice for j2k_cuda::Codec",
                    "impl j2k_core::traits::ImageDecodeDevice<'a> for j2k_cuda::J2kDecoder",
                    "impl j2k_core::traits::DeviceSurface for j2k_cuda::Surface",
                    "impl j2k_core::traits::DeviceSubmission for j2k_cuda::SubmittedJ2kLosslessCudaEncode",
                    "impl j2k_core::accelerator::AcceleratorSession for j2k_cuda::CudaSession",
                    "pub fn j2k_cuda::Codec::decode_tiles_to_device(&mut j2k_core::context::DecoderContext",
                    "pub fn j2k_cuda::Surface::backend_kind",
                    "pub fn j2k_cuda::SubmittedJ2kLosslessCudaEncode::wait",
                    "j2k_metal::DecodeOperation",
                    "j2k_metal::DecodeRouteReport",
                    "j2k_metal::DecodeSurfaceWithReport",
                    "j2k_metal::J2kDecoder::decode_request_to_device_with_report",
                    "j2k_metal::MetalLosslessBufferEncodeBatchOutcome",
                    "j2k_metal::MetalLosslessBufferEncodeOutcome",
                    "j2k_metal::MetalLosslessEncodeBatchStats",
                    "j2k_metal::MetalLosslessEncodeOutcome",
                    "j2k_metal::MetalLosslessEncodeStageStats",
                    "j2k_metal::encode_lossless_batch_with_report",
                    "J2kError::adapter_backend",
                    "CpuBackedImageDecode",
                    "j2k_core::CpuBackedImageDecode",
                    "j2k_core::DeviceSubmitSession",
                    "j2k_core::GpuAbi",
                    "pub fn f32::as_bytes",
                    "pub fn f64::as_bytes",
                    "pub fn i16::as_bytes",
                    "pub fn i32::as_bytes",
                    "pub fn i64::as_bytes",
                    "pub fn i8::as_bytes",
                    "pub fn u16::as_bytes",
                    "pub fn u32::as_bytes",
                    "pub fn u64::as_bytes",
                    "pub fn u8::as_bytes",
                    "pub fn [T; N]::as_bytes",
                    "j2k_core::ReadySubmission",
                    "j2k_core::traits::ReadySubmission",
                    "j2k_core::submit_ready_device",
                    "j2k_cuda_runtime::CudaBufferPool::upload_i16",
                    "j2k_cuda_runtime::CudaBufferPool::upload_i16_pinned",
                    "j2k_cuda_runtime::CudaBufferPool::take_with_trace",
                    "j2k_cuda_runtime::CudaBufferPoolTakeTrace",
                    "j2k_cuda_runtime::CudaContext::decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool",
                    "j2k_cuda_runtime::CudaContext::decode_htj2k_codeblocks_cleanup_dequantize_multi_with_resources_and_pool_timed",
                    "j2k_cuda_runtime::CudaContext::decode_htj2k_codeblocks_cleanup_multi_with_resources_and_pool_timed",
                    "j2k_cuda_runtime::CudaContext::decode_htj2k_codeblocks(&self",
                    "j2k_cuda_runtime::CudaContext::allocate_htj2k_codeblock_coefficients_with_pool",
                    "j2k_cuda_runtime::CudaContext::j2k_dequantize_htj2k_codeblocks_multi_device_with_pool",
                    "j2k_cuda_runtime::CudaContext::upload_htj2k_decode_resources_with_tables",
                    "j2k_cuda_runtime::CudaContext::upload_htj2k_decode_resources_with_tables_and_pool",
                    "j2k_cuda_runtime::CudaContext::upload_htj2k_decode_table_resources",
                    "j2k_cuda_runtime::CudaHtj2kCleanupTarget",
                    "j2k_cuda_runtime::CudaHtj2kDequantizeTarget",
                    "j2k_cuda_runtime::CudaHtj2kStatus",
                    "j2k_cuda_runtime::CudaHtj2kDecodeStageTimings",
                    "j2k_cuda_runtime::CudaHtj2kDecodeOutput",
                    "j2k_cuda_runtime::CudaPooledHtj2kDecodeOutput",
                    "j2k_cuda_runtime::CudaContext::encode_htj2k_codeblocks(&self",
                    "j2k_cuda_runtime::CudaContext::encode_htj2k_codeblocks_resident(&self",
                    "j2k_cuda_runtime::CudaContext::encode_htj2k_codeblock_regions_resident(&self",
                    "j2k_cuda_runtime::CudaContext::encode_htj2k_codeblock_regions_resident_with_resources_and_pool",
                    "j2k_cuda_runtime::CudaContext::encode_htj2k_codeblocks_multi_resident_compact_with_resources_and_pool",
                    "j2k_cuda_runtime::CudaContext::encode_htj2k_codeblocks_multi_resident_with_resources_and_pool",
                    "j2k_cuda_runtime::CudaContext::encode_htj2k_codeblocks_resident_with_resources_and_pool",
                    "j2k_cuda_runtime::CudaContext::encode_htj2k_codeblocks_with_resources",
                    "j2k_cuda_runtime::CudaContext::upload_htj2k_encode_resources",
                    "j2k_cuda_runtime::CudaHtj2kEncodeResidentTarget",
                    "j2k_cuda_runtime::CudaHtj2kEncodeStatus",
                    "j2k_cuda_runtime::CudaHtj2kEncodeStageTimings",
                    "j2k_cuda_runtime::CudaHtj2kEncodedCodeBlock",
                    "j2k_cuda_runtime::CudaHtj2kEncodedCodeBlocks",
                    "j2k_cuda_runtime::CudaHtj2kCompactEncodedCodeBlock",
                    "j2k_cuda_runtime::CudaHtj2kCompactEncodedCodeBlocks",
                    "j2k_cuda_runtime::CudaHtj2kDecodeResources",
                    "j2k_cuda_runtime::CudaHtj2kDecodeTableResources",
                    "j2k_cuda_runtime::CudaHtj2kEncodeResources",
                    "j2k_cuda_runtime::CudaContext::j2k_inverse_dwt_batch_device_enqueue_with_pool",
                    "j2k_cuda_runtime::CudaContext::j2k_inverse_dwt_batch_sequence_enqueue_with_pool",
                    "j2k_cuda_runtime::CudaContext::j2k_dequantize_queued_htj2k_cleanup_with_pool",
                    "j2k_cuda_runtime::CudaContext::packetize_htj2k_cleanup_packets_with_tag_state",
                    "j2k_cuda_runtime::CudaContext::create_event",
                    "j2k_cuda_runtime::CudaContext::create_stream",
                    "j2k_cuda_runtime::CudaContext::copy_with_kernel",
                    "j2k_cuda_runtime::CudaContext::copy_device_to_device_with_kernel",
                    "j2k_cuda_runtime::CudaContext::copy_with_cuda_oxide_kernel",
                    "j2k_cuda_runtime::CudaContext::decode_jpeg_420_rgb8_owned",
                    "j2k_cuda_runtime::CudaContext::decode_jpeg_420_rgb8_owned_into",
                    "j2k_cuda_runtime::CudaContext::decode_jpeg_rgb8_owned",
                    "j2k_cuda_runtime::CudaContext::decode_jpeg_rgb8_owned_into",
                    "j2k_cuda_runtime::CudaContext::encode_jpeg_baseline_entropy",
                    "j2k_cuda_runtime::CudaContext::encode_jpeg_baseline_entropy_batch",
                    "j2k_cuda_runtime::CudaDwt97BatchRequest",
                    "j2k_cuda_runtime::CudaHtj2k97CodeblockBatchRequest",
                    "j2k_cuda_runtime::CudaContext::j2k_transcode_reversible_dwt53",
                    "j2k_cuda_runtime::CudaContext::j2k_transcode_dwt97",
                    "j2k_cuda_runtime::CudaContext::j2k_transcode_dwt97_batch_with_pool",
                    "j2k_cuda_runtime::CudaContext::j2k_transcode_htj2k97_codeblock_batch_resident_with_pool",
                    "j2k_cuda_runtime::CudaContext::j2k_transcode_htj2k97_codeblock_i16_batch_resident_with_pool",
                    "j2k_cuda_runtime::CudaContext::j2k_transcode_htj2k97_codeblock_batch_with_pool",
                    "j2k_cuda_runtime::CudaDwt97BatchGeometry",
                    "j2k_cuda_runtime::CudaDwt97BatchWithPoolRequest",
                    "j2k_cuda_runtime::CudaHtj2k97CodeblockBands",
                    "j2k_cuda_runtime::CudaHtj2k97CodeblockBatchWithPoolRequest",
                    "j2k_cuda_runtime::CudaHtj2k97DeviceCodeblockBands",
                    "j2k_cuda_runtime::CudaHtj2k97I16CodeblockBatchWithPoolRequest",
                    "j2k_cuda_runtime::CudaHtj2k97QuantizeParams",
                    "j2k_cuda_runtime::CudaTranscodeDwt97Bands",
                    "j2k_cuda_runtime::CudaTranscodeReversible53Bands",
                    "j2k_cuda_runtime::CudaQueuedExecution",
                    "j2k_cuda_runtime::CudaQueuedHtj2kCleanup",
                    "j2k_cuda_runtime::CudaHtj2kPacketizationBlock",
                    "j2k_cuda_runtime::CudaHtj2kPacketizationPacket",
                    "j2k_cuda_runtime::CudaHtj2kPacketizationStageTimings",
                    "j2k_cuda_runtime::CudaHtj2kPacketizationStatus",
                    "j2k_cuda_runtime::CudaHtj2kPacketizationSubband",
                    "j2k_cuda_runtime::CudaHtj2kPacketizationSubbandTagState",
                    "j2k_cuda_runtime::CudaHtj2kPacketizationTagNodeState",
                    "j2k_cuda_runtime::CudaHtj2kPacketizedTile",
                    "j2k_cuda_runtime::CudaContext::upload_htj2k_decode_resources(&self",
                    "j2k_cuda_runtime::CudaContext::decode_htj2k_codeblocks_cleanup_multi_with_resources_and_pool(&self",
                    "j2k_cuda_runtime::CudaContext::pinned_host_buffer",
                    "j2k_cuda_runtime::CudaContext::decode_htj2k_codeblocks_cleanup_with_resources_untimed_and_pool",
                    "j2k_cuda_runtime::CudaContext::decode_htj2k_codeblocks_untimed",
                    "j2k_cuda_runtime::CudaContext::decode_htj2k_codeblocks_with_resources_untimed",
                    "j2k_cuda_runtime::CudaContext::decode_htj2k_codeblocks_with_resources_untimed_and_pool",
                    "j2k_cuda_runtime::CudaContext::decode_htj2k_codeblocks_with_resources(&self",
                    "j2k_cuda_runtime::CudaContext::decode_htj2k_codeblocks_with_resources_and_pool(&self",
                    "j2k_cuda_runtime::CudaContext::decode_htj2k_codeblocks_cleanup_with_resources_and_pool(&self",
                    "j2k_cuda_runtime::CudaContext::encode_htj2k_codeblock(",
                    "j2k_cuda_runtime::CudaContext::encode_htj2k_codeblock_with_resources(",
                    "j2k_cuda_runtime::CudaContext::encode_htj2k_codeblock_regions_resident_with_resources(&self",
                    "j2k_cuda_runtime::CudaContext::encode_htj2k_codeblocks_resident_with_resources(&self",
                    "j2k_cuda_runtime::CudaHtj2kCodeBlockJob",
                    "j2k_cuda_runtime::CudaHtj2kDecodeTables",
                    "j2k_cuda_runtime::CudaHtj2kEncodeCodeBlockJob",
                    "j2k_cuda_runtime::CudaHtj2kEncodeCodeBlockRegionJob",
                    "j2k_cuda_runtime::CudaHtj2kEncodeTables",
                    "j2k_cuda_runtime::CudaContext::j2k_dequantize_htj2k_codeblocks_multi_device(&self",
                    "j2k_cuda_runtime::CudaContext::j2k_dequantize_htj2k_codeblocks_multi_device_untimed_with_pool",
                    "j2k_cuda_runtime::CudaContext::j2k_inverse_dwt_single_device_untimed(&self",
                    "j2k_cuda_runtime::CudaContext::j2k_inverse_dwt_batch_device_untimed_with_pool",
                    "j2k_cuda_runtime::CudaContext::packetize_htj2k_cleanup_packets(&self",
                    "j2k_cuda_runtime::CudaContext::preload_kernel_module",
                    "j2k_cuda_runtime::CudaContext::time_default_stream_named_us",
                    "j2k_cuda_runtime::CudaContext::time_default_stream_us",
                    "j2k_cuda_runtime::CudaContext::upload_i32_pinned",
                    "j2k_cuda_runtime::CudaContext::with_nvtx_range",
                    "j2k_cuda_runtime::CudaContext::diagnose_jpeg_420_entropy_self_sync",
                    "j2k_cuda_runtime::CudaContext::j2k_deinterleave_strided_to_f32_resident",
                    "j2k_cuda_runtime::CudaContext::j2k_deinterleave_to_f32",
                    "j2k_cuda_runtime::CudaContext::j2k_deinterleave_to_f32_resident",
                    "j2k_cuda_runtime::CudaContext::j2k_forward_dwt53",
                    "j2k_cuda_runtime::CudaContext::j2k_forward_dwt53_resident_component",
                    "j2k_cuda_runtime::CudaContext::j2k_forward_dwt97",
                    "j2k_cuda_runtime::CudaContext::j2k_forward_dwt97_resident_component",
                    "j2k_cuda_runtime::CudaContext::j2k_forward_ict",
                    "j2k_cuda_runtime::CudaContext::j2k_forward_ict_resident",
                    "j2k_cuda_runtime::CudaContext::j2k_forward_rct",
                    "j2k_cuda_runtime::CudaContext::j2k_forward_rct_resident",
                    "j2k_cuda_runtime::CudaContext::j2k_inverse_dwt_batch_device_with_pool",
                    "j2k_cuda_runtime::CudaContext::j2k_inverse_dwt_single_device(&self",
                    "j2k_cuda_runtime::CudaContext::j2k_inverse_dwt_single_device_with_pool",
                    "j2k_cuda_runtime::CudaContext::j2k_inverse_mct_device",
                    "j2k_cuda_runtime::CudaContext::j2k_quantize_subband",
                    "j2k_cuda_runtime::CudaContext::j2k_quantize_subband_region_resident",
                    "j2k_cuda_runtime::CudaContext::j2k_quantize_subband_resident",
                    "j2k_cuda_runtime::CudaContext::j2k_store_gray16_device",
                    "j2k_cuda_runtime::CudaContext::j2k_store_gray8_device",
                    "j2k_cuda_runtime::CudaContext::j2k_store_rgb16_device",
                    "j2k_cuda_runtime::CudaContext::j2k_store_rgb16_mct_device",
                    "j2k_cuda_runtime::CudaContext::j2k_store_rgb8_device",
                    "j2k_cuda_runtime::CudaContext::j2k_store_rgb8_mct_batch_contiguous_device",
                    "j2k_cuda_runtime::CudaContext::j2k_store_rgb8_mct_batch_device",
                    "j2k_cuda_runtime::CudaContext::j2k_store_rgb8_mct_device",
                    "j2k_cuda_runtime::CudaDwt53LevelShape",
                    "j2k_cuda_runtime::CudaDwt53Output",
                    "j2k_cuda_runtime::CudaDwt97BatchStageTimings",
                    "j2k_cuda_runtime::CudaDwt97Output",
                    "j2k_cuda_runtime::CudaEvent",
                    "j2k_cuda_runtime::CudaJ2kDeinterleavedComponents",
                    "j2k_cuda_runtime::CudaJ2kIdwtJob",
                    "j2k_cuda_runtime::CudaJ2kIdwtTarget",
                    "j2k_cuda_runtime::CudaJ2kInverseMctJob",
                    "j2k_cuda_runtime::CudaJ2kQuantizeJob",
                    "j2k_cuda_runtime::CudaJ2kQuantizeSubbandRegionJob",
                    "j2k_cuda_runtime::CudaJ2kQuantizedSubband",
                    "j2k_cuda_runtime::CudaJ2kRect",
                    "j2k_cuda_runtime::CudaJ2kResidentComponents",
                    "j2k_cuda_runtime::CudaJ2kResidentQuantizedSubband",
                    "j2k_cuda_runtime::CudaJ2kStoreGray16Job",
                    "j2k_cuda_runtime::CudaJ2kStoreGray8Job",
                    "j2k_cuda_runtime::CudaJ2kStoreRgb16Job",
                    "j2k_cuda_runtime::CudaJ2kStoreRgb16MctJob",
                    "j2k_cuda_runtime::CudaJ2kStoreRgb8Job",
                    "j2k_cuda_runtime::CudaJ2kStoreRgb8MctJob",
                    "j2k_cuda_runtime::CudaJ2kStoreRgb8MctTarget",
                    "j2k_cuda_runtime::CudaJ2kStridedInterleavedPixels",
                    "j2k_cuda_runtime::CudaPinnedHostBuffer",
                    "j2k_cuda_runtime::CudaResidentDwt53Output",
                    "j2k_cuda_runtime::CudaResidentDwt97Output",
                    "j2k_cuda_runtime::CudaDeviceBufferRange",
                    "j2k_cuda_runtime::CudaDeviceBuffer::typed_view",
                    "j2k_cuda_runtime::CudaDeviceBufferView",
                    "j2k_cuda_runtime::CudaDeviceBufferViewMut",
                    "j2k_cuda_runtime::CudaKernelBatchOutput",
                    "j2k_cuda_runtime::CudaKernelContiguousBatchOutput",
                    "j2k_cuda_runtime::CudaKernelOutput",
                    "j2k_cuda_runtime::CudaPooledKernelOutput",
                    "j2k_cuda_runtime::CudaKernelModule",
                    "j2k_cuda_runtime::CudaKernelName",
                    "j2k_cuda_runtime::CudaJpeg420Rgb8DecodePlan",
                    "j2k_cuda_runtime::CudaJpegBaselineEncodeFormat",
                    "j2k_cuda_runtime::CudaJpegBaselineEncodeHuffmanTable",
                    "j2k_cuda_runtime::CudaJpegBaselineEncodeParams",
                    "j2k_cuda_runtime::CudaJpegBaselineEntropyEncodeBatchJob",
                    "j2k_cuda_runtime::CudaJpegBaselineEntropyEncodeJob",
                    "j2k_cuda_runtime::CudaJpegChunkedEntropyConfig",
                    "j2k_cuda_runtime::CudaJpegChunkedEntropyPlan",
                    "j2k_cuda_runtime::CudaJpegChunkedEntropyReport",
                    "j2k_cuda_runtime::CudaJpegEntropyCheckpoint",
                    "j2k_cuda_runtime::CudaJpegEntropyOverflowState",
                    "j2k_cuda_runtime::CudaJpegEntropySyncState",
                    "j2k_cuda_runtime::CudaJpegHuffmanTable",
                    "j2k_cuda_runtime::CudaJpegRgb8DecodePlan",
                    "j2k_cuda_runtime::CudaJpegRgb8Sampling",
                    "j2k_cuda_runtime::CudaStream",
                    "j2k_jpeg_cuda::Codec::diagnose_tile_rgb8_chunked_entropy_with_session",
                    "j2k_cuda::CudaEncodeStageAccelerator::deinterleave_attempts",
                    "j2k_cuda::CudaEncodeStageAccelerator::forward_dwt53_attempts",
                    "j2k_cuda::CudaEncodeStageAccelerator::forward_dwt97_attempts",
                    "j2k_cuda::CudaEncodeStageAccelerator::forward_ict_attempts",
                    "j2k_cuda::CudaEncodeStageAccelerator::forward_rct_attempts",
                    "j2k_cuda::CudaEncodeStageAccelerator::ht_code_block_attempts",
                    "j2k_cuda::CudaEncodeStageAccelerator::ht_subband_attempts",
                    "j2k_cuda::CudaEncodeStageAccelerator::htj2k_tile_attempts",
                    "j2k_cuda::CudaEncodeStageAccelerator::packetization_attempts",
                    "j2k_cuda::CudaEncodeStageAccelerator::quantize_subband_attempts",
                    "j2k_cuda::CudaEncodeStageAccelerator::tier1_code_block_attempts",
                    "j2k_cuda::CudaEncodeStageAccelerator::deinterleave_dispatches",
                    "j2k_cuda::CudaEncodeStageAccelerator::forward_dwt53_dispatches",
                    "j2k_cuda::CudaEncodeStageAccelerator::forward_dwt97_dispatches",
                    "j2k_cuda::CudaEncodeStageAccelerator::forward_ict_dispatches",
                    "j2k_cuda::CudaEncodeStageAccelerator::forward_rct_dispatches",
                    "j2k_cuda::CudaEncodeStageAccelerator::ht_code_block_dispatches",
                    "j2k_cuda::CudaEncodeStageAccelerator::ht_subband_dispatches",
                    "j2k_cuda::CudaEncodeStageAccelerator::htj2k_tile_dispatches",
                    "j2k_cuda::CudaEncodeStageAccelerator::packetization_dispatches",
                    "j2k_cuda::CudaEncodeStageAccelerator::quantize_subband_dispatches",
                    "j2k_cuda::CudaEncodeStageAccelerator::tier1_code_block_dispatches",
                    "j2k_cuda::J2kDecoder::build_cuda_htj2k_grayscale_plan_with_profile",
                    "j2k_cuda::J2kDecoder::build_cuda_htj2k_grayscale_region_plan_with_profile",
                    "j2k_cuda::J2kDecoder::build_cuda_htj2k_grayscale_scaled_plan_with_profile",
                    "j2k_cuda::J2kDecoder::build_cuda_htj2k_grayscale_region_scaled_plan_with_profile",
                    "j2k_cuda::Surface::download_into_profiled",
                    "j2k_metal::MetalEncodeStageAccelerator::deinterleave_attempts",
                    "j2k_metal::MetalEncodeStageAccelerator::forward_dwt53_attempts",
                    "j2k_metal::MetalEncodeStageAccelerator::forward_dwt97_attempts",
                    "j2k_metal::MetalEncodeStageAccelerator::forward_ict_attempts",
                    "j2k_metal::MetalEncodeStageAccelerator::forward_rct_attempts",
                    "j2k_metal::MetalEncodeStageAccelerator::ht_code_block_attempts",
                    "j2k_metal::MetalEncodeStageAccelerator::packetization_attempts",
                    "j2k_metal::MetalEncodeStageAccelerator::quantize_subband_attempts",
                    "j2k_metal::MetalEncodeStageAccelerator::tier1_code_block_attempts",
                    "j2k_metal::MetalEncodeStageAccelerator::deinterleave_dispatches",
                    "j2k_metal::MetalEncodeStageAccelerator::forward_dwt53_dispatches",
                    "j2k_metal::MetalEncodeStageAccelerator::forward_dwt97_dispatches",
                    "j2k_metal::MetalEncodeStageAccelerator::forward_ict_dispatches",
                    "j2k_metal::MetalEncodeStageAccelerator::forward_rct_dispatches",
                    "j2k_metal::MetalEncodeStageAccelerator::ht_code_block_dispatches",
                    "j2k_metal::MetalEncodeStageAccelerator::packetization_dispatches",
                    "j2k_metal::MetalEncodeStageAccelerator::quantize_subband_dispatches",
                    "j2k_metal::MetalEncodeStageAccelerator::tier1_code_block_dispatches",
                ]),
        ],
    );
}
