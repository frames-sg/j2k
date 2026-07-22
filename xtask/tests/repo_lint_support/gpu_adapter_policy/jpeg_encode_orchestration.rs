// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, read_source_files, repo_root, PatternCheck};

#[test]
fn jpeg_gpu_encode_host_orchestration_uses_shared_adapter_helper() {
    let root = repo_root();
    let shared = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/adapter/baseline_encode.rs",
            "crates/j2k-jpeg/src/adapter/baseline_encode/frame.rs",
            "crates/j2k-jpeg/src/adapter/baseline_encode/orchestrate.rs",
            "crates/j2k-jpeg/src/adapter/baseline_encode/orchestrate/batch.rs",
            "crates/j2k-jpeg/src/adapter/baseline_encode/planning.rs",
            "crates/j2k-jpeg/src/adapter/baseline_encode/planning/batch.rs",
            "crates/j2k-jpeg/src/adapter/baseline_encode/tables.rs",
            "crates/j2k-jpeg/src/adapter/baseline_encode/types.rs",
            "crates/j2k-jpeg/src/adapter/baseline_encode/validation.rs",
        ],
    );
    let cuda_encode = fs::read_to_string(root.join("crates/j2k-jpeg-cuda/src/encode.rs"))
        .expect("read JPEG CUDA encode host");
    let cuda_encode_error =
        fs::read_to_string(root.join("crates/j2k-jpeg-cuda/src/encode/error.rs"))
            .expect("read JPEG CUDA encode error mapping");
    let metal_encode = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/encode.rs"))
        .expect("read JPEG Metal encode host");
    let metal_adapter =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/encode/adapter.rs"))
            .expect("read JPEG Metal encode adapter");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-jpeg shared GPU encode helper", &shared).required(&[
            "pub struct JpegBaselineGpuEncodeTile",
            "pub struct JpegBaselineGpuEncodeParams",
            "pub trait JpegBaselineGpuEncodeHostAdapter",
            "pub enum JpegBaselineGpuEncodeError",
            "fn validate_jpeg_baseline_gpu_encode_tile",
            "fn jpeg_baseline_gpu_encode_params",
            "fn jpeg_baseline_gpu_entropy_capacity_bytes",
            "fn same_source_buffer_batch_end",
            "pub fn encode_jpeg_baseline_gpu_tile",
            "pub fn encode_jpeg_baseline_gpu_batch",
            "pub fn encode_jpeg_baseline_gpu_tile_with_external_live",
            "pub fn encode_jpeg_baseline_gpu_batch_with_external_live",
            "while start < tiles.len()",
            "assemble_jpeg_baseline_frame(",
        ]),
    ]);

    let forbidden_host_orchestration = [
        "baseline_encode_tables",
        "assemble_jpeg_baseline_frame",
        "jpeg_baseline_gpu_encode_tile_plan",
        "jpeg_baseline_gpu_encode_batch_plan",
        "same_source_buffer_batch_end",
        "while start < tiles.len()",
        "validate_jpeg_baseline_dimensions",
        "jpeg_baseline_entropy_capacity_bytes",
        "checked_mul(bytes_per_pixel)",
        "let mcu_width =",
        "let mcu_height =",
        "JpegSubsampling",
    ];
    let metal_orchestration = format!("{metal_encode}\n{metal_adapter}");
    assert_pattern_checks(&[
        PatternCheck::new("crates/j2k-jpeg-cuda/src/encode.rs", &cuda_encode)
            .required(&[
                "mod error;",
                "JpegBaselineGpuEncodeHostAdapter",
                "encode_jpeg_baseline_gpu_tile_with_external_live(",
                "encode_jpeg_baseline_gpu_batch_with_external_live(",
                "external_live_bytes",
                "fn encode_tile_entropy(",
                "fn encode_batch_entropy(",
                "cuda_gpu_encode_error(error)",
            ])
            .forbidden(&forbidden_host_orchestration),
        PatternCheck::new("JPEG Metal encode API shell", &metal_encode)
            .required(&[
                "mod adapter;",
                "struct MetalJpegBaselineEncodeAdapter",
                "encode_jpeg_baseline_gpu_tile(tile, options, &mut adapter)",
                "encode_jpeg_baseline_gpu_batch(tiles, options, &mut adapter)",
            ])
            .forbidden(&[
                "impl<'tile> JpegBaselineGpuEncodeHostAdapter",
                "fn encode_tile_entropy(",
                "fn encode_batch_entropy(",
            ]),
        PatternCheck::new("JPEG Metal encode adapter", &metal_adapter).required(&[
            "impl<'tile> JpegBaselineGpuEncodeHostAdapter",
            "fn encode_tile_entropy(",
            "fn encode_batch_entropy(",
            "compute::encode_jpeg_baseline_entropy_with_session(",
            "compute::encode_jpeg_baseline_entropy_batch_with_session(",
        ]),
        PatternCheck::new("JPEG Metal encode host orchestration", &metal_orchestration)
            .forbidden(&forbidden_host_orchestration),
    ]);
    assert!(
        cuda_encode.lines().count() < 310
            && cuda_encode_error.lines().count() < 100
            && metal_encode.lines().count() < 200
            && metal_adapter.lines().count() < 250,
        "JPEG GPU encode adapters must stay below the post-driver line ratchets"
    );
}

#[test]
fn metal_backend_session_lifecycle_lives_in_support_crate() {
    let root = repo_root();
    let support = fs::read_to_string(root.join("crates/j2k-metal-support/src/runtime.rs"))
        .expect("read Metal support runtime module");
    let jpeg_metal = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/lib.rs"))
        .expect("read JPEG Metal lib");
    let jpeg_metal_session = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/session.rs"))
        .expect("read JPEG Metal session module");
    let j2k_metal_session = fs::read_to_string(root.join("crates/j2k-metal/src/session.rs"))
        .expect("read J2K Metal session module");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal-support session lifecycle helper", &support).required(&[
            "pub struct MetalRuntimeSession<R, E>",
            "runtime: Arc<OnceLock<Result<R, E>>>",
            "pub fn system_default() -> Result<Self, MetalSupportError>",
            "pub fn runtime_initialized(&self) -> bool",
            "pub fn get_or_init_runtime",
        ]),
        PatternCheck::new("j2k-jpeg-metal public session re-exports", &jpeg_metal)
            .required(&["pub use session::{MetalBackendSession, MetalSession};"])
            .forbidden(&["pub struct MetalBackendSession", "pub struct MetalSession"]),
        PatternCheck::new(
            "j2k-jpeg-metal session module public types",
            &jpeg_metal_session,
        )
        .required(&["pub struct MetalBackendSession", "pub struct MetalSession"]),
    ]);

    for (relative, source) in [
        ("crates/j2k-jpeg-metal/src/session.rs", &jpeg_metal_session),
        ("crates/j2k-metal/src/session.rs", &j2k_metal_session),
    ] {
        assert_pattern_checks(&[PatternCheck::new(relative, source)
            .required(&["MetalRuntimeSession<", "runtime_session:"])
            .forbidden(&[
                "runtime: Arc<OnceLock<Result",
                "system_default_device()\n            .map(Self::new)",
            ])]);
    }
}
