// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path};

use crate::repo_lint_support::{
    assert_file_pattern_checks, assert_pattern_checks, repo_root, rust_sources, FilePatternCheck,
    PatternCheck,
};

#[test]
fn cuda_htj2k_compact_jobs_use_shared_planner() {
    let root = repo_root();
    let htj2k_encode = fs::read_to_string(
        root.join("crates/j2k-cuda-runtime/src/htj2k_encode/planning/compact.rs"),
    )
    .expect("read CUDA runtime HTJ2K compact planning module");
    let runtime_tests = fs::read_to_string(root.join("crates/j2k-cuda-runtime/src/tests.rs"))
        .expect("read CUDA runtime tests");

    assert_pattern_checks(&[PatternCheck::new(
        "CUDA HTJ2K compact planner implementation",
        &htj2k_encode,
    )
    .required(&[
        "trait Htj2kCompactPlanJob",
        "impl Htj2kCompactPlanJob for CudaHtj2kEncodeKernelJob",
        "impl Htj2kCompactPlanJob for CudaHtj2kEncodeMultiInputKernelJob",
        "fn htj2k_encode_compact_jobs_impl<J: Htj2kCompactPlanJob>",
        "htj2k_encode_compact_jobs_impl(statuses, kernel_jobs, host_budget)",
    ])]);
    assert_eq!(
        htj2k_encode.matches("let source_end =").count(),
        1,
        "compact output-range validation must live in one planner"
    );
    assert_pattern_checks(&[
        PatternCheck::new("CUDA HTJ2K compact planner tests", &runtime_tests).required(&[
            "assert_compact_jobs_match_for_single_and_multi_input",
            "htj2k_encode_compact_jobs_accept_empty_batches",
            "htj2k_encode_compact_jobs_accept_exact_capacity_payloads",
            "htj2k_encode_compact_jobs_reject_payloads_larger_than_capacity",
            "htj2k_encode_compact_jobs_pack_actual_payloads",
        ]),
    ]);
}

#[test]
fn cuda_oxide_simt_helpers_use_shared_prelude() {
    let root = repo_root();
    let prelude =
        fs::read_to_string(root.join("crates/j2k-cuda-runtime/src/cuda_oxide_simt_prelude.rs"))
            .expect("read CUDA Oxide SIMT prelude");
    let build_script = fs::read_to_string(root.join("crates/j2k-cuda-runtime/build.rs"))
        .expect("read CUDA runtime build script");
    let unsafe_audit =
        fs::read_to_string(root.join("docs/unsafe-audit.md")).expect("read unsafe audit");

    assert_pattern_checks(&[
        PatternCheck::new("CUDA Oxide SIMT prelude", &prelude).required(&[
            "fn simt_load<T: Copy>",
            "fn simt_store<T>",
            "fn simt_mut_ptr_at<T>",
            "SAFETY: CUDA-Oxide kernels pass validated device buffers",
        ]),
    ]);
    assert_pattern_checks(&[
        PatternCheck::new("CUDA runtime SIMT prelude build dependency", &build_script).required(&[
            "DEP_J2K_CODEC_MATH_MANIFEST_DIR",
            "\"src/classic.rs\"",
            "codec_math_crate_path.join(relative).display()",
            "cargo:rerun-if-changed=src/cuda_oxide_simt_prelude.rs",
            "stage_cuda_oxide_shared_prelude(context.out_dir);",
            "out_dir.join(\"cuda_oxide_simt_prelude.rs\")",
        ]),
        PatternCheck::new(
            "unsafe audit CUDA Oxide SIMT prelude invariants",
            &unsafe_audit,
        )
        .required(&[
            "cuda_oxide_simt_prelude.rs",
            "Shared cuda-oxide SIMT pointer prelude",
        ]),
    ]);

    let mut simt_sources = rust_sources(&root.join("crates/j2k-cuda-runtime/src"))
        .into_iter()
        .filter(|path| {
            path.ends_with(Path::new("simt/src/main.rs"))
                && path.components().any(|component| {
                    component
                        .as_os_str()
                        .to_string_lossy()
                        .starts_with("cuda_oxide_")
                })
        })
        .collect::<Vec<_>>();
    simt_sources.sort();
    assert!(
        simt_sources.len() >= 10,
        "expected all CUDA Oxide SIMT kernel sources to be discovered"
    );

    for path in simt_sources {
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        let relative = path.strip_prefix(root).unwrap_or(&path).display();
        let relative_name = relative.to_string();
        assert_pattern_checks(&[PatternCheck::new(&relative_name, &source)
            .required(&["include!(\"../../../cuda_oxide_simt_prelude.rs\");"])]);

        if source.contains("fn load_")
            || source.contains("fn store_")
            || source.contains("fn offset_")
            || source.contains("pub unsafe fn j2k_copy_u8")
        {
            assert!(
                source.contains("simt_load")
                    || source.contains("simt_store")
                    || source.contains("simt_mut_ptr_at"),
                "{relative} helper wrappers must delegate to the shared SIMT prelude"
            );
        }

        assert_pattern_checks(&[PatternCheck::new(&relative_name, &source).forbidden(&[
            "unsafe { *ptr.add",
            "unsafe { ptr.add",
            "unsafe { *ptr }",
            "*dst.add(",
            "*src.add(",
            "*decoded_data.add(",
        ])]);
    }
}

#[test]
fn backend_surfaces_use_core_metadata_and_residency() {
    let root = repo_root();
    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("crates/j2k-core/src/accelerator.rs")
                .named("j2k-core accelerator contracts")
                .required(&[
                    "pub struct SurfaceMetadata",
                    "pub enum SurfaceResidency",
                    "pub pitch_bytes: usize",
                    "pub byte_offset: usize",
                ]),
            FilePatternCheck::new("crates/j2k-jpeg-metal/src/lib.rs")
                .named("JPEG Metal lib module")
                .required(&["mod surface;", "pub use surface::{"])
                .forbidden(&[
                    "pub struct Surface",
                    "pub struct MetalBatchOutputBuffer",
                    "pub struct MetalBatchTextureOutput",
                ]),
            FilePatternCheck::new("crates/j2k-jpeg-metal/src/surface.rs")
                .named("JPEG Metal surface facade")
                .required(&[
                    "pub struct Surface",
                    "mod resident_tile;",
                    "mod batch_buffer;",
                    "mod batch_texture;",
                    "mod texture_tile;",
                    "pub use resident_tile::ResidentPrivateJpegTile;",
                    "pub use batch_buffer::MetalBatchOutputBuffer;",
                    "pub use batch_texture::MetalBatchTextureOutput;",
                    "pub use texture_tile::MetalTextureTile;",
                ])
                .forbidden(&[
                    "pub struct MetalBatchOutputBuffer",
                    "pub struct MetalBatchTextureOutput",
                    "pub struct MetalTextureTile",
                    "pub struct ResidentPrivateJpegTile",
                ]),
            FilePatternCheck::new("crates/j2k-jpeg-metal/src/surface/resident_tile.rs")
                .required(&["pub struct ResidentPrivateJpegTile"]),
            FilePatternCheck::new("crates/j2k-jpeg-metal/src/surface/batch_buffer.rs")
                .required(&["pub struct MetalBatchOutputBuffer"]),
            FilePatternCheck::new("crates/j2k-jpeg-metal/src/surface/batch_texture.rs")
                .required(&["pub struct MetalBatchTextureOutput"]),
            FilePatternCheck::new("crates/j2k-jpeg-metal/src/surface/texture_tile.rs")
                .required(&["pub struct MetalTextureTile"]),
        ],
    );

    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("crates/j2k-cuda/src/surface.rs")
                .required(&["SurfaceMetadata", "fn metadata(&self)"])
                .forbidden(&["pub enum SurfaceResidency"]),
            FilePatternCheck::new("crates/j2k-jpeg-cuda/src/surface.rs")
                .required(&["SurfaceMetadata", "fn metadata(&self)"])
                .forbidden(&["pub enum SurfaceResidency"]),
            FilePatternCheck::new("crates/j2k-metal/src/surface.rs")
                .required(&["SurfaceMetadata", "fn metadata(&self)"])
                .forbidden(&["pub enum SurfaceResidency"]),
            FilePatternCheck::new("crates/j2k-jpeg-metal/src/surface.rs")
                .required(&["SurfaceMetadata", "fn metadata(&self)"])
                .forbidden(&["pub enum SurfaceResidency"]),
            FilePatternCheck::new("crates/j2k-cuda/src/surface.rs")
                .required(&["pub use j2k_core::SurfaceResidency;"]),
            FilePatternCheck::new("crates/j2k-metal/src/lib.rs")
                .required(&["pub use j2k_core::SurfaceResidency;"]),
            FilePatternCheck::new("crates/j2k-jpeg-metal/src/lib.rs")
                .required(&["pub use j2k_core::SurfaceResidency;"]),
        ],
    );
}

#[test]
fn cuda_encode_api_and_resident_types_live_in_focused_modules() {
    let root = repo_root();
    let encode = fs::read_to_string(root.join("crates/j2k-cuda/src/encode.rs"))
        .expect("read CUDA encode module");
    let api = fs::read_to_string(root.join("crates/j2k-cuda/src/encode/api.rs"))
        .expect("read CUDA encode API module");
    let resident = fs::read_to_string(root.join("crates/j2k-cuda/src/encode/resident.rs"))
        .expect("read CUDA encode resident module");
    let stage = fs::read_to_string(root.join("crates/j2k-cuda/src/encode/stage.rs"))
        .expect("read CUDA encode stage module");

    let api_helpers = [
        "pub fn encode_j2k_lossless_with_cuda(",
        "pub fn encode_j2k_lossless_with_cuda_and_profile(",
        "pub(super) fn strict_cuda_encode_options",
        "pub(super) fn reject_non_cuda_encode_backend",
    ];
    let resident_types = [
        "pub struct CudaLosslessEncodeTile",
        "pub struct CudaLosslessEncodeResidency",
        "pub struct CudaLosslessEncodeOutcome",
        "pub struct CudaResidentCodestreamBuffer",
        "pub struct CudaEncodedJ2k",
        "pub struct CudaLosslessBufferEncodeOutcome",
        "pub struct SubmittedJ2kLosslessCudaEncode",
        "pub struct SubmittedJ2kLosslessCudaEncodeBatch",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("CUDA encode API module shell", &encode)
            .required(&[
                "mod api;",
                "pub use self::api::{encode_j2k_lossless_with_cuda",
                "strict_cuda_encode_options",
            ])
            .forbidden(&api_helpers),
        PatternCheck::new("CUDA encode API helper ownership", &api).required(&api_helpers),
        PatternCheck::new("CUDA encode resident module shell", &encode)
            .required(&[
                "mod resident;",
                "pub use self::resident",
                "CudaLosslessEncodeTile",
            ])
            .forbidden(&resident_types),
        PatternCheck::new("CUDA encode resident type ownership", &resident)
            .required(&resident_types),
    ]);
    assert!(
        encode.lines().count() < 3_000,
        "j2k-cuda encode.rs must stay below the post-split god-file threshold"
    );
    assert!(
        stage.lines().count() < 1_200,
        "j2k-cuda encode/stage.rs must stay below its accepted cohesive-adapter threshold"
    );
    let stage_items = [
        "pub struct CudaEncodeStageAccelerator",
        "pub struct CudaEncodeStageTimings",
        "impl J2kEncodeStageAccelerator for CudaEncodeStageAccelerator",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("CUDA encode focused module shell", &encode).required(&[
            "mod packetization;",
            "mod stage;",
            "pub use self::stage::{CudaEncodeStageAccelerator",
            "mod htj2k;",
        ]),
        PatternCheck::new("CUDA encode stage exclusion", &encode).forbidden(&stage_items),
        PatternCheck::new("CUDA encode stage ownership", &stage).required(&stage_items),
    ]);
}
