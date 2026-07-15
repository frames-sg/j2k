// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::*;

mod cuda_decoder_policy;
mod cuda_encode_structure_policy;
mod cuda_encode_test_structure_policy;
mod cuda_htj2k_runtime_structure_policy;
mod cuda_profile_policy;
mod cuda_runtime_safety_policy;
mod jpeg_allocation_policy;
mod jpeg_batch_error_policy;
mod jpeg_fast_packet_routing_policy;
mod jpeg_generic_scan_policy;
mod jpeg_metal_compute_structure_policy;
mod jpeg_metal_plan_owner_policy;
mod jpeg_metal_surface_access_policy;
mod jpeg_metal_viewport_structure_policy;
mod jpeg_plan_cache_policy;
mod metal_batch_allocation_policy;
mod metal_plan_cache_policy;
mod metal_surface_access_policy;
mod metal_transcode_allocation_policy;
mod metal_typed_error_policy;
mod resident_encode_policy;

#[test]
fn metal_resident_retry_uses_typed_error_classification() {
    let root = repo_root();
    let resident_estimate =
        fs::read_to_string(root.join("crates/j2k-metal/src/encode/resident_estimate.rs"))
            .expect("read resident estimate");
    let metal_error = fs::read_to_string(root.join("crates/j2k-metal/src/error.rs"))
        .expect("read j2k-metal error source");
    let tier1_encode =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/tier1_encode.rs"))
            .expect("read j2k-metal tier1 encode source");
    let classification_sources = [
        resident_estimate.as_str(),
        metal_error.as_str(),
        tier1_encode.as_str(),
    ]
    .join("\n");

    assert_pattern_checks(&[
        PatternCheck::new("Metal resident retry decision source", &resident_estimate)
            .forbidden(&[".contains("]),
        PatternCheck::new(
            "typed Metal retry classification sources",
            &classification_sources,
        )
        .required(&[
            "MetalKernelRetryable",
            "encode_status_retry_class",
            "ResidentClassicBatch",
            "ResidentHtBatch",
            "is_conservative_retry_candidate",
        ]),
    ]);
}

#[test]
fn gpu_adapter_error_classification_uses_shared_core_impl() {
    let root = repo_root();
    let core_error =
        fs::read_to_string(root.join("crates/j2k-core/src/error.rs")).expect("read core error");
    assert_pattern_checks(&[
        PatternCheck::new("j2k-core adapter error classifier", &core_error).required(&[
            "pub enum AdapterErrorKind",
            "pub trait AdapterErrorParts",
            "adapter_error_is_unsupported",
            "adapter_error_is_buffer_error",
        ]),
    ]);
    let jpeg_metal_lib = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/lib.rs"))
        .expect("read JPEG Metal lib module");
    let jpeg_metal_decode_request =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/decode_request.rs"))
            .expect("read JPEG Metal decode request module");
    let jpeg_metal_decoder = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/decoder.rs"))
        .expect("read JPEG Metal decoder module");
    let jpeg_metal_codec_batch =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/codec_batch.rs"))
            .expect("read JPEG Metal codec batch module");
    assert!(
        jpeg_metal_lib.lines().count() < 932,
        "j2k-jpeg-metal lib.rs must keep focused public paths re-exported under the post-request-type-split line ratchet"
    );
    assert_pattern_checks(&[
        PatternCheck::new("j2k-jpeg-metal public module shell", &jpeg_metal_lib)
            .required(&[
                "mod error;",
                "pub use error::Error;",
                "mod decode_request;",
                "pub use decode_request::{MetalDecodeOp, MetalDecodeRequest};",
                "mod decoder;",
                "pub use decoder::Decoder;",
                "mod codec_batch;",
                "pub use codec_batch::{",
            ])
            .forbidden(&[
                "pub enum MetalDecodeOp",
                "pub struct MetalDecodeRequest",
                "pub enum Rgb8MetalBatchOp",
                "pub struct Decoder<'a>",
                "impl Codec {",
            ]),
        PatternCheck::new(
            "j2k-jpeg-metal decode request module",
            &jpeg_metal_decode_request,
        )
        .required(&["pub enum MetalDecodeOp", "pub struct MetalDecodeRequest"]),
        PatternCheck::new("j2k-jpeg-metal decoder module", &jpeg_metal_decoder)
            .required(&["pub struct Decoder<'a>", "impl<'a> Decoder<'a>"]),
        PatternCheck::new("j2k-jpeg-metal codec batch module", &jpeg_metal_codec_batch).required(
            &[
                "impl Codec",
                "pub enum Rgb8MetalBatchOp",
                "pub fn inspect_rgb8_decoder_batch_metal_output(",
            ],
        ),
    ]);

    let adapter_classifier_patterns = [
        "impl AdapterErrorParts for Error",
        "adapter_error_is_truncated(self)",
        "adapter_error_is_not_implemented(self)",
        "adapter_error_is_unsupported(self)",
        "adapter_error_is_buffer_error(self)",
    ];
    for relative in [
        "crates/j2k-cuda/src/error.rs",
        "crates/j2k-metal/src/error.rs",
        "crates/j2k-jpeg-cuda/src/error.rs",
        "crates/j2k-jpeg-metal/src/error.rs",
    ] {
        let source = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|err| panic!("read {relative}: {err}"));
        assert_pattern_checks(&[
            PatternCheck::new(relative, &source).required(&adapter_classifier_patterns)
        ]);
    }
}

#[test]
fn gpu_decoder_cpu_host_facades_use_core_blanket_impl() {
    let root = repo_root();
    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("crates/j2k-core/src/traits.rs")
                .named("j2k-core CPU-backed ImageDecode blanket impl")
                .required(&[
                    "pub trait CpuBackedImageDecode<'a>",
                    "impl<'a, T> ImageDecode<'a> for T",
                    "T: CpuBackedImageDecode<'a>",
                ]),
            FilePatternCheck::new("crates/j2k-cuda/src/decoder/api.rs")
                .required(&["impl<'a> CpuBackedImageDecode<'a>"])
                .forbidden(&["impl<'a> ImageDecode<'a>"]),
            FilePatternCheck::new("crates/j2k-metal/src/decoder/adapters.rs")
                .required(&["impl<'a> CpuBackedImageDecode<'a>"])
                .forbidden(&["impl<'a> ImageDecode<'a>"]),
            FilePatternCheck::new("crates/j2k-jpeg-cuda/src/decoder.rs")
                .required(&["impl<'a> CpuBackedImageDecode<'a>"])
                .forbidden(&["impl<'a> ImageDecode<'a>"]),
            FilePatternCheck::new("crates/j2k-jpeg-metal/src/decoder.rs")
                .required(&["impl<'a> CpuBackedImageDecode<'a>"])
                .forbidden(&["impl<'a> ImageDecode<'a>"]),
        ],
    );
}

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

#[test]
fn jpeg_metal_huffman_derivation_uses_shared_entropy_canonical_tables() {
    let root = repo_root();
    let codec_math = fs::read_to_string(root.join("crates/j2k-codec-math/src/jpeg.rs"))
        .expect("read codec-math JPEG helpers");
    let entropy_huffman = fs::read_to_string(root.join("crates/j2k-jpeg/src/entropy/huffman.rs"))
        .expect("read JPEG entropy Huffman implementation");
    let fast_packet_types =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/adapter/fast_packet/types.rs"))
            .expect("read JPEG fast packet type module");
    let metal_abi = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/abi.rs"))
        .expect("read JPEG Metal ABI");
    let cuda_runtime = read_source_files(
        root,
        &[
            "crates/j2k-cuda-runtime/src/jpeg.rs",
            "crates/j2k-cuda-runtime/src/jpeg/types.rs",
        ],
    );

    assert!(
        codec_math.contains("pub fn derive_canonical_huffman")
            && codec_math.contains("pub struct CanonicalHuffmanDerivation")
            && codec_math.contains("let mut huffsize")
            && codec_math.contains("let mut huffcode"),
        "j2k-codec-math must own the Annex C canonical Huffman derivation"
    );
    assert!(
        entropy_huffman.contains("pub(crate) fn derive_canonical_huffman")
            && entropy_huffman.contains("derive_canonical_huffman(raw)?"),
        "j2k-jpeg entropy must expose and use one shared Annex C canonical Huffman derivation"
    );
    assert!(
        fast_packet_types.contains("pub struct JpegCanonicalHuffmanTable")
            && fast_packet_types.contains("pub fn derive_canonical(&self)")
            && fast_packet_types.contains("derive_canonical_huffman(&raw)?"),
        "j2k-jpeg adapter must expose backend-facing canonical Huffman derivation"
    );
    assert!(
        metal_abi.contains(".derive_canonical()")
            && !metal_abi.contains("let mut huffsize")
            && !metal_abi.contains("let mut huffcode")
            && !metal_abi.contains("let mut code = 0u32")
            && !metal_abi.contains("for (len_minus_1, &count) in value.bits.iter().enumerate()"),
        "JPEG Metal ABI must pack shared canonical Huffman tables instead of deriving Annex C locally"
    );
    assert!(
        cuda_runtime.contains("j2k_codec_math::jpeg::derive_canonical_huffman")
            && !cuda_runtime.contains("let mut huffsize")
            && !cuda_runtime.contains("let mut huffcode")
            && !cuda_runtime.contains("let mut code = 0u32"),
        "CUDA JPEG runtime must use shared codec-math canonical Huffman derivation"
    );
}

#[test]
fn jpeg_metal_gpu_abi_uploads_are_padding_free() {
    let root = repo_root();
    let abi = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/abi.rs"))
        .expect("read JPEG Metal ABI");
    let buffers = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/buffers.rs"))
        .expect("read JPEG Metal buffers");
    let params =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/fast_packets/params.rs"))
            .expect("read JPEG Metal fast-packet params");
    let status = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/status.rs"))
        .expect("read JPEG Metal status");
    let shader = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/shaders_shared.metal"))
        .expect("read JPEG Metal shared shader ABI");

    assert_pattern_checks(&[
        PatternCheck::new("JPEG Metal padding-free ABI proof", &abi).required(&[
            "pub(crate) reserved_tail: u32",
            "macro_rules! prove_gpu_readback_layout",
            "let _: [(); core::mem::size_of::<$ty>()] = [(); $offset];",
            "core::mem::offset_of!($ty, $field)",
            "$offset + core::mem::size_of::<$field_ty>();",
            "prove_gpu_readback_layout!(",
            "JpegEntropyCheckpointHost {",
            "reserved_tail: u32",
        ]),
        PatternCheck::new("JPEG Metal typed upload boundary", &buffers)
            .required(&[
                "pub(crate) fn shared_buffer_with_slice<T: GpuAbi>",
                "let bytes = T::slice_as_bytes(values);",
            ])
            .forbidden(&["from_raw_parts(values.as_ptr().cast::<u8>()"]),
        PatternCheck::new("JPEG Metal checkpoint staging", &params)
            .required(&[
                "<u32 as GpuAbi>::slice_as_bytes(restart_offsets)",
                "checked_count_product(",
                "let buffer = new_shared_buffer(device, total_bytes)?;",
                "for (index, checkpoint) in entropy_checkpoints.iter().copied().enumerate()",
                "JpegEntropyCheckpointHost::as_bytes(&checkpoint)",
                "checked_copy_bytes_to_buffer_at(",
            ])
            .forbidden(&["from_raw_parts("]),
        PatternCheck::new("JPEG Metal status staging", &status)
            .required(&[
                "checked_count_product(",
                "core::mem::size_of::<JpegDecodeStatus>()",
                "let buffer = new_shared_buffer(device, bytes)?;",
                "checked_fill_buffer_u8(&buffer, bytes, 0",
                "checked_buffer_slice::<JpegDecodeStatus>(",
            ])
            .forbidden(&["from_raw_parts("]),
        PatternCheck::new("JPEG Metal checkpoint shader padding", &shader)
            .required(&["uint reserved_tail;"]),
    ]);
}

#[test]
fn j2k_metal_ht_uvlc_upload_uses_a_local_padding_free_abi_row() {
    let root = repo_root();
    let abi = fs::read_to_string(root.join("crates/j2k-metal/src/compute/abi.rs"))
        .expect("read J2K Metal ABI");
    let runtime = fs::read_to_string(root.join("crates/j2k-metal/src/compute/runtime.rs"))
        .expect("read J2K Metal runtime");
    let shader = fs::read_to_string(root.join("crates/j2k-metal/src/encode_bitstream_ht.metal"))
        .expect("read J2K Metal HT encoder shader");

    assert_pattern_checks(&[
        PatternCheck::new("J2K Metal HT UVLC padding-free upload row", &abi).required(&[
            "pub(crate) struct J2kHtUvlcEncodeTableEntry",
            "core::mem::offset_of!(J2kHtUvlcEncodeTableEntry, ext_len)",
            "core::mem::size_of::<J2kHtUvlcEncodeTableEntry>()",
            "unsafe impl j2k_core::accelerator::GpuAbi for J2kHtUvlcEncodeTableEntry",
            "ht_uvlc_upload_rows_match_the_canonical_packed_table",
            "j2k_native::ht_uvlc_encode_table_bytes()",
        ]),
        PatternCheck::new("J2K Metal typed HT UVLC upload", &runtime)
            .required(&[
                "(*ht_uvlc_encode_table()).map(J2kHtUvlcEncodeTableEntry::from)",
                "checked_shared_buffer_with_slice(",
                "&ht_uvlc_encode_rows",
            ])
            .forbidden(&[
                "ht_uvlc_encode_table_bytes",
                "checked_shared_buffer_with_bytes",
            ]),
        PatternCheck::new("J2K Metal byte-addressed HT UVLC shader ABI", &shader)
            .required(&["return table[index * 6u + field];"]),
    ]);
}

#[test]
fn fast444_region_scaled_batches_use_shared_region_scaled_metal_path() {
    let root = repo_root();
    let fast_packets = [
        "crates/j2k-jpeg-metal/src/compute/fast_packets/descriptors.rs",
        "crates/j2k-jpeg-metal/src/compute/fast_packets/pipelines.rs",
    ]
    .map(|path| fs::read_to_string(root.join(path)).expect("read JPEG Metal fast packet module"))
    .join("\n");
    let region_plan =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/region_scaled_plan.rs"))
            .expect("read JPEG Metal region scaled plan");
    let batch_decode =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/batch_region/rgb.rs"))
            .expect("read JPEG Metal region RGB batch decoder");

    assert_pattern_checks(&[
        PatternCheck::new("fast packet region-scaled Metal trait", &fast_packets).required(&[
            "trait FastRegionScaledMetal",
            "impl FastRegionScaledMetal for JpegFast444PacketV1",
            "fn chroma_width(width: u32) -> u32",
        ]),
        PatternCheck::new("region-scaled packet-family planning", &region_plan).required(&[
            "mode: PlaneMode",
            "plane_mode_to_u32(mode)",
            "P::chroma_width(source_window.w)",
        ]),
        PatternCheck::new("fast444 RGB region-scaled batch path", &batch_decode)
            .required(&[
                "try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output::<JpegFast444PacketV1>",
            ])
            .forbidden(&[
                "fn try_decode_fast444_region_scaled_rgb_batch_to_surfaces_with_output(",
                "fn try_decode_fast444_restart_region_scaled_rgb_batch_to_surfaces_with_output(",
                "fn try_decode_grouped_fast444_region_scaled_rgb_batch_to_surfaces_with_output(",
            ]),
    ]);
}

#[test]
fn fast444_full_batches_use_shared_fastsubsampled_metal_path() {
    let root = repo_root();
    let fast_packets = fs::read_to_string(
        root.join("crates/j2k-jpeg-metal/src/compute/fast_packets/pipelines.rs"),
    )
    .expect("read JPEG Metal fast packet pipelines");
    let batch_decode =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/batch_full/fast444.rs"))
            .expect("read JPEG Metal fast444 full batch decoder");

    assert_pattern_checks(&[
        PatternCheck::new("fast444 shared FastSubsampledMetal impl", &fast_packets)
            .required(&["impl FastSubsampledMetal for JpegFast444PacketV1"]),
        PatternCheck::new(
            "fast444 full shared region-scaled batch path",
            &batch_decode,
        )
        .required(&[
            "fn fast444_full_region_scaled_requests(",
            "scale: j2k_core::Downscale::None",
            "try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output::<",
            "JpegFast444PacketV1",
            "try_decode_fast444_region_scaled_rgba_batch_to_textures(",
        ])
        .forbidden(&[
            "struct Fast444FullRgbSurfaceShape",
            "struct Fast444FullRgbaTextureShape",
            "fn fast444_full_packets(",
            "fn try_decode_grouped_fast444_full_rgb_batch_to_surfaces_with_output(",
            "fn try_decode_grouped_fast444_full_rgba_batch_to_textures(",
            "fn encode_fast444_full_rgba_texture_decode(",
            "fast444_full_entropy",
        ]),
    ]);
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the complete fast-420 ownership and single-scan-loop ledger is one cohesive policy audit"
)]
fn jpeg_fast420_profiled_decode_uses_shared_scan_loop() {
    let root = repo_root();
    let sequential = fs::read_to_string(root.join("crates/j2k-jpeg/src/entropy/sequential.rs"))
        .expect("read JPEG entropy sequential decoder");
    let fast420 = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/entropy/sequential/fast420/mod.rs",
            "crates/j2k-jpeg/src/entropy/sequential/fast420/rows.rs",
        ],
    );
    let profile =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/entropy/sequential/profile.rs"))
            .expect("read JPEG entropy sequential profile module");
    let layout = fs::read_to_string(root.join("crates/j2k-jpeg/src/entropy/sequential/layout.rs"))
        .expect("read JPEG entropy sequential layout module");
    let restart =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/entropy/sequential/restart.rs"))
            .expect("read JPEG entropy sequential restart module");
    let deposit =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/entropy/sequential/deposit.rs"))
            .expect("read JPEG entropy sequential deposit module");
    let emit = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/entropy/sequential/emit.rs",
            "crates/j2k-jpeg/src/entropy/sequential/emit/region420.rs",
            "crates/j2k-jpeg/src/entropy/sequential/emit/rgb.rs",
            "crates/j2k-jpeg/src/entropy/sequential/emit/upsample.rs",
        ],
    );
    let tests = fs::read_to_string(root.join("crates/j2k-jpeg/src/entropy/sequential/tests.rs"))
        .expect("read JPEG entropy sequential tests module");

    assert!(
        sequential.lines().count() < 2_500,
        "entropy/sequential.rs must stay below the post-helper-split line-count ratchet"
    );

    assert_pattern_checks(&[
        PatternCheck::new("JPEG fast420 profile module ownership", &profile)
            .required(&[
                "trait Fast420ScanProfiler",
                "struct NoopFast420ScanProfile",
                "impl Fast420ScanProfiler for BenchFast420Profile",
            ])
            .forbidden(&["let mcu_start = Instant::now();"]),
        PatternCheck::new("JPEG fast420 shared scan loop ownership", &sequential)
            .required(&[
                "mod profile;",
                "mod layout;",
                "mod restart;",
                "mod deposit;",
                "mod emit;",
                "mod fast420;",
            ])
            .forbidden(&[
                "fn decode_scan_fast_tile_rgb_impl",
                "struct NoopFast420ScanProfile",
                "struct Fast420RegionLayout",
                "struct McuSkipState",
                "pub(super) fn deposit_block",
                "struct StripeEmit",
            ]),
        PatternCheck::new("JPEG fast420 shared scan loop implementation", &fast420)
            .required(&[
                "fn decode_scan_fast_tile_rgb_impl",
                "decode_scan_fast_tile_rgb_impl(plan, backend, scan_bytes, pool, writer, &mut profile)",
                "decode_scan_fast_tile_rgb_impl(plan, backend, scan_bytes, pool, writer, profile)",
            ])
            .forbidden(&[
                "struct NoopFast420ScanProfile",
                "struct Fast420RegionLayout",
                "struct McuSkipState",
                "pub(super) fn deposit_block",
                "struct StripeEmit",
            ]),
        PatternCheck::new("JPEG fast420 layout helper ownership", &layout).required(&[
            "pub(crate) fn stripe_region_layout(",
            "pub(crate) fn fast_tile_region_first_decode_mcu(",
            "struct Fast420RegionLayout",
            "fn expanded_output_rect(",
        ]),
        PatternCheck::new("JPEG fast420 restart helper ownership", &restart).required(&[
            "fn reader_from_checkpoint",
            "fn restart_seek_for_mcu",
            "struct McuSkipState",
            "fn skip_to_mcu",
        ]),
        PatternCheck::new("JPEG fast420 deposit helper ownership", &deposit).required(&[
            "fn assert_stripe_deposit_capacity",
            "fn deposit_block(",
            "fn deposit_dc_block(",
            "fn idct_deposit_fast_tile_block",
        ]),
        PatternCheck::new("JPEG fast420 emit helper ownership", &emit).required(&[
            "fn emit_stripe_rgb_420_region",
            "fn emit_stripe_rgb",
            "fn component_row_triplet",
            "fn upsample_component_row_stripe",
        ]),
        PatternCheck::new("JPEG fast420 sequential helper regression tests", &tests)
            .required(&["fast_tile_profiled_rgb_matches_unprofiled_decode"]),
    ]);
    assert_eq!(
        fast420.matches("finish_scan(&mut br, true)").count(),
        1,
        "JPEG fast420 profiled/unprofiled scan paths must not duplicate the scan loop"
    );
}

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

#[test]
fn transcode_gpu_auto_threshold_policy_is_documented() {
    let root = repo_root();
    let cuda = fs::read_to_string(root.join("crates/j2k-transcode-cuda/src/lib.rs"))
        .expect("read CUDA transcode adapter");
    let metal_root = fs::read_to_string(root.join("crates/j2k-transcode-metal/src/lib.rs"))
        .expect("read Metal transcode adapter");
    let metal_accelerator =
        fs::read_to_string(root.join("crates/j2k-transcode-metal/src/accelerator.rs"))
            .expect("read Metal transcode accelerator");
    let metal = format!("{metal_root}\n{metal_accelerator}");
    let cuda_readme = fs::read_to_string(root.join("crates/j2k-transcode-cuda/README.md"))
        .expect("read CUDA transcode README");
    let metal_readme = fs::read_to_string(root.join("crates/j2k-transcode-metal/README.md"))
        .expect("read Metal transcode README");

    let shared_auto_batch_thresholds = [
        "const DEFAULT_AUTO_REVERSIBLE_BATCH_MIN_JOBS: usize = 32;",
        "const DEFAULT_AUTO_REVERSIBLE_BATCH_MIN_SAMPLES: usize = 224 * 224 * 32;",
        "const DEFAULT_AUTO_DWT97_BATCH_MIN_JOBS: usize = 32;",
        "const DEFAULT_AUTO_DWT97_BATCH_MIN_SAMPLES: usize = 224 * 224 * 32;",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("CUDA transcode Auto batch thresholds", &cuda)
            .required(&shared_auto_batch_thresholds),
        PatternCheck::new("Metal transcode Auto batch thresholds", &metal)
            .required(&shared_auto_batch_thresholds),
        PatternCheck::new("CUDA transcode Auto threshold rationale", &cuda)
            .required(&["Batch thresholds below intentionally match Metal"]),
        PatternCheck::new("CUDA transcode README threshold rationale", &cuda_readme).required(&[
            "shared `224 * 224` component-sample floor",
            "defaults are routing policy, not a speedup promise",
        ]),
        PatternCheck::new("Metal transcode Auto threshold policy", &metal).required(&[
            "single-job Auto dispatch is disabled",
            "const DEFAULT_AUTO_DWT97_MIN_SAMPLES: usize = usize::MAX;",
            "const DEFAULT_AUTO_REVERSIBLE_MIN_SAMPLES: usize = usize::MAX;",
            "const MAX_AUTO_DWT97_STAGED_BATCH_AXIS: usize = 1024;",
        ]),
        PatternCheck::new("Metal transcode README staged-axis policy", &metal_readme).required(&[
            "either tile axis exceeds 1024 samples",
            "defaults are routing policy, not a speedup promise",
        ]),
    ]);
}

#[test]
fn transcode_stage_counters_are_shared_between_gpu_adapters() {
    let root = repo_root();
    let accelerator =
        fs::read_to_string(root.join("crates/j2k-transcode/src/accelerator_contracts.rs"))
            .expect("read transcode accelerator contracts");
    let cuda = fs::read_to_string(root.join("crates/j2k-transcode-cuda/src/lib.rs"))
        .expect("read CUDA transcode adapter");
    let metal_root = fs::read_to_string(root.join("crates/j2k-transcode-metal/src/lib.rs"))
        .expect("read Metal transcode adapter");
    let metal_accelerator =
        fs::read_to_string(root.join("crates/j2k-transcode-metal/src/accelerator.rs"))
            .expect("read Metal transcode accelerator");
    let metal_dispatch =
        fs::read_to_string(root.join("crates/j2k-transcode-metal/src/accelerator/dispatch.rs"))
            .expect("read Metal transcode dispatch implementation");
    let metal = format!("{metal_root}\n{metal_accelerator}\n{metal_dispatch}");

    assert_pattern_checks(&[PatternCheck::new(
        "j2k-transcode accelerator shared counters",
        &accelerator,
    )
    .required(&[
        "pub struct DctToWaveletStageCounters",
        "pub enum DctToWaveletStageCounterEvent",
        "pub enum TranscodeStageDispatchMode",
        "pub const fn unavailable<T>",
        "pub fn recover<T, E>",
        "pub fn record(&mut self, event: DctToWaveletStageCounterEvent, count: usize)",
        "DctToWaveletStageCounterEvent::Htj2k97CodeblockBatchAttempt",
        "DctToWaveletStageCounterEvent::Htj2k97CodeblockBatchDispatch",
    ])]);

    for (label, source) in [("CUDA", cuda.as_str()), ("Metal", metal.as_str())] {
        let check_name = format!("{label} transcode shared counters and dispatch policy");
        assert_pattern_checks(&[PatternCheck::new(&check_name, source)
            .required(&[
                "DctToWaveletStageCounterEvent as CounterEvent",
                "counters: DctToWaveletStageCounters",
                "self.counters.record(CounterEvent::",
                "mode: TranscodeStageDispatchMode",
                "self.mode.unavailable()",
            ])
            .forbidden(&[
                "reversible_dwt53_attempts: usize",
                "dwt53_attempts: usize",
                "dwt97_attempts: usize",
                "htj2k97_codeblock_batch_attempts: usize",
                "enum CudaDispatchMode",
                "enum MetalDispatchMode",
                "fn unavailable<T>(&self)",
                "MetalTranscodeError::MetalUnavailable | MetalTranscodeError::UnsupportedJob(_)",
            ])]);
    }

    assert_pattern_checks(&[
        PatternCheck::new("CUDA transcode shared recovery policy", &cuda)
            .required(&[".recover(error, CudaTranscodeError::is_recoverable)"]),
        PatternCheck::new("Metal transcode shared recovery policy", &metal)
            .required(&[".recover(error, MetalTranscodeError::is_recoverable)"]),
    ]);
}

#[test]
fn metal_public_error_lives_in_focused_module() {
    let root = repo_root();
    let lib = fs::read_to_string(root.join("crates/j2k-metal/src/lib.rs"))
        .expect("read j2k-metal lib module");
    let error = fs::read_to_string(root.join("crates/j2k-metal/src/error.rs"))
        .expect("read j2k-metal error module");

    let error_items = [
        "pub enum Error",
        "pub enum MetalDirectFallbackReason",
        "pub enum MetalKernelRetryClass",
        "impl AdapterErrorParts for Error",
        "impl CodecError for Error",
    ];
    let error_helpers = [
        "adapter_error_is_truncated",
        "adapter_error_is_not_implemented",
        "adapter_error_is_unsupported",
        "adapter_error_is_buffer_error",
        "is_conservative_retry_candidate",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal error module shell", &lib)
            .required(&[
                "mod error;",
                "pub use self::error::{",
                "Error, MetalDirectFallbackReason, MetalKernelRetryClass, NativeBackendError,",
            ])
            .forbidden(&error_items),
        PatternCheck::new("j2k-metal error item ownership", &error).required(&error_items),
        PatternCheck::new("j2k-metal error classification helpers", &error)
            .required(&error_helpers),
    ]);
}

#[test]
fn metal_surface_lives_in_focused_module() {
    let root = repo_root();
    let lib = fs::read_to_string(root.join("crates/j2k-metal/src/lib.rs"))
        .expect("read j2k-metal lib module");
    let surface = fs::read_to_string(root.join("crates/j2k-metal/src/surface.rs"))
        .expect("read j2k-metal surface module");

    let surface_items = [
        "pub struct Surface",
        "pub(crate) enum Storage",
        "impl Surface",
        "impl DeviceSurface for Surface",
        "fn checked_storage_range",
    ];
    let surface_helpers = [
        "SurfaceMetadata::new",
        "copy_tight_pixels_to_strided_output",
        "DeviceMemoryRange::new",
        "from_metal_buffer_with_offset",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal surface module shell", &lib)
            .required(&["mod surface;", "pub use self::surface::Surface;"])
            .forbidden(&surface_items),
        PatternCheck::new("j2k-metal surface item ownership", &surface).required(&surface_items),
        PatternCheck::new("j2k-metal surface helper ownership", &surface)
            .required(&surface_helpers),
    ]);
}

#[test]
fn j2k_metal_public_buffer_aliases_require_unsafe_boundaries() {
    let root = repo_root();
    let surface = fs::read_to_string(root.join("crates/j2k-metal/src/surface.rs"))
        .expect("read j2k-metal surface module");
    let encoded = fs::read_to_string(root.join("crates/j2k-metal/src/encode/encoded.rs"))
        .expect("read j2k-metal encoded output module");
    let encode_types = fs::read_to_string(root.join("crates/j2k-metal/src/encode/types.rs"))
        .expect("read j2k-metal encode types module");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal Surface raw buffer boundary", &surface)
            .required(&[
                "pub unsafe fn metal_buffer(&self) -> Option<(&Buffer, usize)>",
                "pub(crate) fn metal_buffer_trusted(&self) -> Option<(&Buffer, usize)>",
                "while this surface or any clone",
            ])
            .forbidden(&["pub fn metal_buffer(&self)"]),
        PatternCheck::new("j2k-metal encoded output raw buffer boundary", &encoded)
            .required(&[
                "pub(crate) codestream_buffer: Buffer",
                "pub unsafe fn from_raw_parts(",
                "pub unsafe fn into_codestream_buffer(self) -> Buffer",
                "pub(crate) fn codestream_buffer_trusted(&self) -> &Buffer",
                "pub fn byte_offset(&self) -> usize",
                "pub fn byte_len(&self) -> usize",
                "pub fn capacity(&self) -> usize",
                "until the returned object is dropped",
                "sibling tiles in a batch",
            ])
            .forbidden(&[
                "pub codestream_buffer: Buffer",
                "pub byte_offset: usize",
                "pub byte_len: usize",
                "pub capacity: usize",
                "pub fn codestream_buffer(&self)",
                "pub unsafe fn codestream_buffer(&self)",
                "pub fn into_codestream_buffer(self)",
            ]),
        PatternCheck::new("j2k-metal encode input raw buffer boundary", &encode_types)
            .required(&[
                "pub(super) buffer: &'a Buffer",
                "pub unsafe fn from_buffer(",
                "pub(crate) fn from_trusted_buffer(",
                "Dropping a submitted operation",
                "actually completed",
                "same Metal device",
                "every [`crate::MetalBackendSession`]",
            ])
            .forbidden(&["pub buffer: &'a Buffer", "pub fn from_buffer("]),
    ]);
}

#[test]
fn metal_sessions_and_direct_plan_caches_live_in_focused_module() {
    let root = repo_root();
    let lib = fs::read_to_string(root.join("crates/j2k-metal/src/lib.rs"))
        .expect("read j2k-metal lib module");
    let session = fs::read_to_string(root.join("crates/j2k-metal/src/session.rs"))
        .expect("read j2k-metal session module");

    let session_items = [
        "pub struct MetalBackendSession",
        "pub struct MetalSession",
        "struct DirectGrayPlanCacheEntry",
        "struct DirectColorPlanCacheEntry",
        "const DIRECT_PLAN_CACHE_CAP",
        "fn evict_one_direct_plan_if_needed",
        "pub(crate) fn record_submit",
    ];
    let session_helpers = [
        "pub(crate) fn direct_plan_cache_key",
        "pub(crate) fn direct_gray_plan_cache_key",
        "pub(crate) fn cached_session_direct_gray_plan",
        "pub(crate) fn store_session_direct_gray_plan",
        "pub(crate) fn cached_session_direct_color_plan",
        "pub(crate) fn store_session_direct_color_plan",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal session module shell", &lib)
            .required(&[
                "mod session;",
                "pub use self::session::{MetalBackendSession, MetalSession};",
            ])
            .forbidden(&session_items),
        PatternCheck::new("j2k-metal session item ownership", &session).required(&session_items),
        PatternCheck::new("j2k-metal direct-plan cache helper ownership", &session)
            .required(&session_helpers),
    ]);
}

#[test]
fn metal_tile_batch_lives_in_focused_module() {
    let root = repo_root();
    let lib = fs::read_to_string(root.join("crates/j2k-metal/src/lib.rs"))
        .expect("read j2k-metal lib module");
    let tile_batch = fs::read_to_string(root.join("crates/j2k-metal/src/tile_batch.rs"))
        .expect("read j2k-metal tile batch module");

    let tile_batch_items = [
        "pub struct MetalTileBatch",
        "impl MetalTileBatch",
        "pub fn push_tile_request(",
        "pub fn push_shared_tile_request(",
        "pub fn decode_all(",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal tile batch module shell", &lib)
            .required(&[
                "mod tile_batch;",
                "pub use self::tile_batch::MetalTileBatch;",
            ])
            .forbidden(&tile_batch_items),
        PatternCheck::new("j2k-metal tile batch item ownership", &tile_batch)
            .required(&tile_batch_items),
    ]);
}

#[test]
fn metal_decoder_api_lives_in_focused_module() {
    let root = repo_root();
    let lib = fs::read_to_string(root.join("crates/j2k-metal/src/lib.rs"))
        .expect("read j2k-metal lib module");
    let decoder = fs::read_to_string(root.join("crates/j2k-metal/src/decoder.rs"))
        .expect("read j2k-metal decoder facade");
    let adapters = fs::read_to_string(root.join("crates/j2k-metal/src/decoder/adapters.rs"))
        .expect("read j2k-metal decoder adapters");
    let core = fs::read_to_string(root.join("crates/j2k-metal/src/decoder/core.rs"))
        .expect("read j2k-metal decoder core");
    let direct_paths =
        fs::read_to_string(root.join("crates/j2k-metal/src/decoder/direct_paths.rs"))
            .expect("read j2k-metal decoder direct paths");
    let request = fs::read_to_string(root.join("crates/j2k-metal/src/decoder/request.rs"))
        .expect("read j2k-metal decoder request types");
    let routes = fs::read_to_string(root.join("crates/j2k-metal/src/decoder/routes.rs"))
        .expect("read j2k-metal decoder routes");
    let surface = fs::read_to_string(root.join("crates/j2k-metal/src/decoder/surface.rs"))
        .expect("read j2k-metal decoder surface transfer");

    assert!(
        lib.lines().count() < 300,
        "j2k-metal lib.rs must stay below 300 lines after the item 53 split"
    );
    let decoder_items = [
        "pub struct J2kDecoder",
        "pub struct Codec",
        "pub enum DecodeOperation",
        "pub struct DecodeRouteReport",
        "pub struct DecodeSurfaceWithReport",
        "fn upload_surface(",
        "pub(crate) fn decode_to_surface_impl",
        "macro_rules! define_ensure_prepared_direct_plan",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal decoder crate shell", &lib)
            .required(&["mod decoder;", "pub use self::decoder::{", "J2kDecoder"])
            .forbidden(&decoder_items),
        PatternCheck::new("j2k-metal decoder module shell", &decoder)
            .required(&[
                "mod adapters;",
                "mod core;",
                "mod direct_paths;",
                "mod request;",
                "mod routes;",
                "mod surface;",
            ])
            .forbidden(&decoder_items),
        PatternCheck::new("j2k-metal decoder core ownership", &core)
            .required(&["pub struct J2kDecoder"]),
        PatternCheck::new("j2k-metal decoder adapter ownership", &adapters)
            .required(&["pub struct Codec", "impl<'a> CpuBackedImageDecode<'a>"]),
        PatternCheck::new("j2k-metal decoder request ownership", &request).required(&[
            "pub enum DecodeOperation",
            "pub enum MetalDecodeOp",
            "pub struct MetalDecodeRequest",
            "pub struct DecodeRouteReport",
            "pub struct DecodeSurfaceWithReport",
        ]),
        PatternCheck::new("j2k-metal decoder direct-path ownership", &direct_paths)
            .required(&["macro_rules! define_ensure_prepared_direct_plan"]),
        PatternCheck::new("j2k-metal decoder route ownership", &routes)
            .required(&["pub(crate) fn decode_to_surface_impl"]),
        PatternCheck::new("j2k-metal decoder surface ownership", &surface)
            .required(&["fn upload_surface("]),
    ]);
}

#[test]
fn metal_batch_heuristics_live_in_focused_module() {
    let root = repo_root();
    let batch = fs::read_to_string(root.join("crates/j2k-metal/src/batch.rs"))
        .expect("read j2k-metal batch module");
    let heuristics = fs::read_to_string(root.join("crates/j2k-metal/src/batch/heuristics.rs"))
        .expect("read j2k-metal batch heuristics module");

    let heuristic_items = [
        "pub(super) enum BatchRoute",
        "pub(super) struct GroupedRequests",
        "pub(super) fn group_metal_requests",
        "pub(super) fn profile_route_label",
        "pub(super) fn is_region_scaled_direct_batch_candidate",
        "pub(super) fn should_auto_use_metal_for_region_scaled_direct_batch",
        "pub(super) fn can_decode_requests_as_repeated_region_scaled_batch",
    ];
    let heuristic_required = [
        "pub(super) enum BatchRoute",
        "pub(super) struct GroupedRequests",
        "pub(super) fn group_metal_requests",
        "pub(super) fn profile_route_label",
        "pub(super) fn is_region_scaled_direct_batch_candidate",
        "pub(super) fn should_auto_use_metal_for_region_scaled_direct_batch",
        "pub(super) fn can_decode_requests_as_repeated_region_scaled_batch",
        "AUTO_REGION_SCALED_DIRECT_BATCH64_MIN_DIM",
        "REGION_SCALED_DIRECT_FORMATS",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal batch heuristic module shell", &batch)
            .required(&[
                "mod heuristics;",
                "use self::heuristics::{",
                "group_metal_requests",
            ])
            .forbidden(&heuristic_items),
        PatternCheck::new("j2k-metal batch heuristic ownership", &heuristics)
            .required(&heuristic_required),
    ]);
}

#[test]
fn metal_batch_cpu_fallback_lives_in_focused_module() {
    let root = repo_root();
    let batch = fs::read_to_string(root.join("crates/j2k-metal/src/batch.rs"))
        .expect("read j2k-metal batch module");
    let cpu = fs::read_to_string(root.join("crates/j2k-metal/src/batch/cpu.rs"))
        .expect("read j2k-metal batch CPU module");

    let cpu_items = [
        "pub(super) fn decode_cpu_host_batch",
        "fn decode_cpu_full_batch",
        "fn decode_cpu_region_scaled_batch",
        "fn checked_cpu_batch_surface",
        "fn cpu_batch_error",
        "fn host_surface",
        "decode_tiles_into",
        "decode_tiles_region_scaled_into",
        "BatchDecodeError::Tile(error)",
        "BatchDecodeError::Infrastructure(error)",
        "BufferError::AllocationTooLarge",
        "BufferError::HostAllocationFailed",
        "Error::BatchInfrastructure(other)",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal batch CPU fallback module shell", &batch)
            .required(&["mod cpu;", "use self::cpu::decode_cpu_host_batch;"])
            .forbidden(&cpu_items),
        PatternCheck::new("j2k-metal batch CPU fallback ownership", &cpu).required(&cpu_items),
    ]);
}

#[test]
fn metal_batch_execute_lives_in_focused_module() {
    let root = repo_root();
    let batch = fs::read_to_string(root.join("crates/j2k-metal/src/batch.rs"))
        .expect("read j2k-metal batch module");
    let execute = fs::read_to_string(root.join("crates/j2k-metal/src/batch/execute.rs"))
        .expect("read j2k-metal batch execute module");

    let execute_items = [
        "pub(super) fn process_batch",
        "fn process_batch_inner",
        "fn complete_cpu_host_fallback",
        "fn complete_batch_surfaces",
        "fn profile_completed_outcome",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal batch execute module shell", &batch)
            .required(&["mod execute;", "use self::execute::process_batch;"])
            .forbidden(&execute_items),
        PatternCheck::new("j2k-metal batch execute ownership", &execute).required(&execute_items),
    ]);
    assert_eq!(
        execute
            .matches("session.completed[request.output_slot] = Some(Ok(surface));")
            .count(),
        1,
        "batch execution must use one shared successful-completion block"
    );
}

#[test]
fn jpeg_metal_viewport_plane_rows_use_shared_target() {
    let root = repo_root();
    let viewport_cache =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/viewport_cache.rs"))
            .expect("read j2k-jpeg-metal viewport cache");

    assert_pattern_checks(&[PatternCheck::new(
        "j2k-jpeg-metal viewport row writers",
        &viewport_cache,
    )
    .required(&[
        "struct PlaneRowTarget<'a>",
        "impl ComponentRowWriter for PlaneStage",
        "impl ComponentRowWriter for ViewportPlaneWriter<'_>",
        "impl PlaneRowTarget<'_>",
        "fn write_plane_row(&self, buffer: &Buffer, y: u32, src: &[u8]) -> Result<(), Error>",
        "fn checked_write_row_u8_at(",
        "checked_copy_bytes_to_buffer_at(",
    ])
    .normalized_required(&[
        "self.row_target() .write_gray_row(y, gray_row) .map_err(jpeg_plane_write_error)",
        "self.row_target() .write_ycbcr_row(y, y_row, chroma_blue_row, chroma_red_row) .map_err(jpeg_plane_write_error)",
        "self.row_target() .write_rgb_row(y, r_row, g_row, b_row) .map_err(jpeg_plane_write_error)",
    ])
    .forbidden(&["fn write_row_u8(", ".contents()"])]);
    assert!(
        viewport_cache
            .matches("fn row_target(&self) -> PlaneRowTarget<'_>")
            .count()
            == 2,
        "PlaneStage and ViewportPlaneWriter must both delegate through PlaneRowTarget"
    );
}

#[test]
fn jpeg_metal_single_decode_uses_request_api() {
    let root = repo_root();
    let lib = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/lib.rs"))
        .expect("read j2k-jpeg-metal lib");
    let codec_batch = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/codec_batch.rs"))
        .expect("read j2k-jpeg-metal codec batch module");
    let decode_request =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/decode_request.rs"))
            .expect("read j2k-jpeg-metal decode request module");
    let decoder = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/decoder.rs"))
        .expect("read j2k-jpeg-metal decoder module");
    let tile_batch = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/tile_batch.rs"))
        .expect("read j2k-jpeg-metal tile batch module");
    let source = format!("{lib}\n{codec_batch}\n{decode_request}\n{decoder}\n{tile_batch}");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-jpeg-metal request API routing", &source)
            .required(&[
                "pub enum MetalDecodeOp",
                "pub struct MetalDecodeRequest",
                "pub fn decode_request_to_device(",
                "pub fn push_tile_request(",
                "pub fn push_shared_tile_request(",
                "pub fn submit_tile_request_to_device(",
                "Self::submit_tile_request_to_device(",
                "MetalDecodeRequest::region_scaled(fmt, roi, scale, backend)",
            ])
            .forbidden(&[
                "pub fn decode_region_scaled_to_device(",
                "pub fn push_tile(",
                "pub fn push_shared_tile(",
                "pub fn push_tile_region(",
                "pub fn push_shared_tile_region(",
                "pub fn push_tile_scaled(",
                "pub fn push_shared_tile_scaled(",
                "pub fn push_tile_region_scaled(",
                "pub fn push_shared_tile_region_scaled(",
            ]),
    ]);
    assert_pattern_checks(&[
        PatternCheck::new("j2k-jpeg-metal tile batch module shell", &lib)
            .required(&["mod tile_batch;", "pub use tile_batch::JpegTileBatch;"])
            .forbidden(&["pub struct JpegTileBatch"]),
        PatternCheck::new("j2k-jpeg-metal tile batch ownership", &tile_batch)
            .required(&["pub struct JpegTileBatch", "impl JpegTileBatch"]),
        PatternCheck::new("j2k-jpeg-metal decoder module shell", &lib)
            .required(&["mod decoder;", "pub use decoder::Decoder;"])
            .forbidden(&["pub struct Decoder<'a>"]),
        PatternCheck::new("j2k-jpeg-metal decoder ownership", &decoder)
            .required(&["pub struct Decoder<'a>", "impl<'a> Decoder<'a>"]),
        PatternCheck::new("j2k-jpeg-metal codec batch module shell", &lib)
            .required(&["mod codec_batch;", "pub use codec_batch::{"])
            .forbidden(&["impl Codec {", "pub enum Rgb8MetalBatchOp"]),
        PatternCheck::new("j2k-jpeg-metal codec batch ownership", &codec_batch)
            .required(&[
                "impl Codec",
                "pub enum Rgb8MetalBatchOp",
                "pub fn submit_tile_request_to_device(",
                "pub fn decode_rgb8_batch_into_buffer_with_session(",
            ])
            .forbidden(&["pub fn submit_tile_region_scaled_to_device("]),
        PatternCheck::new("j2k-jpeg-metal decode request module shell", &lib)
            .required(&[
                "mod decode_request;",
                "pub use decode_request::{MetalDecodeOp, MetalDecodeRequest};",
            ])
            .forbidden(&["pub enum MetalDecodeOp", "pub struct MetalDecodeRequest"]),
        PatternCheck::new("j2k-jpeg-metal decode request ownership", &decode_request)
            .required(&["pub enum MetalDecodeOp", "pub struct MetalDecodeRequest"]),
    ]);
}

#[test]
fn j2k_metal_decode_and_tile_batch_use_request_api() {
    let root = repo_root();
    let lib =
        fs::read_to_string(root.join("crates/j2k-metal/src/lib.rs")).expect("read j2k-metal lib");
    let decoder = read_source_files(
        root,
        &[
            "crates/j2k-metal/src/decoder.rs",
            "crates/j2k-metal/src/decoder/adapters.rs",
            "crates/j2k-metal/src/decoder/core.rs",
            "crates/j2k-metal/src/decoder/request.rs",
        ],
    );
    let tile_batch = fs::read_to_string(root.join("crates/j2k-metal/src/tile_batch.rs"))
        .expect("read j2k-metal tile batch");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal decoder request API routing", &decoder)
            .required(&[
                "pub enum MetalDecodeOp",
                "pub struct MetalDecodeRequest",
                "pub fn decode_request_to_device(",
                "pub fn decode_request_to_device_with_report(",
                "pub fn decode_request_to_device_with_session(",
                "pub fn decode_request_to_host_surface(",
                "pub fn decode_request_to_cpu_staged_metal_surface_with_session(",
                "let request = MetalDecodeRequest::region_scaled(fmt, roi, scale, backend);",
                "request.op.batch_op()",
            ])
            .forbidden(&[
                "pub fn decode_to_device_with_report(",
                "pub fn decode_region_to_device_with_report(",
                "pub fn decode_scaled_to_device_with_report(",
                "pub fn decode_region_scaled_to_device_with_report(",
                "pub fn decode_to_device_with_session(",
                "pub fn decode_region_to_device_with_session(",
                "pub fn decode_scaled_to_device_with_session(",
                "pub fn decode_region_scaled_to_device_with_session(",
                "pub fn decode_to_host_surface(",
                "pub fn decode_region_to_host_surface(",
                "pub fn decode_scaled_to_host_surface(",
                "pub fn decode_region_scaled_to_host_surface(",
                "pub fn decode_to_cpu_staged_metal_surface_with_session(",
                "pub fn decode_region_to_cpu_staged_metal_surface_with_session(",
                "pub fn decode_scaled_to_cpu_staged_metal_surface_with_session(",
                "pub fn decode_region_scaled_to_cpu_staged_metal_surface_with_session(",
            ]),
        PatternCheck::new("j2k-metal decode request type re-export", &lib)
            .required(&["MetalDecodeOp", "MetalDecodeRequest"]),
    ]);
    assert_pattern_checks(&[PatternCheck::new(
        "j2k-metal tile batch request API routing",
        &tile_batch,
    )
    .required(&[
        "pub fn push_tile_request(",
        "pub fn push_shared_tile_request(",
        "self.push_shared_tile_request(",
    ])
    .forbidden(&[
        "pub fn push_tile(",
        "pub fn push_shared_tile(",
        "pub fn push_tile_region(",
        "pub fn push_shared_tile_region(",
        "pub fn push_tile_scaled(",
        "pub fn push_shared_tile_scaled(",
        "pub fn push_tile_region_scaled(",
        "pub fn push_shared_tile_region_scaled(",
    ])]);
}
