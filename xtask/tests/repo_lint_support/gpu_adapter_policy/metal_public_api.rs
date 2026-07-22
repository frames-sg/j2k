// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

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
    let direct_plan_cache =
        fs::read_to_string(root.join("crates/j2k-metal/src/session/direct_plan_cache.rs"))
            .expect("read j2k-metal direct-plan cache module");
    let direct_plan_cache_tests =
        fs::read_to_string(root.join("crates/j2k-metal/src/session/direct_plan_cache/tests.rs"))
            .expect("read j2k-metal direct-plan cache tests");

    let session_items = [
        "pub struct MetalBackendSession",
        "pub struct MetalSession",
        "pub(crate) fn record_submit",
    ];
    let direct_plan_cache_items = [
        "pub(super) struct DirectPlanCaches",
        "struct DirectGrayPlanCacheEntry",
        "struct DirectColorPlanCacheEntry",
        "const DIRECT_PLAN_CACHE_CAP",
        "fn evict_one_direct_plan_if_needed",
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
        PatternCheck::new("j2k-metal session lifecycle ownership", &session)
            .required(&[
                "mod direct_plan_cache;",
                "direct_plan_caches: direct_plan_cache::DirectPlanCaches",
            ])
            .required(&session_items)
            .forbidden(&direct_plan_cache_items),
        PatternCheck::new("j2k-metal direct-plan cache ownership", &direct_plan_cache)
            .required(&direct_plan_cache_items)
            .required(&["mod tests;"])
            .forbidden(&[
                "fn prepared_plan_cache_allocation_keeps_its_source_and_classification",
                "fn prepared_plan_cache_invariant_keeps_static_reason_without_source",
            ]),
        PatternCheck::new(
            "j2k-metal direct-plan cache test ownership",
            &direct_plan_cache_tests,
        )
        .required(&[
            "fn prepared_plan_cache_allocation_keeps_its_source_and_classification",
            "fn prepared_plan_cache_invariant_keeps_static_reason_without_source",
        ]),
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
fn published_metal_batch_method_shapes_remain_compatible() {
    let root = repo_root();
    let contracts =
        fs::read_to_string(root.join("crates/j2k-metal/src/batch_decoder/contracts.rs"))
            .expect("read Metal batch contracts");
    let decoder = fs::read_to_string(root.join("crates/j2k-metal/src/batch_decoder/decoder.rs"))
        .expect("read Metal batch decoder");

    assert!(contracts.contains("pub fn resident_batch(&self) -> Option<&MetalResidentBatch>"));
    assert!(contracts.contains("pub fn into_resident_batch(self) -> Option<MetalResidentBatch>"));
    assert_eq!(
        contracts
            .matches("pub type MetalBatchGroupParts = (")
            .count(),
        1,
        "the published owned-parts alias must have one platform-independent shape"
    );
    let parts_alias = contracts
        .split_once("pub type MetalBatchGroupParts = (")
        .expect("find published Metal batch parts alias")
        .1
        .split_once(");")
        .expect("isolate published Metal batch parts alias")
        .0;
    assert!(!parts_alias.contains("MetalResidentBatch"));
    assert!(decoder.contains("pub fn submissions(&self) -> Result<u64, Error>"));
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
