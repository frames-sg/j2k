// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::*;

#[test]
fn jpeg_metal_host_readback_aliases_require_unsafe_contracts() {
    let root = repo_root();
    let surface = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/surface.rs"))
        .expect("read JPEG Metal surface module");
    let encode = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/encode.rs"))
        .expect("read JPEG Metal encode module");
    let batch_entry =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/batch_entry.rs"))
            .expect("read JPEG Metal batch decode entry module");
    let batch_support =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/batch_support.rs"))
            .expect("read JPEG Metal batch support module");
    let pack_dispatch =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/pack_dispatch/common.rs"))
            .expect("read JPEG Metal common pack dispatch module");
    let viewport_cache =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/viewport_cache.rs"))
            .expect("read JPEG Metal viewport cache module");
    let compute = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute.rs"))
        .expect("read JPEG Metal compute module");
    let compute_tests = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/tests.rs"))
        .expect("read JPEG Metal compute tests");

    assert_pattern_checks(&[
        PatternCheck::new("JPEG Metal host-readback alias boundary", &surface)
            .required(&[
                "pub unsafe fn metal_buffer(&self)",
                "pub(crate) fn metal_buffer_trusted(&self)",
                "pub unsafe fn buffer(&self) -> &BufferRef",
                "pub(crate) fn buffer_trusted(&self)",
                "no command may write the surface range while",
                "safe decode into this output or readback",
                "access_gate: Option<Arc<Mutex<()>>>",
                "access_gate: Arc<Mutex<()>>",
                "gate.lock()",
                "fn storage_bytes(&self) -> Result<Cow",
                "let bytes = self.storage_bytes()?;",
                "pub(crate) fn from_batch_output_buffer_offset(",
                "access_gate: Some(Arc::clone(&output.access_gate))",
                "pub(crate) fn lock_for_safe_access(&self)",
            ])
            .forbidden(&[
                "pub fn metal_buffer(&self)",
                "pub fn buffer(&self) -> &BufferRef",
            ]),
        PatternCheck::new("JPEG Metal encode source invariant", &encode)
            .required(&[
                "pub unsafe fn new(",
                "pub unsafe fn buffer(&self) -> &BufferRef",
                "pub(crate) fn buffer_trusted(&self)",
                "pub fn byte_offset(&self)",
                "pub fn dimensions(&self)",
                "pub fn pitch_bytes(&self)",
                "pub fn output_dimensions(&self)",
                "pub fn pixel_format(&self)",
                "The caller must keep that range immutable",
            ])
            .forbidden(&[
                "pub buffer: &'a Buffer",
                "pub byte_offset: usize",
                "pub width: u32",
                "pub height: u32",
                "pub pitch_bytes: usize",
                "pub output_width: u32",
                "pub output_height: u32",
                "pub format: PixelFormat",
            ]),
        PatternCheck::new("JPEG Metal reusable-output write gates", &batch_entry).required(&[
            "fn decode_full_rgb8_batch_into_output_with_session(",
            "fn decode_region_scaled_rgb8_batch_into_output_with_session(",
            "let _output_access = output.lock_for_safe_access()?;",
        ]),
        PatternCheck::new(
            "JPEG Metal direct reusable-output surface aliases",
            &batch_support,
        )
        .required(&[
            "output: Option<&crate::MetalBatchOutputBuffer>",
            "Surface::from_batch_output_buffer_offset(",
        ]),
        PatternCheck::new(
            "JPEG Metal grouped reusable-output surface aliases",
            &pack_dispatch,
        )
        .required(&["Surface::from_batch_output_buffer_offset("]),
        PatternCheck::new(
            "JPEG Metal viewport reusable-output surface aliases",
            &viewport_cache,
        )
        .required(&[
            "let _output_access = output.lock_for_safe_access()?;",
            "Surface::from_batch_output_buffer_offset(",
            "pub(super) struct ViewportPlaneCacheGate",
            "pub(super) struct ViewportPlaneCacheLease",
            "cache_lease: Option<ViewportPlaneCacheLease>",
            "let cache_lease = runtime.viewport_plane_cache_lease()?;",
            "cache_lease: Some(cache_lease)",
            "let cache_owned = self.cache_lease.is_some();",
            "PixelFormat::Gray8) if !cache_owned",
        ]),
        PatternCheck::new("JPEG Metal viewport cache runtime lease", &compute).required(&[
            "viewport_plane_cache_gate: Arc<ViewportPlaneCacheGate>",
            "viewport_plane_cache_gate: ViewportPlaneCacheGate::new()",
            "fn viewport_plane_cache_lease(&self) -> Result<ViewportPlaneCacheLease, Error>",
        ]),
        PatternCheck::new(
            "JPEG Metal viewport cache concurrency regressions",
            &compute_tests,
        )
        .required(&[
            "fn viewport_plane_cache_lease_serializes_cloned_sessions()",
            "fn cached_gray_stage_returns_fresh_public_surface()",
            "safe-readable Gray8 output must not alias the reusable plane cache",
        ]),
    ]);

    assert_eq!(
        batch_entry
            .matches("let _output_access = output.lock_for_safe_access()?;")
            .count(),
        2,
        "full and region-scaled reusable buffer writes must each hold the allocation gate"
    );

    let lease_index = viewport_cache
        .find("let cache_lease = runtime.viewport_plane_cache_lease()?;")
        .expect("viewport cache lease acquisition");
    let slot_index = viewport_cache
        .find("let mut slot = runtime.viewport_plane_cache()?;")
        .expect("viewport cache slot acquisition");
    assert!(
        lease_index < slot_index,
        "the viewport cache lease must be acquired before any cached buffer is cloned"
    );
}

#[test]
fn jpeg_metal_private_texture_aliases_share_safe_write_ordering() {
    let root = repo_root();
    let surface = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/surface.rs"))
        .expect("read JPEG Metal surface module");
    let batch_entry =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/batch_entry.rs"))
            .expect("read JPEG Metal batch decode entry module");
    let viewport_compose =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/viewport_compose.rs"))
            .expect("read JPEG Metal viewport composition module");
    let reusable_output_tests =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/tests/reusable_output.rs"))
            .expect("read JPEG Metal reusable output tests");
    let texture_tests =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/tests/textures.rs"))
            .expect("read JPEG Metal texture tests");

    assert_pattern_checks(&[
        PatternCheck::new("JPEG Metal raw private texture boundaries", &surface)
            .required(&[
                "pub unsafe fn texture(&self, index: usize) -> Option<&TextureRef>",
                "pub(crate) fn texture_trusted(&self, index: usize)",
                "pub unsafe fn texture(&self) -> &TextureRef",
                "pub(crate) fn texture_trusted(&self) -> &TextureRef",
                "gate cannot observe work submitted through raw handles",
                "safe decode gate",
            ])
            .forbidden(&[
                "pub fn texture(&self, index: usize) -> Option<&TextureRef>",
                "pub fn texture(&self) -> &TextureRef",
            ]),
        PatternCheck::new("JPEG Metal shared texture allocation gate", &surface).required(&[
            "pub struct MetalBatchTextureOutput",
            "access_gate: Arc<Mutex<()>>",
            "access_gate: Arc::new(Mutex::new(()))",
            "pub(crate) fn lock_for_safe_access(&self)",
            "pub(crate) fn clone_slots(&self, indices: &[usize])",
            "access_gate: Arc::clone(&self.access_gate)",
            "pub(crate) fn clone_access_gate(&self)",
            "pub struct MetalTextureTile",
        ]),
        PatternCheck::new("JPEG Metal synchronous texture write gates", &batch_entry).required(&[
            "fn decode_full_rgb8_batch_into_textures_with_session(",
            "fn decode_region_scaled_rgb8_batch_into_textures_with_session(",
            "let _texture_output_access = output.lock_for_safe_access()?;",
        ]),
        PatternCheck::new("JPEG Metal viewport texture write gate", &viewport_compose).required(&[
            "fn compose_rgb_viewport_from_regions_into_textures_with_session(",
            "let _texture_output_access = output.lock_for_safe_access()?;",
            "finish_rgba8_into_texture_output_with_runtime(runtime, output)",
        ]),
        PatternCheck::new(
            "JPEG Metal texture gate behavior regressions",
            &reusable_output_tests,
        )
        .required(&[
            "fn reusable_texture_output_clones_and_subsets_share_one_access_gate()",
            "output.shares_access_gate_with(&output_clone)",
            "output.shares_access_gate_with(&output_subset)",
            "a cloned output must wait while the shared allocation gate is held",
            "a resized output must receive a gate for its new allocation",
        ]),
        PatternCheck::new("JPEG Metal raw texture contract regression", &texture_tests).required(
            &[
                "output.shares_access_gate_with_tile(&tile)",
                "let tile_texture = unsafe { tile.texture() };",
                "let output_texture = unsafe { output.texture(index) }",
                "submits no overlapping writer",
            ],
        ),
    ]);

    assert_eq!(
        batch_entry
            .matches("let _texture_output_access = output.lock_for_safe_access()?;")
            .count(),
        2,
        "full and region-scaled texture writes must each hold the allocation gate"
    );
    assert_eq!(
        viewport_compose
            .matches("let _texture_output_access = output.lock_for_safe_access()?;")
            .count(),
        1,
        "viewport texture composition must hold the allocation gate through completion"
    );
}

#[test]
fn jpeg_metal_resident_private_tile_hides_raw_keepalive_resources() {
    let root = repo_root();
    let surface = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/surface.rs"))
        .expect("read JPEG Metal surface module");
    let texture_tests =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/tests/textures.rs"))
            .expect("read JPEG Metal texture tests");

    assert_pattern_checks(&[
        PatternCheck::new(
            "JPEG Metal resident private tile resource boundary",
            &surface,
        )
        .required(&[
            "pub struct ResidentPrivateJpegTile",
            "pub(crate) fn new(",
            "pub fn byte_offset(&self) -> usize",
            "pub fn dimensions(&self) -> (u32, u32)",
            "pub fn pixel_format(&self) -> PixelFormat",
            "pub fn pitch_bytes(&self) -> usize",
            "pub unsafe fn buffer(&self) -> &BufferRef",
            "pub(crate) fn buffer_trusted(&self) -> &BufferRef",
            "pub fn into_buffer(self) -> Buffer",
            "every clone of this tile",
            "No surviving tile offers safe host readback",
            "normal Metal synchronization remains each",
        ])
        .forbidden(&[
            "pub buffer: Buffer",
            "pub byte_offset: usize",
            "pub dimensions: (u32, u32)",
            "pub pixel_format: PixelFormat",
            "pub pitch_bytes: usize",
            "pub status_buffer: Buffer",
            "pub command_buffer: CommandBuffer",
            "pub fn buffer(&self) -> &BufferRef",
        ]),
        PatternCheck::new(
            "JPEG Metal resident private tile regressions",
            &texture_tests,
        )
        .required(&[
            "let raw_buffer = unsafe { tile.buffer() };",
            "let handed_off = tile.clone().into_buffer();",
            "assert_eq!(tile.dimensions(), (16, 16));",
        ]),
    ]);
}
