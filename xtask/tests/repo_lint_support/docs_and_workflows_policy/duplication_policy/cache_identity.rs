// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_file_pattern_checks, repo_root, FilePatternCheck};

#[test]
fn jpeg_cache_identity_uses_canonical_digest_boundaries() {
    let root = repo_root();
    let jpeg_context =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/context.rs")).expect("read JPEG context");
    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("crates/j2k-core/src/lib.rs")
                .named("j2k-core FNV-1a helper macros")
                .required(&[
                    "macro_rules! __j2k_fnv1a64_init",
                    "macro_rules! __j2k_fnv1a64_update",
                    "macro_rules! __j2k_fnv1a64_bytes",
                    "0xcbf2_9ce4_8422_2325_u64",
                    "0x0000_0100_0000_01B3_u64",
                ]),
            FilePatternCheck::new("crates/j2k-jpeg/src/context.rs")
                .named("JPEG context FNV helper use")
                .required(&["j2k_core::__j2k_fnv1a64_bytes!(bytes)"])
                .forbidden(&["FNV_OFFSET", "FNV_PRIME"]),
            FilePatternCheck::new("crates/j2k-jpeg-cuda/src/session.rs")
                .named("JPEG CUDA neutral cache session owner")
                .required(&[
                    "mod packet_cache;",
                    "owned_packet_cache: Arc<OwnedPacketPlanCache>",
                ])
                .forbidden(&[
                    "j2k_core::__j2k_fnv1a64_bytes!",
                    "FNV_OFFSET",
                    "FNV_PRIME",
                    "VecDeque",
                ]),
            FilePatternCheck::new("crates/j2k-jpeg-cuda/src/session/packet_cache.rs")
                .named("JPEG CUDA neutral full-input cache use")
                .required(&[
                    "state: Mutex<JpegPlanCache>",
                    "cache.resolve_with_external_live(input, external_live_bytes)",
                    "cache.resolve_from_decoder_with_external_live(decoder, external_live_bytes)",
                    "let result = resolve(&mut cache, active_before);",
                ])
                .forbidden(&[
                    "j2k_core::__j2k_fnv1a64_bytes!",
                    "FNV_OFFSET",
                    "FNV_PRIME",
                    "VecDeque",
                ]),
            FilePatternCheck::new("crates/j2k-jpeg-metal/src/session.rs")
                .named("JPEG Metal neutral full-input cache use")
                .required(&[
                    "jpeg_plans: JpegPlanCache",
                    "self.jpeg_plans\n            .resolve_with_external_live(input, adapter_live_bytes)",
                    "SharedJpegInput::try_copy_from_slice_with_external_live(",
                ])
                .forbidden(&[
                    "j2k_core::__j2k_fnv1a64_bytes!",
                    "FNV_OFFSET",
                    "FNV_PRIME",
                    "VecDeque",
                    "CachedInputAlias",
                ]),
        ],
    );
    assert!(
        jpeg_context.contains("j2k_core::__j2k_fnv1a64_init!()")
            && jpeg_context.contains("j2k_core::__j2k_fnv1a64_update!(hash, byte)"),
        "JPEG table digest continuations must use the shared FNV-1a init/update helpers"
    );
}
