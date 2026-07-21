// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, fs};

use super::super::{
    assert_file_pattern_checks, assert_pattern_checks, read_source_files, repo_root, rust_sources,
    FilePatternCheck, PatternCheck,
};

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the cross-module Metal consistency checks are one public-contract policy"
)]
fn metal_consistency_cleanup_keeps_names_status_buffers_and_marker_sizes_single_sourced() {
    let root = repo_root();
    let buffer_validation =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/buffer_validation.rs"))
            .expect("read buffer validation");
    let decode_dispatch =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/decode_dispatch.rs"))
            .expect("read decode dispatch");
    let lossless_prepare =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/lossless_prepare.rs"))
            .expect("read lossless prepare");
    let tier1_encode =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/tier1_encode.rs"))
            .expect("read tier1 encode");
    let resident_codestream =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/resident_codestream.rs"))
            .expect("read resident codestream");
    let resident_tier1 =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/resident_tier1.rs"))
            .expect("read resident tier1");
    let resident_tier1_types =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/resident_tier1/types.rs"))
            .expect("read resident tier1 types");
    let direct_buffers =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_buffers.rs"))
            .expect("read direct buffers");
    let direct_roi = fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_roi.rs"))
        .expect("read direct ROI");
    let resident_types =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/resident_types.rs"))
            .expect("read resident types");
    let resident_packet_plan =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/resident_packet_plan.rs"))
            .expect("read resident packet plan");
    let encode_capacity =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/encode_capacity.rs"))
            .expect("read encode capacity");
    let jpeg_extended12 = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/decoder/extended12.rs",
            "crates/j2k-jpeg/src/decoder/extended12/upsample.rs",
        ],
    );
    let split_metal_status_users = [
        "crates/j2k-metal/src/compute/decode_dispatch/classic_cleanup.rs",
        "crates/j2k-metal/src/compute/decode_dispatch/classic_subband.rs",
        "crates/j2k-metal/src/compute/decode_dispatch/ht_distinct.rs",
        "crates/j2k-metal/src/compute/decode_dispatch/ht_subband.rs",
        "crates/j2k-metal/src/compute/decode_dispatch/idwt.rs",
        "crates/j2k-metal/src/compute/decode_dispatch/mct.rs",
        "crates/j2k-metal/src/compute/decode_dispatch/store.rs",
        "crates/j2k-metal/src/compute/lossless_prepare/batch.rs",
        "crates/j2k-metal/src/compute/lossless_prepare/batch_item.rs",
        "crates/j2k-metal/src/compute/lossless_prepare/commands.rs",
        "crates/j2k-metal/src/compute/lossless_prepare/forward_encode.rs",
        "crates/j2k-metal/src/compute/lossless_prepare/single.rs",
        "crates/j2k-metal/src/compute/lossless_prepare/sizes.rs",
        "crates/j2k-metal/src/compute/resident_tier1/profile_dispatch/analysis.rs",
        "crates/j2k-metal/src/compute/resident_tier1/profile_dispatch/tokens.rs",
        "crates/j2k-metal/src/compute/resident_tier1/readback.rs",
        "crates/j2k-metal/src/compute/resident_tier1/result_harvest.rs",
    ]
    .into_iter()
    .map(|relative| {
        fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"))
    })
    .collect::<Vec<_>>()
    .join("\n");
    let metal_status_users = [
        buffer_validation.as_str(),
        decode_dispatch.as_str(),
        lossless_prepare.as_str(),
        tier1_encode.as_str(),
        resident_codestream.as_str(),
        resident_tier1.as_str(),
        split_metal_status_users.as_str(),
    ]
    .join("\n");

    assert_pattern_checks(&[
        PatternCheck::new(
            "Metal resident tier1 component-count field",
            &resident_tier1_types,
        )
        .required(&["pub(crate) component_count: u8"])
        .forbidden(&["pub(crate) components: u8", "pub(crate) num_components: u8"]),
        PatternCheck::new(
            "Metal resident types component-count field",
            &resident_types,
        )
        .required(&["pub(crate) component_count: u8"])
        .forbidden(&["pub(crate) num_components: u8"]),
        PatternCheck::new(
            "Metal resident packet plan component-count field",
            &resident_packet_plan,
        )
        .required(&["pub(super) component_count: u8"])
        .forbidden(&["pub(super) num_components: u8"]),
        PatternCheck::new("Metal direct buffer helper", &direct_buffers)
            .required(&["pub(super) fn zeroed_shared_buffer", "early-returning"]),
        PatternCheck::new("Metal status readback buffer users", &metal_status_users)
            .required(&["zeroed_shared_buffer(&runtime.device"])
            .forbidden(&[
                "let status_buffer = runtime.device.new_buffer(",
                "let status_buffer = runtime.device.new_buffer_with_data(",
                "let status_buffer = runtime\n                .device\n                .new_buffer",
            ]),
        PatternCheck::new(
            "codestream capacity marker-size constants",
            &encode_capacity,
        )
        .required(&[
            "JP2K_SIZ_FIXED_BYTES",
            "JP2K_SIZ_BYTES_PER_COMPONENT",
            "JP2K_CAP_MARKER_SEGMENT_BYTES",
            "JP2K_COD_MARKER_SEGMENT_BYTES",
            "JP2K_QCD_FIXED_BYTES",
            "JP2K_TLM_MARKER_SEGMENT_BYTES",
            "JP2K_SOT_MARKER_SEGMENT_BYTES",
            "JP2K_SOD_MARKER_BYTES",
            "JP2K_EOC_MARKER_BYTES",
        ])
        .forbidden(&[
            "40usize",
            "len.checked_add(14)",
            "if job.write_tlm { 12",
            "len.checked_add(12)",
            "len.checked_add(2)",
        ]),
        PatternCheck::new("IDWT margin explanations", &direct_roi)
            .required(&["16 samples", "40 for irreversible 9/7"]),
        PatternCheck::new(
            "extended 12-bit fancy upsample rounding explanation",
            &jpeg_extended12,
        )
        .required(&["IJG/libjpeg fancy h2v2 upsampling"]),
    ]);
}

#[test]
fn metal_raw_buffer_contents_access_stays_confined_to_checked_helpers() {
    let root = repo_root();
    let allowed = BTreeSet::from([
        "crates/j2k-metal-support/src/buffer_access.rs",
        "crates/j2k-metal/src/compute/direct_buffers.rs",
        "crates/j2k-jpeg-metal/src/buffers.rs",
    ]);

    for src_root in [
        "crates/j2k-metal-support/src",
        "crates/j2k-metal/src",
        "crates/j2k-jpeg-metal/src",
        "crates/j2k-transcode-metal/src",
    ] {
        for path in rust_sources(&root.join(src_root)) {
            let rel = path
                .strip_prefix(root)
                .expect("source path under repo root")
                .to_string_lossy()
                .replace('\\', "/");
            let source =
                fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {rel}: {err}"));
            if allowed.contains(rel.as_str()) {
                continue;
            }
            assert!(
                !source.contains(".contents()"),
                "raw Metal buffer contents access must stay inside checked helpers; found in {rel}"
            );
        }
    }
}

#[test]
fn j2k_metal_bench_surface_stays_clean_after_reset() {
    let root = repo_root();
    let removed_j2k_metal_bench_command = ["cargo bench -p ", "j2k-metal", " --bench"].concat();
    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("crates/j2k-metal/Cargo.toml")
                .named("J2K Metal manifest")
                .forbidden(&[
                    "[[bench]]",
                    "criterion =",
                    "j2k-compare =",
                    "name = \"device_upload\"",
                    "name = \"compare\"",
                    "name = \"encode_stages\"",
                    "name = \"decode_stages\"",
                ]),
            FilePatternCheck::new("README.md")
                .forbidden(&[removed_j2k_metal_bench_command.as_str()]),
            FilePatternCheck::new("xtask/src/main.rs")
                .forbidden(&[removed_j2k_metal_bench_command.as_str()]),
            FilePatternCheck::new("crates/j2k-compare/src/openjpeg.rs")
                .named("OpenJPEG comparator")
                .required(&["pub fn version"]),
            FilePatternCheck::new("crates/j2k-compare/src/grok.rs")
                .named("Grok comparator")
                .required(&["pub fn version", "pub fn library_path"]),
        ],
    );

    let benches_dir = root.join("crates/j2k-metal/benches");
    if benches_dir.exists() {
        let stale_entries: Vec<_> = fs::read_dir(&benches_dir)
            .expect("read J2K Metal benches dir")
            .map(|entry| {
                let path = entry.expect("read J2K Metal bench entry").path();
                path.strip_prefix(root)
                    .expect("bench entry under repo root")
                    .display()
                    .to_string()
            })
            .collect();
        assert!(
            stale_entries.is_empty(),
            "j2k-metal benches dir must stay empty after reset: {stale_entries:?}"
        );
    }
}
