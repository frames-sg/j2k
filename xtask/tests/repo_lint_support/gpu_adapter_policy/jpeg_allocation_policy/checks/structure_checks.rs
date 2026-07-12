// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::JpegAllocationSources;

pub(super) fn assert_policy(sources: &JpegAllocationSources) {
    assert_modules_stay_focused(sources);
    assert_tests_are_explicit_submodules(sources);
}

fn assert_modules_stay_focused(sources: &JpegAllocationSources) {
    for (relative, source, max_lines) in [
        ("internal/checkpoint.rs", sources.checkpoint.as_str(), 130),
        (
            "internal/checkpoint/allocation.rs",
            sources.checkpoint_allocation.as_str(),
            150,
        ),
        (
            "internal/checkpoint/build.rs",
            sources.checkpoint_build.as_str(),
            210,
        ),
        (
            "internal/checkpoint/cache.rs",
            sources.checkpoint_cache.as_str(),
            150,
        ),
        (
            "internal/checkpoint/cache/allocation.rs",
            sources.checkpoint_cache_allocation.as_str(),
            90,
        ),
        (
            "internal/checkpoint/eoi.rs",
            sources.checkpoint_eoi.as_str(),
            150,
        ),
        (
            "internal/checkpoint/planning.rs",
            sources.checkpoint_planning.as_str(),
            90,
        ),
        ("adapter/device_plan.rs", sources.device_plan.as_str(), 325),
        ("adapter/fast_packet.rs", sources.fast_packet.as_str(), 50),
        (
            "adapter/fast_packet/allocation.rs",
            sources.fast_packet_allocation.as_str(),
            130,
        ),
        (
            "adapter/fast_packet/build.rs",
            sources.fast_packet_build_owner.as_str(),
            250,
        ),
        (
            "adapter/fast_packet/build/gray.rs",
            sources.fast_packet_build_gray.as_str(),
            100,
        ),
        (
            "adapter/fast_packet/build/materialization.rs",
            sources.fast_packet_build_materialization.as_str(),
            40,
        ),
        (
            "adapter/fast_packet/checkpoints.rs",
            sources.fast_packet_checkpoints.as_str(),
            175,
        ),
        (
            "adapter/fast_packet/entropy.rs",
            sources.fast_packet_entropy.as_str(),
            250,
        ),
        (
            "adapter/fast_packet/header.rs",
            sources.fast_packet_header.as_str(),
            230,
        ),
        (
            "j2k-jpeg-cuda/src/owned_decode.rs",
            sources.owned_decode.as_str(),
            450,
        ),
        (
            "j2k-jpeg-cuda/src/owned_decode/plan.rs",
            sources.owned_decode_plan.as_str(),
            230,
        ),
        (
            "j2k-jpeg-cuda/src/owned_decode/tests.rs",
            sources.owned_decode_tests.as_str(),
            220,
        ),
    ] {
        assert!(
            source.lines().count() < max_lines,
            "JPEG allocation module {relative} exceeded its focused line-count ratchet"
        );
    }
}

fn assert_tests_are_explicit_submodules(sources: &JpegAllocationSources) {
    for (label, source, modules) in [
        (
            "checkpoint",
            sources.checkpoint_tests.as_str(),
            &["allocation", "build", "cache", "eoi", "fixtures"][..],
        ),
        (
            "fast packet",
            sources.fast_packet_tests.as_str(),
            &["allocation", "behavior", "checkpoints", "entropy", "source"][..],
        ),
    ] {
        assert!(
            !source.contains("#[path"),
            "{label} tests must use real modules"
        );
        for module in modules {
            assert!(
                source.contains(&format!("mod {module};")),
                "{label} tests must retain the {module} submodule"
            );
        }
    }
    assert!(
        sources.checkpoint.contains("#[cfg(test)]\nmod tests;")
            && sources.fast_packet.contains("#[cfg(test)]\nmod tests;")
            && sources
                .owned_decode
                .contains("#[cfg(all(test, feature = \"cuda-runtime\"))]\nmod tests;")
            && !sources.owned_decode_tests.contains("use super::*"),
        "JPEG allocation tests must remain explicit real submodules"
    );
}
