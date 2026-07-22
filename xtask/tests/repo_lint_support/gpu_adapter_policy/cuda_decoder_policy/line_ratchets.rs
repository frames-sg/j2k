// SPDX-License-Identifier: MIT OR Apache-2.0

use super::CudaDecoderSources;

#[test]
fn focused_modules_stay_below_line_ratchets() {
    let sources = CudaDecoderSources::read();
    assert_facade_and_plan_line_ratchets(&sources);
    assert_resident_line_ratchets(&sources);
    assert_color_line_ratchets(&sources);
    assert_grayscale_line_ratchets(&sources);
}

fn assert_grayscale_line_ratchets(sources: &CudaDecoderSources) {
    for (module_name, source, maximum_lines) in [
        ("grayscale_batch.rs", &sources.grayscale_batch, 225),
        (
            "grayscale_batch/completion.rs",
            &sources.grayscale_batch_completion,
            300,
        ),
        (
            "grayscale_batch/execution.rs",
            &sources.grayscale_batch_execution,
            400,
        ),
        (
            "grayscale_batch/preparation.rs",
            &sources.grayscale_batch_preparation,
            250,
        ),
        (
            "grayscale_batch/store.rs",
            &sources.grayscale_batch_store,
            350,
        ),
        (
            "grayscale_batch/tests.rs",
            &sources.grayscale_batch_tests,
            250,
        ),
    ] {
        assert!(
            source.lines().count() < maximum_lines,
            "j2k-cuda/src/decoder/{module_name} must stay below its focused-module line-count ratchet"
        );
    }
}

fn assert_facade_and_plan_line_ratchets(sources: &CudaDecoderSources) {
    assert!(
        sources.decoder.lines().count() < 1_500,
        "j2k-cuda/src/decoder.rs must stay below the post-runtime-split line-count ratchet"
    );
    for (module_name, source, maximum_lines) in [
        ("api.rs", &sources.api, 1_800),
        ("plan.rs", &sources.plan, 75),
        ("plan/grayscale.rs", &sources.plan_grayscale, 475),
        ("plan/color.rs", &sources.plan_color, 200),
        ("plan/color_decoder.rs", &sources.plan_color_decoder, 275),
        ("profile.rs", &sources.profile, 1_800),
    ] {
        assert!(
            source.lines().count() < maximum_lines,
            "j2k-cuda/src/decoder/{module_name} must stay below the focused-module line-count ratchet"
        );
    }
    assert!(
        sources.plan_color_owners.lines().count() < 100,
        "j2k-cuda/src/decoder/plan/color_owners.rs must remain a focused owner-accounting leaf"
    );
}

fn assert_resident_line_ratchets(sources: &CudaDecoderSources) {
    for (module_name, source, maximum_lines) in [
        ("resident.rs", &sources.resident, 50),
        (
            "resident/cleanup_dequant.rs",
            &sources.resident_cleanup_dequant,
            325,
        ),
        ("resident/component.rs", &sources.resident_component, 225),
        (
            "resident/buffer_access.rs",
            &sources.resident_buffer_access,
            50,
        ),
        ("resident/error.rs", &sources.resident_error, 50),
        ("resident/idwt.rs", &sources.resident_idwt, 350),
        (
            "resident/idwt/conversions.rs",
            &sources.resident_idwt_conversions,
            75,
        ),
        ("resident/routing.rs", &sources.resident_routing, 425),
        ("resident/surface.rs", &sources.resident_surface, 175),
    ] {
        assert!(
            source.lines().count() < maximum_lines,
            "j2k-cuda/src/decoder/{module_name} must stay below its semantic-module line-count ratchet"
        );
    }
}

fn assert_color_line_ratchets(sources: &CudaDecoderSources) {
    assert!(
        sources.color_batch.lines().count() < 100,
        "j2k-cuda decoder/color_batch.rs must remain a facade"
    );
    for (module_name, source, maximum_lines) in [
        (
            "color_batch/batch_execution.rs",
            &sources.color_batch_execution,
            75,
        ),
        (
            "color_batch/batch_execution/preparation.rs",
            &sources.color_batch_execution_preparation,
            75,
        ),
        (
            "color_batch/batch_execution/execution.rs",
            &sources.color_batch_execution_execution,
            175,
        ),
        (
            "color_batch/batch_execution/completion.rs",
            &sources.color_batch_execution_completion,
            100,
        ),
        (
            "color_batch/batch_execution/completion/batch_store.rs",
            &sources.color_batch_execution_completion_batch_store,
            250,
        ),
        (
            "color_batch/batch_execution/completion/fallback.rs",
            &sources.color_batch_execution_completion_fallback,
            75,
        ),
        ("color_batch/single.rs", &sources.color_batch_single, 200),
        ("color_batch/finish.rs", &sources.color_batch_finish, 125),
        (
            "color_batch/finish/component.rs",
            &sources.color_batch_finish_component,
            100,
        ),
        (
            "color_batch/finish/surface.rs",
            &sources.color_batch_finish_surface,
            75,
        ),
        (
            "color_batch/native_batch.rs",
            &sources.color_batch_native,
            325,
        ),
        (
            "color_batch/native_batch/completion.rs",
            &sources.color_batch_native_completion,
            125,
        ),
        (
            "color_batch/native_batch/execution.rs",
            &sources.color_batch_native_execution,
            275,
        ),
        (
            "color_batch/native_batch/prepare.rs",
            &sources.color_batch_native_prepare,
            150,
        ),
    ] {
        assert!(
            source.lines().count() < maximum_lines,
            "j2k-cuda/src/decoder/{module_name} must stay below its focused-module line-count ratchet"
        );
    }
    assert!(
        sources.color_batch_host_owners.lines().count() < 125,
        "j2k-cuda decoder/color_batch/host_owners.rs must remain a focused owner-accounting leaf"
    );
    assert!(
        sources.color_store.lines().count() < 500,
        "j2k-cuda decoder/color_batch/store.rs must stay below its focused-module line-count ratchet"
    );
    assert!(
        sources.color_store_batch.lines().count() < 150,
        "j2k-cuda decoder/color_batch/store/batch.rs must remain a focused preparation leaf"
    );
    assert!(
        sources.color_store_validation.lines().count() < 100,
        "j2k-cuda decoder/color_batch/store/validation.rs must remain a focused validation leaf"
    );
}
