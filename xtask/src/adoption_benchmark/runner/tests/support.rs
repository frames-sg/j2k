// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

pub(super) fn temp_dir(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "j2k-adoption-runner-{label}-{}-{}",
        std::process::id(),
        NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).expect("create runner test directory");
    path
}

pub(super) fn recording_program(root: &Path) -> PathBuf {
    let path = root.join("record-command.sh");
    fs::write(
        &path,
        r#"#!/bin/sh
for argument in "$@"; do
    printf 'arg=%s\n' "$argument"
done
printf 'CUSTOM=%s\n' "${CUSTOM-unset}"
printf 'CARGO_TARGET_DIR=%s\n' "${CARGO_TARGET_DIR-unset}"
printf 'J2K_FIXTURE_COMPARE_REPEATS=%s\n' "${J2K_FIXTURE_COMPARE_REPEATS-unset}"
printf 'J2K_FIXTURE_COMPARE_INPUT_DIRS=%s\n' "${J2K_FIXTURE_COMPARE_INPUT_DIRS-unset}"
printf 'J2K_FIXTURE_COMPARE_MANIFEST=%s\n' "${J2K_FIXTURE_COMPARE_MANIFEST-unset}"
printf 'J2K_FIXTURE_COMPARE_INCLUDE_GENERATED=%s\n' "${J2K_FIXTURE_COMPARE_INCLUDE_GENERATED-unset}"
printf 'J2K_INCLUDE_OPENJPH=%s\n' "${J2K_INCLUDE_OPENJPH-unset}"
printf 'J2K_REQUIRE_OPENJPH=%s\n' "${J2K_REQUIRE_OPENJPH-unset}"
printf 'J2K_INCLUDE_KAKADU=%s\n' "${J2K_INCLUDE_KAKADU-unset}"
printf 'J2K_REQUIRE_KAKADU=%s\n' "${J2K_REQUIRE_KAKADU-unset}"
printf 'J2K_ENCODE_COMPARE_REPEATS=%s\n' "${J2K_ENCODE_COMPARE_REPEATS-unset}"
printf 'J2K_ENCODE_COMPARE_INPUT_DIRS=%s\n' "${J2K_ENCODE_COMPARE_INPUT_DIRS-unset}"
printf 'J2K_ENCODE_COMPARE_MANIFEST=%s\n' "${J2K_ENCODE_COMPARE_MANIFEST-unset}"
printf 'J2K_ENCODE_COMPARE_INCLUDE_GENERATED=%s\n' "${J2K_ENCODE_COMPARE_INCLUDE_GENERATED-unset}"
printf 'J2K_CUDA_DECODE_BATCH_SIZES=%s\n' "${J2K_CUDA_DECODE_BATCH_SIZES-unset}"
printf 'J2K_CUDA_DECODE_INCLUDE_GENERATED=%s\n' "${J2K_CUDA_DECODE_INCLUDE_GENERATED-unset}"
printf 'J2K_CUDA_ENCODE_INPUT_DIRS=%s\n' "${J2K_CUDA_ENCODE_INPUT_DIRS-unset}"
printf 'J2K_CUDA_ENCODE_INCLUDE_GENERATED=%s\n' "${J2K_CUDA_ENCODE_INCLUDE_GENERATED-unset}"
printf 'J2K_REQUIRE_CUDA_BENCH=%s\n' "${J2K_REQUIRE_CUDA_BENCH-unset}"
printf 'J2K_REQUIRE_CUDA_OXIDE_BUILD=%s\n' "${J2K_REQUIRE_CUDA_OXIDE_BUILD-unset}"
printf 'J2K_REQUIRE_METAL_BENCH=%s\n' "${J2K_REQUIRE_METAL_BENCH-unset}"
printf 'J2K_METAL_DECODE_INCLUDE_GENERATED=%s\n' "${J2K_METAL_DECODE_INCLUDE_GENERATED-unset}"
printf 'J2K_METAL_ENCODE_INCLUDE_GENERATED=%s\n' "${J2K_METAL_ENCODE_INCLUDE_GENERATED-unset}"
printf 'J2K_TRANSCODE_METAL_PROFILE_STAGES=%s\n' "${J2K_TRANSCODE_METAL_PROFILE_STAGES-unset}"
printf 'stderr-custom=%s\n' "${CUSTOM-unset}" >&2
exit "${EXIT_CODE-0}"
"#,
    )
    .expect("write recording program");
    let mut permissions = fs::metadata(&path)
        .expect("recording program metadata")
        .permissions();
    permissions.set_mode(0o700);
    fs::set_permissions(&path, permissions).expect("make recording program executable");
    path
}

pub(super) fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}
