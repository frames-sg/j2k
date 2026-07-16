// SPDX-License-Identifier: MIT OR Apache-2.0

use std::ffi::OsString;

use crate::process::{self, CommandContext};

use super::{
    semver_cargo_args, PackageApiDiff, ReleaseType, SEMVER_BASELINE_VERSION,
    SOURCE_INCOMPATIBLE_PATCH_EXCEPTION_VERSION,
};

pub(super) fn run_semver_checks(diffs: &[PackageApiDiff]) -> Result<(), String> {
    for diff in diffs.iter().filter(|diff| diff.release_type.is_some()) {
        let args = semver_check_args(diff);
        let args = args.iter().map(String::as_str).collect::<Vec<_>>();
        process::run_command(OsString::from("rustup"), &args, CommandContext::new())?;
    }
    Ok(())
}

pub(super) fn semver_check_args(diff: &PackageApiDiff) -> Vec<String> {
    let release_type = semver_check_release_type(diff);
    semver_cargo_args([
        "semver-checks",
        "check-release",
        "--package",
        diff.package.as_str(),
        "--baseline-version",
        SEMVER_BASELINE_VERSION,
        "--release-type",
        release_type.as_str(),
        "--color",
        "never",
    ])
}

pub(super) fn semver_check_release_type(diff: &PackageApiDiff) -> ReleaseType {
    if diff.candidate_version == SOURCE_INCOMPATIBLE_PATCH_EXCEPTION_VERSION {
        ReleaseType::Major
    } else {
        diff.release_type.expect("published diff release type")
    }
}
