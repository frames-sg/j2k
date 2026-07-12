// SPDX-License-Identifier: MIT OR Apache-2.0

//! Deterministic collection of the ordinary and rustdoc-hidden public API.

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
};

use crate::command_support::command_output_os_detailed_with_env;

pub(super) const PUBLIC_API_SNAPSHOT: &str = "docs/stable-api-1.0.public-api.txt";
pub(super) const HIDDEN_API_SNAPSHOT: &str = "docs/stable-api-1.0.implementation-public-api.txt";
pub(super) const CARGO_PUBLIC_API_VERSION: &str = "0.52.0";
pub(super) const PUBLIC_API_TOOLCHAIN: &str = "nightly-2026-06-28";
pub(super) const PUBLIC_API_TARGET: &str = "aarch64-apple-darwin";
pub(super) const ORDINARY_RUSTDOCFLAGS: &str = "-D warnings";
const HIDDEN_RUSTDOCFLAGS: &str = "-D warnings --document-hidden-items";

const FORBIDDEN_ENV_VARS: &[&str] = &[
    "CARGO_BUILD_TARGET",
    "CARGO_ENCODED_RUSTDOCFLAGS",
    "CARGO_ENCODED_RUSTFLAGS",
    "DOCS_RS",
    "MACOSX_DEPLOYMENT_TARGET",
    "RUSTC",
    "RUSTC_BOOTSTRAP",
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
    "RUSTDOC",
    "RUSTDOCFLAGS",
    "RUSTFLAGS",
];

#[derive(Debug, Eq, PartialEq)]
pub(super) struct PackageApiInventory {
    pub(super) ordinary: BTreeSet<String>,
    pub(super) hidden: BTreeSet<String>,
}

pub(super) fn verify_cargo_public_api_version() -> Result<(), String> {
    validate_public_api_environment()?;
    let args = public_api_cargo_args(["public-api", "--version"]);
    let args = args.iter().map(String::as_str).collect::<Vec<_>>();
    let output =
        command_output_os_detailed_with_env("rustup".into(), &args, &[]).map_err(|error| {
            format!(
                "failed to detect cargo-public-api with {PUBLIC_API_TOOLCHAIN}: {error}; \
             install cargo-public-api with `cargo install cargo-public-api --version \
             {CARGO_PUBLIC_API_VERSION} --locked` and install {PUBLIC_API_TOOLCHAIN}"
            )
        })?;
    validate_cargo_public_api_version_output(&output)
}

pub(super) fn collect_package_apis(
    packages: &[&str],
) -> Result<BTreeMap<String, PackageApiInventory>, String> {
    validate_public_api_environment()?;
    let mut inventories = BTreeMap::new();
    for package in packages {
        eprintln!("collecting ordinary and rustdoc-hidden public API for `{package}`");
        let inventory = collect_package_api(package)?;
        if inventories
            .insert((*package).to_string(), inventory)
            .is_some()
        {
            return Err(format!(
                "public API package list contains duplicate package `{package}`"
            ));
        }
    }
    if inventories.is_empty() {
        return Err("public API package list is empty".to_string());
    }
    Ok(inventories)
}

pub(super) fn validate_public_api_environment() -> Result<(), String> {
    validate_public_api_environment_keys(
        env::vars_os().map(|(key, _)| key.to_string_lossy().into_owned()),
    )
}

fn collect_package_api(package: &str) -> Result<PackageApiInventory, String> {
    let ordinary = package_public_api(package, ORDINARY_RUSTDOCFLAGS)?;
    let hidden_enabled = package_public_api(package, HIDDEN_RUSTDOCFLAGS)?;
    split_hidden_api(package, ordinary, &hidden_enabled)
}

fn split_hidden_api(
    package: &str,
    ordinary: BTreeSet<String>,
    hidden_enabled: &BTreeSet<String>,
) -> Result<PackageApiInventory, String> {
    if ordinary.is_empty() {
        return Err(format!(
            "ordinary cargo-public-api pass returned no public items for stable package `{package}`"
        ));
    }
    if hidden_enabled.is_empty() {
        return Err(format!(
            "rustdoc-hidden cargo-public-api pass returned no public items for stable package `{package}`"
        ));
    }

    let complete = ordinary
        .union(hidden_enabled)
        .cloned()
        .collect::<BTreeSet<_>>();
    Ok(PackageApiInventory {
        hidden: complete.difference(&ordinary).cloned().collect(),
        ordinary,
    })
}

fn package_public_api(package: &str, rustdoc_flags: &str) -> Result<BTreeSet<String>, String> {
    let args = public_api_cargo_args([
        "public-api",
        "-p",
        package,
        "--all-features",
        "-sss",
        "--color",
        "never",
        "--target",
        PUBLIC_API_TARGET,
    ]);
    let args = args.iter().map(String::as_str).collect::<Vec<_>>();
    let output = command_output_os_detailed_with_env(
        "rustup".into(),
        &args,
        &[("RUSTDOCFLAGS", rustdoc_flags), ("RUSTFLAGS", "")],
    )
    .map_err(|error| {
        format!(
            "failed to generate public API for {package} with toolchain \
             {PUBLIC_API_TOOLCHAIN}, target {PUBLIC_API_TARGET}, and \
             RUSTDOCFLAGS=`{rustdoc_flags}`: {error}"
        )
    })?;
    Ok(public_api_line_set(&output))
}

fn public_api_cargo_args<'a>(args: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    ["run", PUBLIC_API_TOOLCHAIN, "cargo"]
        .into_iter()
        .chain(args)
        .map(str::to_string)
        .collect()
}

fn validate_cargo_public_api_version_output(output: &str) -> Result<(), String> {
    let expected = format!("cargo-public-api {CARGO_PUBLIC_API_VERSION}");
    if output == expected {
        Ok(())
    } else {
        Err(format!(
            "cargo-public-api version must be {CARGO_PUBLIC_API_VERSION}; found `{output}`"
        ))
    }
}

fn validate_public_api_environment_keys<I, S>(keys: I) -> Result<(), String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let conflicts = keys
        .into_iter()
        .filter_map(|key| {
            let key = key.as_ref();
            (FORBIDDEN_ENV_VARS.contains(&key)
                || key.starts_with("CARGO_TARGET_") && key.ends_with("_RUSTFLAGS"))
            .then(|| key.to_string())
        })
        .collect::<BTreeSet<_>>();
    if conflicts.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "stable public API generation refuses API-affecting environment overrides: \
             {conflicts:?}"
        ))
    }
}

fn public_api_line_set(output: &str) -> BTreeSet<String> {
    output
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        public_api_cargo_args, public_api_line_set, split_hidden_api,
        validate_cargo_public_api_version_output, validate_public_api_environment_keys,
        CARGO_PUBLIC_API_VERSION, HIDDEN_API_SNAPSHOT, PUBLIC_API_SNAPSHOT, PUBLIC_API_TARGET,
        PUBLIC_API_TOOLCHAIN,
    };

    #[test]
    fn snapshot_paths_and_generation_inputs_are_pinned() {
        assert_ne!(PUBLIC_API_SNAPSHOT, HIDDEN_API_SNAPSHOT);
        assert_eq!(CARGO_PUBLIC_API_VERSION, "0.52.0");
        assert_eq!(PUBLIC_API_TOOLCHAIN, "nightly-2026-06-28");
        assert_eq!(PUBLIC_API_TARGET, "aarch64-apple-darwin");
    }

    #[test]
    fn public_api_commands_use_the_pinned_rustup_toolchain() {
        assert_eq!(
            public_api_cargo_args(["public-api", "--version"]),
            [
                "run",
                "nightly-2026-06-28",
                "cargo",
                "public-api",
                "--version",
            ]
        );
    }

    #[test]
    fn public_api_lines_are_sorted_and_deduplicated() {
        let output = "pub fn demo::b()  \npub fn demo::a()\npub fn demo::a()\n";
        assert_eq!(
            public_api_line_set(output).into_iter().collect::<Vec<_>>(),
            ["pub fn demo::a()", "pub fn demo::b()"]
        );
    }

    #[test]
    fn hidden_inventory_preserves_ordinary_items_when_rustdoc_rewrites_paths() {
        let ordinary = public_api_line_set("pub fn demo::b()\npub fn demo::a()\n");
        let hidden_enabled = public_api_line_set(
            "pub fn demo::hidden_z()\npub fn demo::b()\npub fn demo::hidden_a()\n\
             pub fn demo::a()\n",
        );
        let inventory = split_hidden_api("demo", ordinary, &hidden_enabled).unwrap();
        assert_eq!(
            inventory.hidden.into_iter().collect::<Vec<_>>(),
            ["pub fn demo::hidden_a()", "pub fn demo::hidden_z()"]
        );

        let ordinary =
            public_api_line_set("pub demo::Report::device: crate::Device\npub use demo::Device\n");
        let rewritten = public_api_line_set(
            "pub demo::Report::device: demo::Device\npub struct demo::Device\n\
             pub struct demo::adapter::HiddenPlan\n",
        );
        let inventory = split_hidden_api("demo", ordinary, &rewritten).unwrap();
        assert_eq!(
            inventory.ordinary.into_iter().collect::<Vec<_>>(),
            [
                "pub demo::Report::device: crate::Device",
                "pub use demo::Device",
            ]
        );
        assert_eq!(
            inventory.hidden.into_iter().collect::<Vec<_>>(),
            [
                "pub demo::Report::device: demo::Device",
                "pub struct demo::Device",
                "pub struct demo::adapter::HiddenPlan",
            ]
        );
    }

    #[test]
    fn cargo_public_api_version_match_is_exact() {
        assert!(validate_cargo_public_api_version_output("cargo-public-api 0.52.0").is_ok());
        for invalid in [
            "cargo-public-api 10.52.0",
            "wrapper cargo-public-api 0.52.0",
            "cargo-public-api 0.52.0-dev",
        ] {
            assert!(validate_cargo_public_api_version_output(invalid).is_err());
        }
    }

    #[test]
    fn environment_guard_allows_rustup_selection_but_rejects_api_overrides() {
        assert!(validate_public_api_environment_keys(["PATH", "HOME", "RUSTUP_TOOLCHAIN"]).is_ok());
        for key in [
            "CARGO_BUILD_TARGET",
            "CARGO_ENCODED_RUSTDOCFLAGS",
            "CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS",
            "DOCS_RS",
            "MACOSX_DEPLOYMENT_TARGET",
            "RUSTC_BOOTSTRAP",
            "RUSTDOCFLAGS",
        ] {
            assert!(
                validate_public_api_environment_keys([key]).is_err(),
                "{key}"
            );
        }
    }
}
