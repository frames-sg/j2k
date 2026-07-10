use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use crate::command_support::{
    command_output_os, ensure_clean_worktree, run_cargo, workspace_version,
};
use crate::process::cargo;

const PUBLISHABLE_PACKAGES: &[&str] = &[
    "j2k-core",
    "j2k-profile",
    "j2k-types",
    "j2k-codec-math",
    "j2k-cuda-runtime",
    "j2k-metal-support",
    "j2k-native",
    "j2k-jpeg",
    "j2k-tilecodec",
    "j2k",
    "j2k-transcode",
    "j2k-transcode-cuda",
    "j2k-jpeg-metal",
    "j2k-metal",
    "j2k-transcode-metal",
    "j2k-jpeg-cuda",
    "j2k-cuda",
    "j2k-cli",
];

const REGISTRY_INDEPENDENT_PACKAGES: &[&str] =
    &["j2k-core", "j2k-profile", "j2k-types", "j2k-codec-math"];

const STAGED_DEPENDENCY_PACKAGES: &[&str] = &[
    "j2k-cuda-runtime",
    "j2k-metal-support",
    "j2k-native",
    "j2k-jpeg",
    "j2k-tilecodec",
    "j2k",
    "j2k-transcode",
    "j2k-transcode-cuda",
    "j2k-jpeg-metal",
    "j2k-metal",
    "j2k-transcode-metal",
    "j2k-jpeg-cuda",
    "j2k-cuda",
    "j2k-cli",
];

const CPU_RELEASE_PACKAGES: &[&str] = &[
    "j2k-core",
    "j2k-codec-math",
    "j2k-jpeg",
    "j2k-types",
    "j2k-native",
    "j2k",
    "j2k-tilecodec",
    "j2k-cli",
];

pub(super) const STABLE_SEMVER_PACKAGES: &[&str] = &[
    "j2k",
    "j2k-core",
    "j2k-codec-math",
    "j2k-jpeg",
    "j2k-tilecodec",
    "j2k-jpeg-metal",
    "j2k-metal",
    "j2k-jpeg-cuda",
    "j2k-cuda",
    "j2k-transcode",
    "j2k-transcode-cuda",
    "j2k-metal-support",
    "j2k-transcode-metal",
    "j2k-native",
    "j2k-types",
    "j2k-cuda-runtime",
    "j2k-profile",
];

pub(super) const STABLE_DOC_LIBRARY_PACKAGES: &[&str] = &[
    "j2k",
    "j2k-core",
    "j2k-codec-math",
    "j2k-jpeg",
    "j2k-tilecodec",
    "j2k-jpeg-metal",
    "j2k-metal",
    "j2k-jpeg-cuda",
    "j2k-cuda",
    "j2k-transcode",
    "j2k-transcode-cuda",
    "j2k-metal-support",
    "j2k-transcode-metal",
    "j2k-native",
    "j2k-types",
    "j2k-cuda-runtime",
    "j2k-profile",
];

#[expect(
    clippy::too_many_lines,
    reason = "release integrity evaluates the complete package policy before reporting aggregated failures"
)]
pub(super) fn release_integrity() -> Result<(), String> {
    let metadata = cargo_metadata()?;
    let workspace_version = workspace_version()?;
    let publishable_set = str_set(PUBLISHABLE_PACKAGES);
    let docs_set = str_set(STABLE_DOC_LIBRARY_PACKAGES);
    let semver_set = str_set(STABLE_SEMVER_PACKAGES);
    let mut errors = Vec::new();

    validate_package_gate_partition(&mut errors);

    let packages = metadata
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "cargo metadata did not contain a packages array".to_string())?;
    let workspace_members = metadata
        .get("workspace_members")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "cargo metadata did not contain a workspace_members array".to_string())?
        .iter()
        .filter_map(serde_json::Value::as_str)
        .map(str::to_string)
        .collect::<BTreeSet<_>>();
    let mut workspace_names = BTreeSet::new();

    let unpublished_members = packages
        .iter()
        .filter(|package| {
            package
                .get("id")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|id| workspace_members.contains(id))
                && publish_false(package)
        })
        .filter_map(|package| package.get("name").and_then(serde_json::Value::as_str))
        .collect::<BTreeSet<_>>();

    for package in packages {
        let id = package
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if !workspace_members.contains(id) {
            continue;
        }

        let name = package_name(package)?;
        workspace_names.insert(name.to_string());
        let listed_publishable = publishable_set.contains(name);
        let explicitly_unpublished = publish_false(package);

        if listed_publishable && explicitly_unpublished {
            errors.push(format!(
                "`{name}` is listed as publishable but has `publish = false`"
            ));
            continue;
        }
        if !listed_publishable && !explicitly_unpublished {
            errors.push(format!(
                "`{name}` is neither in PUBLISHABLE_PACKAGES nor explicitly `publish = false`"
            ));
            continue;
        }
        if !listed_publishable {
            continue;
        }

        validate_unpublished_dependencies(name, package, &unpublished_members, &mut errors);

        let version = package
            .get("version")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if version != workspace_version {
            errors.push(format!(
                "`{name}` version {version} does not match workspace version {workspace_version}"
            ));
        }
        if package
            .get("readme")
            .and_then(serde_json::Value::as_str)
            .is_none()
        {
            errors.push(format!("`{name}` is publishable but has no package README"));
        }
        if !has_docs_rs_metadata(package) {
            errors.push(format!(
                "`{name}` is publishable but missing [package.metadata.docs.rs] with all-features and empty targets"
            ));
        }
        if has_lib_target(package) {
            if !docs_set.contains(name) {
                errors.push(format!(
                    "`{name}` has a library target but is missing from STABLE_DOC_LIBRARY_PACKAGES"
                ));
            }
            if !semver_set.contains(name) {
                errors.push(format!(
                    "`{name}` has a library target but is missing from STABLE_SEMVER_PACKAGES"
                ));
            }
        } else if name != "j2k-cli" {
            errors.push(format!(
                "`{name}` is publishable but has no library target and no explicit release-integrity exemption"
            ));
        }
    }

    for package in PUBLISHABLE_PACKAGES {
        if !workspace_names.contains(*package) {
            errors.push(format!(
                "`{package}` is listed in PUBLISHABLE_PACKAGES but is not a workspace member"
            ));
        }
    }
    for package in STABLE_DOC_LIBRARY_PACKAGES {
        if !publishable_set.contains(package) {
            errors.push(format!(
                "`{package}` is in STABLE_DOC_LIBRARY_PACKAGES but is not publishable"
            ));
        }
    }
    for package in STABLE_SEMVER_PACKAGES {
        if !publishable_set.contains(package) {
            errors.push(format!(
                "`{package}` is in STABLE_SEMVER_PACKAGES but is not publishable"
            ));
        }
    }

    validate_publish_workflow(&mut errors)?;
    validate_publish_script(&mut errors)?;
    validate_release_docs(&mut errors)?;

    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "release integrity violations:\n- {}",
            errors.join("\n- ")
        ))
    }
}

fn validate_unpublished_dependencies(
    name: &str,
    package: &serde_json::Value,
    unpublished_members: &BTreeSet<&str>,
    errors: &mut Vec<String>,
) {
    let dependencies = package
        .get("dependencies")
        .and_then(serde_json::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    for dependency in dependencies {
        let dep_name = dependency
            .get("name")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if !unpublished_members.contains(dep_name) {
            continue;
        }
        let kind = dependency
            .get("kind")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("normal");
        let req = dependency
            .get("req")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("*");
        if kind != "dev" {
            errors.push(format!(
                "`{name}` has a {kind} dependency on unpublished crate `{dep_name}`; \
                 cargo publish cannot resolve it"
            ));
        } else if req != "*" {
            errors.push(format!(
                "`{name}` has a versioned dev-dependency `{dep_name} = \"{req}\"` on an \
                 unpublished crate; drop the version so cargo publish strips the path-only dev-dep"
            ));
        }
    }
}

fn cargo_metadata() -> Result<serde_json::Value, String> {
    let data = command_output_os(
        cargo(),
        &["metadata", "--locked", "--no-deps", "--format-version", "1"],
    )?;
    serde_json::from_str(&data).map_err(|err| format!("failed to parse cargo metadata: {err}"))
}

fn package_name(package: &serde_json::Value) -> Result<&str, String> {
    package
        .get("name")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "cargo metadata package missing name".to_string())
}

fn publish_false(package: &serde_json::Value) -> bool {
    package
        .get("publish")
        .and_then(serde_json::Value::as_array)
        .is_some_and(Vec::is_empty)
}

fn has_lib_target(package: &serde_json::Value) -> bool {
    package
        .get("targets")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|targets| {
            targets.iter().any(|target| {
                target
                    .get("kind")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|kind| {
                        kind.iter()
                            .any(|entry| entry.as_str().is_some_and(|entry| entry == "lib"))
                    })
            })
        })
}

fn has_docs_rs_metadata(package: &serde_json::Value) -> bool {
    let Some(docs_rs) = package
        .get("metadata")
        .and_then(|metadata| metadata.get("docs"))
        .and_then(|docs| docs.get("rs"))
    else {
        return false;
    };

    docs_rs
        .get("all-features")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
        && docs_rs
            .get("targets")
            .and_then(serde_json::Value::as_array)
            .is_some_and(Vec::is_empty)
}

fn validate_publish_workflow(errors: &mut Vec<String>) -> Result<(), String> {
    let workflow_path = Path::new(".github/workflows/publish.yml");
    let workflow_source = fs::read_to_string(workflow_path)
        .map_err(|err| format!("failed to read {}: {err}", workflow_path.display()))?;
    for required in [
        "--origin-url \"${origin_url}\"",
        "--server-url \"${GITHUB_SERVER_URL}\"",
        "scripts/publish-crate.sh --preflight-all",
    ] {
        if !workflow_source.contains(required) {
            errors.push(format!(
                "{} does not enforce publication preflight `{required}`",
                workflow_path.display()
            ));
        }
    }
    let workflow: serde_yaml_ng::Value = serde_yaml_ng::from_str(&workflow_source)
        .map_err(|err| format!("failed to parse {}: {err}", workflow_path.display()))?;
    let mut crates = Vec::new();
    collect_publish_workflow_crates(&workflow, &mut crates);

    let expected = PUBLISHABLE_PACKAGES
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    if crates != expected {
        errors.push(format!(
            "{} publish order is {:?}, expected {:?}",
            workflow_path.display(),
            crates,
            expected
        ));
    }

    let seen = crates.iter().map(String::as_str).collect::<BTreeSet<_>>();
    for package in PUBLISHABLE_PACKAGES {
        if !seen.contains(package) {
            errors.push(format!(
                "{} is missing publish job for `{package}`",
                workflow_path.display()
            ));
        }
    }
    for package in crates {
        if !PUBLISHABLE_PACKAGES.contains(&package.as_str()) {
            errors.push(format!(
                "{} publishes unknown workspace crate `{package}`",
                workflow_path.display()
            ));
        }
    }

    Ok(())
}

fn collect_publish_workflow_crates(value: &serde_yaml_ng::Value, crates: &mut Vec<String>) {
    match value {
        serde_yaml_ng::Value::String(text) => {
            for line in text.lines() {
                if let Some(package) = publish_crate_from_run_line(line) {
                    crates.push(package);
                }
            }
        }
        serde_yaml_ng::Value::Sequence(items) => {
            for item in items {
                collect_publish_workflow_crates(item, crates);
            }
        }
        serde_yaml_ng::Value::Mapping(map) => {
            for value in map.values() {
                collect_publish_workflow_crates(value, crates);
            }
        }
        _ => {}
    }
}

fn publish_crate_from_run_line(line: &str) -> Option<String> {
    let marker = "scripts/publish-crate.sh";
    let after = line.split_once(marker)?.1;
    let argument = after
        .split_whitespace()
        .next()
        .map(|package| package.trim_matches(['"', '\'']).to_string())?;
    if argument == "--preflight-all" {
        None
    } else {
        Some(argument)
    }
}

fn validate_publish_script(errors: &mut Vec<String>) -> Result<(), String> {
    let script_path = Path::new("scripts/publish-crate.sh");
    let script = fs::read_to_string(script_path)
        .map_err(|err| format!("failed to read {}: {err}", script_path.display()))?;
    let crates = shell_array_values(&script, "publishable_crates").ok_or_else(|| {
        format!(
            "{} does not define the publishable_crates shell array",
            script_path.display()
        )
    })?;
    let expected = PUBLISHABLE_PACKAGES
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    if crates != expected {
        errors.push(format!(
            "{} publishable_crates is {:?}, expected {:?}",
            script_path.display(),
            crates,
            expected
        ));
    }

    let independent =
        shell_array_values(&script, "registry_independent_crates").ok_or_else(|| {
            format!(
                "{} does not define the registry_independent_crates shell array",
                script_path.display()
            )
        })?;
    let expected_independent = REGISTRY_INDEPENDENT_PACKAGES
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    if independent != expected_independent {
        errors.push(format!(
            "{} registry_independent_crates is {:?}, expected {:?}",
            script_path.display(),
            independent,
            expected_independent
        ));
    }
    for required in [
        "scripts/crates_io_version.py verify-set",
        "scripts/crates_io_version.py state",
        "cargo package -p \"$crate\" --no-verify",
        "cargo publish -p \"$crate\" --dry-run",
    ] {
        if !script.contains(required) {
            errors.push(format!(
                "{} does not enforce publish-script check `{required}`",
                script_path.display()
            ));
        }
    }
    if script.contains("cargo info") {
        errors.push(format!(
            "{} must not treat ambiguous cargo-info failures as version availability",
            script_path.display()
        ));
    }
    Ok(())
}

fn shell_array_values(script: &str, name: &str) -> Option<Vec<String>> {
    let marker = format!("{name}=(");
    let mut values = Vec::new();
    let mut in_array = false;
    for raw_line in script.lines() {
        let line = raw_line.trim();
        if !in_array {
            if line == marker {
                in_array = true;
            }
            continue;
        }
        if line == ")" {
            return Some(values);
        }
        let line = line.split('#').next()?.trim();
        if line.is_empty() {
            continue;
        }
        values.extend(
            line.split_whitespace()
                .map(|entry| entry.trim_matches(['"', '\'']).to_string()),
        );
    }
    None
}

fn validate_release_docs(errors: &mut Vec<String>) -> Result<(), String> {
    let release_doc_path = Path::new("docs/release.md");
    let release_doc = fs::read_to_string(release_doc_path)
        .map_err(|err| format!("failed to read {}: {err}", release_doc_path.display()))?;
    for package in PUBLISHABLE_PACKAGES {
        if !release_doc.contains(&format!("`{package}`")) {
            errors.push(format!(
                "{} does not document publishable crate `{package}`",
                release_doc_path.display()
            ));
        }
    }
    for required in [
        "cargo xtask release-integrity",
        "cargo package --no-verify",
        "already-published prefix",
        "Only an exact HTTP 404",
        "CRATES_IO_ALLOW_PUBLISHED_RERUN",
        "v<workspace.package.version>",
    ] {
        if !release_doc.contains(required) {
            errors.push(format!(
                "{} does not document `{required}`",
                release_doc_path.display()
            ));
        }
    }
    Ok(())
}

fn str_set(values: &[&'static str]) -> BTreeSet<&'static str> {
    values.iter().copied().collect()
}

fn validate_package_gate_partition(errors: &mut Vec<String>) {
    let publishable = str_set(PUBLISHABLE_PACKAGES);
    let independent = str_set(REGISTRY_INDEPENDENT_PACKAGES);
    let staged = str_set(STAGED_DEPENDENCY_PACKAGES);
    let partitioned = independent.union(&staged).copied().collect::<BTreeSet<_>>();

    if independent.len() != REGISTRY_INDEPENDENT_PACKAGES.len() {
        errors.push("REGISTRY_INDEPENDENT_PACKAGES contains duplicates".to_string());
    }
    if staged.len() != STAGED_DEPENDENCY_PACKAGES.len() {
        errors.push("STAGED_DEPENDENCY_PACKAGES contains duplicates".to_string());
    }
    for package in independent.intersection(&staged) {
        errors.push(format!(
            "`{package}` appears in both package-gate partitions"
        ));
    }
    for package in publishable.difference(&partitioned) {
        errors.push(format!(
            "publishable package `{package}` is missing from the package-gate partitions"
        ));
    }
    for package in &partitioned {
        if !publishable.contains(package) {
            errors.push(format!(
                "package-gate partition contains non-publishable package `{package}`"
            ));
        }
    }
}

pub(super) fn release_cpu() -> Result<(), String> {
    let mut args = vec!["test", "--release"];
    for package in CPU_RELEASE_PACKAGES {
        args.push("-p");
        args.push(package);
    }
    run_cargo(&args)
}

pub(super) fn package() -> Result<(), String> {
    ensure_clean_worktree()?;
    for package in PUBLISHABLE_PACKAGES {
        run_cargo(&["package", "-p", package, "--list"])?;
    }
    for package in STAGED_DEPENDENCY_PACKAGES {
        run_cargo(&["package", "-p", package, "--no-verify"])?;
    }
    for package in REGISTRY_INDEPENDENT_PACKAGES {
        run_cargo(&["publish", "-p", package, "--dry-run"])?;
    }
    Ok(())
}
