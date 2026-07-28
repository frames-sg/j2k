// SPDX-License-Identifier: MIT OR Apache-2.0

//! Review-schema parsing and exact fingerprint validation.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
};

use super::{
    PackageApiDiff, API_DIFF_REPORT, API_REVIEW_CONFIG, SEMVER_BASELINE_TAG,
    SEMVER_BASELINE_VERSION,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BreakKind {
    Source,
    Behavior,
}

#[derive(Debug)]
pub(super) struct RemovedApiItem {
    pub(super) package: String,
    pub(super) item: String,
}

#[derive(Debug)]
pub(super) struct BreakLedgerEntry {
    pub(super) id: String,
    pub(super) kind: BreakKind,
    pub(super) packages: BTreeSet<String>,
    pub(super) removed_items: Vec<RemovedApiItem>,
    pub(super) summary: String,
    pub(super) migration: String,
}

#[derive(Debug)]
pub(super) struct ReviewEntry {
    pub(super) removed_fingerprint: String,
    pub(super) added_fingerprint: String,
    pub(super) hidden_count: usize,
    pub(super) hidden_fingerprint: String,
    pub(super) rationale: String,
    pub(super) hidden_rationale: Option<String>,
}

#[derive(Debug)]
pub(super) struct ReviewConfig {
    pub(super) candidate_version: String,
    pub(super) break_ledger: Vec<BreakLedgerEntry>,
    pub(super) reviews: BTreeMap<String, ReviewEntry>,
}

pub(super) fn load_review_config() -> Result<ReviewConfig, String> {
    let source = fs::read_to_string(API_REVIEW_CONFIG)
        .map_err(|err| format!("read {API_REVIEW_CONFIG}: {err}"))?;
    let value: serde_yaml_ng::Value = serde_yaml_ng::from_str(&source)
        .map_err(|err| format!("parse {API_REVIEW_CONFIG}: {err}"))?;
    parse_review_config(&value)
}

pub(super) fn parse_review_config(value: &serde_yaml_ng::Value) -> Result<ReviewConfig, String> {
    let root = value
        .as_mapping()
        .ok_or_else(|| "API review config root must be a mapping".to_string())?;
    reject_unknown_keys(
        root,
        &[
            "version",
            "baseline_tag",
            "baseline_version",
            "candidate_version",
            "break_ledger",
            "reviews",
        ],
        "API review config",
    )?;
    if required_u64(root, "version")? != 3 {
        return Err("API review config version must be 3".to_string());
    }
    require_exact(root, "baseline_tag", SEMVER_BASELINE_TAG)?;
    require_exact(root, "baseline_version", SEMVER_BASELINE_VERSION)?;
    let candidate_version = required_string(root, "candidate_version")?.to_string();
    let break_values = required_value(root, "break_ledger")?
        .as_sequence()
        .ok_or_else(|| "API review config `break_ledger` must be a sequence".to_string())?;
    let mut break_ledger = Vec::with_capacity(break_values.len());
    let mut break_ids = BTreeSet::new();
    for value in break_values {
        let entry = value
            .as_mapping()
            .ok_or_else(|| "each API break-ledger entry must be a mapping".to_string())?;
        let entry = parse_break_ledger_entry(entry)?;
        if !break_ids.insert(entry.id.clone()) {
            return Err(format!("duplicate break-ledger id `{}`", entry.id));
        }
        break_ledger.push(entry);
    }
    let review_values = required_value(root, "reviews")?
        .as_mapping()
        .ok_or_else(|| "API review config `reviews` must be a mapping".to_string())?;
    let mut reviews = BTreeMap::new();
    for (package, value) in review_values {
        let package = package
            .as_str()
            .ok_or_else(|| "API review package keys must be strings".to_string())?;
        let entry = value
            .as_mapping()
            .ok_or_else(|| format!("API review for `{package}` must be a mapping"))?;
        let review = parse_review_entry(package, entry)?;
        if reviews.insert(package.to_string(), review).is_some() {
            return Err(format!("duplicate API review for `{package}`"));
        }
    }
    Ok(ReviewConfig {
        candidate_version,
        break_ledger,
        reviews,
    })
}

fn parse_break_ledger_entry(entry: &serde_yaml_ng::Mapping) -> Result<BreakLedgerEntry, String> {
    reject_unknown_keys(
        entry,
        &[
            "id",
            "kind",
            "packages",
            "removed_items",
            "summary",
            "migration",
        ],
        "API break-ledger entry",
    )?;
    let id = required_string(entry, "id")?.to_string();
    if !valid_break_id(&id) {
        return Err(format!(
            "API break-ledger id `{id}` must contain only lowercase ASCII letters, digits, and \
             interior hyphens"
        ));
    }
    let kind = match required_string(entry, "kind")? {
        "source" => BreakKind::Source,
        "behavior" => BreakKind::Behavior,
        other => {
            return Err(format!(
                "API break-ledger entry `{id}` has unknown kind `{other}`; expected `source` or \
                 `behavior`"
            ))
        }
    };
    let packages = parse_break_packages(entry, &id)?;
    let removed_items = parse_removed_api_items(entry, &id, &packages)?;
    validate_break_kind_shape(&id, kind, &packages, &removed_items)?;
    let summary = reviewed_break_text(entry, "summary", &id)?;
    let migration = reviewed_break_text(entry, "migration", &id)?;
    Ok(BreakLedgerEntry {
        id,
        kind,
        packages,
        removed_items,
        summary,
        migration,
    })
}

fn parse_break_packages(
    entry: &serde_yaml_ng::Mapping,
    id: &str,
) -> Result<BTreeSet<String>, String> {
    let package_values = required_value(entry, "packages")?
        .as_sequence()
        .ok_or_else(|| format!("API break-ledger entry `{id}` packages must be a sequence"))?;
    let mut packages = BTreeSet::new();
    for package in package_values {
        let package = package.as_str().ok_or_else(|| {
            format!("API break-ledger entry `{id}` package names must be strings")
        })?;
        if package.is_empty() || package.trim() != package {
            return Err(format!(
                "API break-ledger entry `{id}` contains an empty or whitespace-padded package name"
            ));
        }
        if !packages.insert(package.to_string()) {
            return Err(format!(
                "API break-ledger entry `{id}` contains duplicate package `{package}`"
            ));
        }
    }
    if packages.is_empty() {
        return Err(format!(
            "API break-ledger entry `{id}` must name at least one affected package"
        ));
    }
    Ok(packages)
}

fn parse_removed_api_items(
    entry: &serde_yaml_ng::Mapping,
    id: &str,
    packages: &BTreeSet<String>,
) -> Result<Vec<RemovedApiItem>, String> {
    let removed_values = required_value(entry, "removed_items")?
        .as_sequence()
        .ok_or_else(|| format!("API break-ledger entry `{id}` removed_items must be a sequence"))?;
    let mut removed_items = Vec::with_capacity(removed_values.len());
    let mut unique_removed_items = BTreeSet::new();
    for value in removed_values {
        let removed = value.as_mapping().ok_or_else(|| {
            format!("API break-ledger entry `{id}` removed items must be mappings")
        })?;
        reject_unknown_keys(
            removed,
            &["package", "item"],
            &format!("removed item in API break-ledger entry `{id}`"),
        )?;
        let package = required_string(removed, "package")?.to_string();
        let item = required_string(removed, "item")?.to_string();
        if !packages.contains(&package) {
            return Err(format!(
                "API break-ledger entry `{id}` records a removed item for unlisted package \
                 `{package}`"
            ));
        }
        if item.is_empty() || item.trim() != item {
            return Err(format!(
                "API break-ledger entry `{id}` contains an empty or whitespace-padded API item"
            ));
        }
        if !unique_removed_items.insert((package.clone(), item.clone())) {
            return Err(format!(
                "API break-ledger entry `{id}` repeats removed item `{item}` for `{package}`"
            ));
        }
        removed_items.push(RemovedApiItem { package, item });
    }
    Ok(removed_items)
}

fn validate_break_kind_shape(
    id: &str,
    kind: BreakKind,
    packages: &BTreeSet<String>,
    removed_items: &[RemovedApiItem],
) -> Result<(), String> {
    match kind {
        BreakKind::Source if removed_items.is_empty() => {
            return Err(format!(
                "source break-ledger entry `{id}` must enumerate at least one removed API item"
            ))
        }
        BreakKind::Source => {
            let packages_with_items = removed_items
                .iter()
                .map(|removed| removed.package.as_str())
                .collect::<BTreeSet<_>>();
            let packages_without_items = packages
                .iter()
                .filter(|package| !packages_with_items.contains(package.as_str()))
                .collect::<Vec<_>>();
            if !packages_without_items.is_empty() {
                return Err(format!(
                    "source break-ledger entry `{id}` names packages without removed items: \
                     {packages_without_items:?}"
                ));
            }
        }
        BreakKind::Behavior if !removed_items.is_empty() => {
            return Err(format!(
                "behavior break-ledger entry `{id}` must not contain removed API items; record \
                 source removals in a separate `source` entry"
            ))
        }
        BreakKind::Behavior => {}
    }
    Ok(())
}

fn valid_break_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        && id.as_bytes().first().is_some_and(u8::is_ascii_alphanumeric)
        && id.as_bytes().last().is_some_and(u8::is_ascii_alphanumeric)
}

fn reviewed_break_text(
    entry: &serde_yaml_ng::Mapping,
    field: &str,
    id: &str,
) -> Result<String, String> {
    let value = required_string(entry, field)?.trim().to_string();
    if value.len() < 20 {
        return Err(format!(
            "API break-ledger {field} for `{id}` must be at least 20 characters"
        ));
    }
    if contains_pending_review_marker(&value) {
        return Err(format!(
            "API break-ledger {field} for `{id}` is still pending maintainer review"
        ));
    }
    Ok(value)
}

fn parse_review_entry(
    package: &str,
    entry: &serde_yaml_ng::Mapping,
) -> Result<ReviewEntry, String> {
    reject_unknown_keys(
        entry,
        &[
            "removed_fingerprint",
            "added_fingerprint",
            "hidden_count",
            "hidden_fingerprint",
            "rationale",
            "hidden_rationale",
        ],
        &format!("API review for `{package}`"),
    )?;
    let removed_fingerprint = required_string(entry, "removed_fingerprint")?.to_string();
    let added_fingerprint = required_string(entry, "added_fingerprint")?.to_string();
    let hidden_count = required_usize(entry, "hidden_count")?;
    let hidden_fingerprint = required_string(entry, "hidden_fingerprint")?.to_string();
    let rationale = required_string(entry, "rationale")?.trim().to_string();
    if rationale.len() < 20 {
        return Err(format!(
            "API review rationale for `{package}` must be at least 20 characters"
        ));
    }
    if contains_pending_review_marker(&rationale) {
        return Err(format!(
            "API review rationale for `{package}` is still pending maintainer review"
        ));
    }
    let hidden_rationale =
        optional_string(entry, "hidden_rationale")?.map(|value| value.trim().to_string());
    if hidden_count == 0 && hidden_fingerprint != "none" {
        return Err(format!(
            "API review for `{package}` has zero hidden items but fingerprint \
             `{hidden_fingerprint}` instead of `none`"
        ));
    }
    if hidden_count > 0
        && hidden_rationale
            .as_deref()
            .is_none_or(|value| value.len() < 20)
    {
        return Err(format!(
            "API review for `{package}` has a nonempty hidden inventory and requires a \
             package-specific hidden rationale of at least 20 characters"
        ));
    }
    if hidden_rationale
        .as_deref()
        .is_some_and(contains_pending_review_marker)
    {
        return Err(format!(
            "hidden API rationale for `{package}` is still pending maintainer review"
        ));
    }
    Ok(ReviewEntry {
        removed_fingerprint,
        added_fingerprint,
        hidden_count,
        hidden_fingerprint,
        rationale,
        hidden_rationale,
    })
}

fn contains_pending_review_marker(value: &str) -> bool {
    value
        .split(|character: char| !character.is_ascii_alphanumeric())
        .any(|word| word.eq_ignore_ascii_case("pending"))
}

pub(super) fn validate_reviews(
    config: &ReviewConfig,
    candidate_version: &str,
    diffs: &[PackageApiDiff],
) -> Result<(), String> {
    let mut errors = Vec::new();
    if config.candidate_version != candidate_version {
        errors.push(format!(
            "review config candidate {} does not match workspace candidate {candidate_version}",
            config.candidate_version
        ));
    }
    let known_packages = diffs
        .iter()
        .map(|diff| diff.package.as_str())
        .collect::<BTreeSet<_>>();
    let ledger_removed = validate_break_ledger(config, &known_packages, &mut errors);
    for diff in diffs {
        validate_package_review(config, diff, &ledger_removed, &mut errors);
    }
    for package in config.reviews.keys() {
        if !known_packages.contains(package.as_str()) {
            errors.push(format!(
                "review config contains unknown package `{package}`"
            ));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "unreviewed or stale public API changes:\n- {}\nReview {API_DIFF_REPORT} and both API snapshot diffs, then update {API_REVIEW_CONFIG} with every exact generated fingerprint/count, every removed item in the source-break ledger, and the required package-specific rationales and migrations.",
            errors.join("\n- ")
        ))
    }
}

fn validate_break_ledger(
    config: &ReviewConfig,
    known_packages: &BTreeSet<&str>,
    errors: &mut Vec<String>,
) -> BTreeMap<String, BTreeSet<String>> {
    let mut ledger_removed = BTreeMap::<String, BTreeSet<String>>::new();
    let mut ledger_owners = BTreeMap::<(String, String), String>::new();
    for entry in &config.break_ledger {
        if entry.summary.trim().len() < 20 || entry.migration.trim().len() < 20 {
            errors.push(format!(
                "break-ledger entry `{}` lacks a substantive summary or migration",
                entry.id
            ));
        }
        match entry.kind {
            BreakKind::Source if entry.removed_items.is_empty() => errors.push(format!(
                "source break-ledger entry `{}` has no removed API items",
                entry.id
            )),
            BreakKind::Behavior if !entry.removed_items.is_empty() => errors.push(format!(
                "behavior break-ledger entry `{}` contains removed API items",
                entry.id
            )),
            BreakKind::Source | BreakKind::Behavior => {}
        }
        for package in &entry.packages {
            if !known_packages.contains(package.as_str()) {
                errors.push(format!(
                    "break-ledger entry `{}` contains unknown package `{package}`",
                    entry.id
                ));
            }
        }
        for removed in &entry.removed_items {
            let key = (removed.package.clone(), removed.item.clone());
            if let Some(previous) = ledger_owners.insert(key, entry.id.clone()) {
                errors.push(format!(
                    "removed API item `{}` for `{}` appears in both source-break entries \
                     `{previous}` and `{}`",
                    removed.item, removed.package, entry.id
                ));
            }
            ledger_removed
                .entry(removed.package.clone())
                .or_default()
                .insert(removed.item.clone());
        }
    }
    ledger_removed
}

fn validate_package_review(
    config: &ReviewConfig,
    diff: &PackageApiDiff,
    ledger_removed: &BTreeMap<String, BTreeSet<String>>,
    errors: &mut Vec<String>,
) {
    let Some(review) = config.reviews.get(&diff.package) else {
        errors.push(format!(
            "`{}` has no review entry covering its ordinary diff and hidden inventory",
            diff.package
        ));
        return;
    };
    let removed_fingerprint = diff.removed_fingerprint();
    if review.removed_fingerprint != removed_fingerprint {
        errors.push(format!(
            "`{}` removed/changed fingerprint is {removed_fingerprint}, but its review records {}",
            diff.package, review.removed_fingerprint
        ));
    }
    let added_fingerprint = diff.added_fingerprint();
    if review.added_fingerprint != added_fingerprint {
        errors.push(format!(
            "`{}` added fingerprint is {added_fingerprint}, but its review records {}",
            diff.package, review.added_fingerprint
        ));
    }
    if review.hidden_count != diff.hidden.len() {
        errors.push(format!(
            "`{}` hidden inventory count is {}, but its review records {}",
            diff.package,
            diff.hidden.len(),
            review.hidden_count
        ));
    }
    let hidden_fingerprint = diff.hidden_fingerprint();
    if review.hidden_fingerprint != hidden_fingerprint {
        errors.push(format!(
            "`{}` hidden inventory fingerprint is {hidden_fingerprint}, but its review records {}",
            diff.package, review.hidden_fingerprint
        ));
    }
    if review.rationale.trim().len() < 20 {
        errors.push(format!("`{}` review rationale is too short", diff.package));
    }
    if !diff.hidden.is_empty()
        && review
            .hidden_rationale
            .as_deref()
            .is_none_or(|rationale| rationale.trim().len() < 20)
    {
        errors.push(format!(
            "`{}` nonempty hidden inventory lacks a package-specific rationale",
            diff.package
        ));
    }
    let reviewed_removed = ledger_removed
        .get(&diff.package)
        .cloned()
        .unwrap_or_default();
    let unlisted = diff
        .removed
        .difference(&reviewed_removed)
        .cloned()
        .collect::<Vec<_>>();
    let stale = reviewed_removed
        .difference(&diff.removed)
        .cloned()
        .collect::<Vec<_>>();
    if !unlisted.is_empty() || !stale.is_empty() {
        errors.push(format!(
            "`{}` source-break ledger does not exactly cover its removed API items; unlisted: \
             {unlisted:?}; stale: {stale:?}",
            diff.package
        ));
    }
}

fn required_value<'a>(
    mapping: &'a serde_yaml_ng::Mapping,
    key: &str,
) -> Result<&'a serde_yaml_ng::Value, String> {
    mapping
        .get(serde_yaml_ng::Value::String(key.to_string()))
        .ok_or_else(|| format!("API review config is missing `{key}`"))
}

fn required_string<'a>(mapping: &'a serde_yaml_ng::Mapping, key: &str) -> Result<&'a str, String> {
    required_value(mapping, key)?
        .as_str()
        .ok_or_else(|| format!("API review config `{key}` must be a string"))
}

fn optional_string(mapping: &serde_yaml_ng::Mapping, key: &str) -> Result<Option<String>, String> {
    mapping
        .get(serde_yaml_ng::Value::String(key.to_string()))
        .map(|value| {
            value
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("API review config `{key}` must be a string"))
        })
        .transpose()
}

fn required_u64(mapping: &serde_yaml_ng::Mapping, key: &str) -> Result<u64, String> {
    required_value(mapping, key)?
        .as_u64()
        .ok_or_else(|| format!("API review config `{key}` must be an unsigned integer"))
}

fn required_usize(mapping: &serde_yaml_ng::Mapping, key: &str) -> Result<usize, String> {
    usize::try_from(required_u64(mapping, key)?)
        .map_err(|error| format!("API review config `{key}` does not fit usize: {error}"))
}

fn require_exact(
    mapping: &serde_yaml_ng::Mapping,
    key: &str,
    expected: &str,
) -> Result<(), String> {
    let actual = required_string(mapping, key)?;
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "API review config `{key}` must be `{expected}`, found `{actual}`"
        ))
    }
}

fn reject_unknown_keys(
    mapping: &serde_yaml_ng::Mapping,
    allowed: &[&str],
    context: &str,
) -> Result<(), String> {
    let allowed = allowed.iter().copied().collect::<BTreeSet<_>>();
    let unknown = mapping
        .keys()
        .map(|key| {
            key.as_str()
                .ok_or_else(|| format!("{context} keys must be strings"))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|key| !allowed.contains(key))
        .collect::<Vec<_>>();
    if unknown.is_empty() {
        Ok(())
    } else {
        Err(format!("{context} contains unknown keys: {unknown:?}"))
    }
}
