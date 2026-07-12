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
            "reviews",
        ],
        "API review config",
    )?;
    if required_u64(root, "version")? != 2 {
        return Err("API review config version must be 2".to_string());
    }
    require_exact(root, "baseline_tag", SEMVER_BASELINE_TAG)?;
    require_exact(root, "baseline_version", SEMVER_BASELINE_VERSION)?;
    let candidate_version = required_string(root, "candidate_version")?.to_string();
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
        reviews,
    })
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
    for diff in diffs {
        let Some(review) = config.reviews.get(&diff.package) else {
            errors.push(format!(
                "`{}` has no review entry covering its ordinary diff and hidden inventory",
                diff.package
            ));
            continue;
        };
        let removed_fingerprint = diff.removed_fingerprint();
        if review.removed_fingerprint != removed_fingerprint {
            errors.push(format!(
                "`{}` removed/changed fingerprint is {removed_fingerprint}, but its review \
                 records {}",
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
                "`{}` hidden inventory fingerprint is {hidden_fingerprint}, but its review \
                 records {}",
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
    }
    let known_packages = diffs
        .iter()
        .map(|diff| diff.package.as_str())
        .collect::<BTreeSet<_>>();
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
            "unreviewed or stale public API changes:\n- {}\nReview {API_DIFF_REPORT} and both API snapshot diffs, then update {API_REVIEW_CONFIG} with every exact generated fingerprint/count and the required package-specific rationales.",
            errors.join("\n- ")
        ))
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
