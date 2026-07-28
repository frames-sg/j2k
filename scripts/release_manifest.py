# SPDX-License-Identifier: MIT OR Apache-2.0
"""Strict release-manifest parsing and Cargo workspace graph validation."""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
from dataclasses import dataclass
from typing import Any, Mapping


CRATE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
API_CONTRACTS = frozenset(("stable", "experimental", "implementation", "binary"))


class ManifestError(RuntimeError):
    """The release manifest or workspace graph is invalid."""


@dataclass(frozen=True)
class ReleaseManifest:
    ordered_crates: tuple[str, ...]
    api_contracts: Mapping[str, str]


@dataclass(frozen=True)
class ValidatedRelease:
    version: str
    registry_independent: frozenset[str]


def object_value(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{context} must be an object")
    return value


def string_value(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{context} must be a non-empty string")
    return value


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ManifestError(
            f"{context} has unexpected fields; "
            f"expected={sorted(expected)}, actual={sorted(actual)}"
        )


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ManifestError(f"release manifest contains duplicate field {key!r}")
        value[key] = item
    return value


def load_release_manifest(path: pathlib.Path) -> ReleaseManifest:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_fields,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"could not read release manifest {path}: {error}") from None

    root = object_value(payload, "release manifest")
    _exact_fields(root, frozenset(("schema", "crates")), "release manifest")
    schema = root.get("schema")
    if type(schema) is not int or schema != 2:
        raise ManifestError("release manifest schema must be exactly 2")
    raw_crates = root.get("crates")
    if not isinstance(raw_crates, list) or not raw_crates:
        raise ManifestError("release manifest crates must be a non-empty array")

    ordered: list[str] = []
    api_contracts: dict[str, str] = {}
    for index, raw_crate in enumerate(raw_crates):
        crate = object_value(raw_crate, f"release manifest crates[{index}]")
        _exact_fields(
            crate,
            frozenset(("name", "api_contract")),
            f"release manifest crates[{index}]",
        )
        name = string_value(
            crate.get("name"), f"release manifest crates[{index}].name"
        )
        contract = string_value(
            crate.get("api_contract"),
            f"release manifest crates[{index}].api_contract",
        )
        if not CRATE_PATTERN.fullmatch(name):
            raise ManifestError(
                f"release manifest crates[{index}] has malformed crate name {name!r}"
            )
        if name in api_contracts:
            raise ManifestError(f"release manifest contains duplicate crate {name}")
        if contract not in API_CONTRACTS:
            raise ManifestError(
                f"release manifest api_contract {contract!r} is not one of "
                "stable, experimental, implementation, or binary"
            )
        ordered.append(name)
        api_contracts[name] = contract

    return ReleaseManifest(tuple(ordered), api_contracts)


def cargo_metadata(root: pathlib.Path) -> Mapping[str, Any]:
    result = subprocess.run(
        ["cargo", "metadata", "--locked", "--format-version", "1", "--no-deps"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ManifestError(f"cargo metadata failed:\n{result.stderr.strip()}")
    try:
        return object_value(json.loads(result.stdout), "cargo metadata")
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ManifestError(f"cargo metadata returned invalid JSON: {error}") from None


def _workspace_packages(metadata: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw_packages = metadata.get("packages")
    if not isinstance(raw_packages, list):
        raise ManifestError("cargo metadata packages must be an array")

    packages_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw_package in enumerate(raw_packages):
        package = object_value(raw_package, "cargo metadata package")
        package_id = string_value(
            package.get("id"), f"cargo metadata packages[{index}].id"
        )
        if package_id in packages_by_id:
            raise ManifestError(
                f"cargo metadata contains duplicate package id {package_id}"
            )
        packages_by_id[package_id] = package

    raw_members = metadata.get("workspace_members")
    if not isinstance(raw_members, list):
        raise ManifestError("cargo metadata workspace_members must be an array")
    packages: dict[str, Mapping[str, Any]] = {}
    seen_members: set[str] = set()
    for index, raw_member in enumerate(raw_members):
        member = string_value(
            raw_member, f"cargo metadata workspace_members[{index}]"
        )
        if member in seen_members:
            raise ManifestError(
                f"cargo metadata workspace_members contains duplicate id {member}"
            )
        seen_members.add(member)
        package = packages_by_id.get(member)
        if package is None:
            raise ManifestError(
                f"cargo metadata workspace member {member} has no package record"
            )
        name = string_value(
            package.get("name"), f"cargo metadata package {member}.name"
        )
        if name in packages:
            raise ManifestError(
                f"cargo metadata contains duplicate workspace package name {name}"
            )
        packages[name] = package
    return packages


def _crates_io_publishable(package: Mapping[str, Any], name: str) -> bool:
    publish = package.get("publish")
    if publish is None:
        return True
    if not isinstance(publish, list):
        raise ManifestError(
            f"cargo metadata package {name} has invalid publish eligibility"
        )
    registries = tuple(
        string_value(registry, f"cargo metadata package {name}.publish[{index}]")
        for index, registry in enumerate(publish)
    )
    return "crates-io" in registries


def _target_kinds(package: Mapping[str, Any], name: str) -> frozenset[str]:
    raw_targets = package.get("targets")
    if not isinstance(raw_targets, list):
        raise ManifestError(f"cargo metadata targets for {name} must be an array")
    kinds: set[str] = set()
    for target_index, raw_target in enumerate(raw_targets):
        target = object_value(
            raw_target, f"cargo metadata {name} target[{target_index}]"
        )
        raw_kinds = target.get("kind")
        if not isinstance(raw_kinds, list):
            raise ManifestError(
                f"cargo metadata {name} target[{target_index}].kind must be an array"
            )
        for kind_index, kind in enumerate(raw_kinds):
            kinds.add(
                string_value(
                    kind,
                    f"cargo metadata {name} "
                    f"target[{target_index}].kind[{kind_index}]",
                )
            )
    return frozenset(kinds)


def _dependency_kind(
    dependency: Mapping[str, Any], package: str, index: int
) -> str:
    kind = dependency.get("kind")
    if kind is None:
        return "normal"
    if not isinstance(kind, str):
        raise ManifestError(
            f"cargo metadata package {package} dependency[{index}] has invalid kind"
        )
    return kind


def _workspace_release_dependency(
    dependency: Mapping[str, Any],
    package: str,
    index: int,
    release_roots: Mapping[str, pathlib.Path],
) -> str | None:
    dependency_name = string_value(
        dependency.get("name"),
        f"cargo metadata package {package} dependency[{index}].name",
    )
    expected_root = release_roots.get(dependency_name)
    if expected_root is None:
        return None
    source = dependency.get("source")
    if source is not None:
        string_value(
            source,
            f"cargo metadata package {package} dependency[{index}].source",
        )
        raise ManifestError(
            f"release package {package} dependency {dependency_name} "
            "must be workspace/path sourced"
        )
    raw_path = dependency.get("path")
    if raw_path is None:
        raise ManifestError(
            f"release package {package} dependency {dependency_name} "
            "must be workspace/path sourced"
        )
    dependency_root = pathlib.Path(
        string_value(
            raw_path,
            f"cargo metadata package {package} dependency[{index}].path",
        )
    )
    if dependency_root != expected_root:
        raise ManifestError(
            f"release package {package} dependency {dependency_name} path "
            f"{dependency_root} does not match workspace package root {expected_root}"
        )
    return dependency_name


def validate_release_graph(
    manifest: ReleaseManifest, metadata: Mapping[str, Any]
) -> ValidatedRelease:
    packages = _workspace_packages(metadata)
    publishable = {
        name
        for name, package in packages.items()
        if _crates_io_publishable(package, name)
    }
    ordered_set = set(manifest.ordered_crates)
    if publishable != ordered_set:
        missing = sorted(publishable - ordered_set)
        extra = sorted(ordered_set - publishable)
        raise ManifestError(
            "release manifest must contain all publishable workspace crates exactly once; "
            f"missing={missing}, extra={extra}"
        )

    versions_by_package = {
        name: string_value(packages[name].get("version"), f"{name}.version")
        for name in manifest.ordered_crates
    }
    versions = set(versions_by_package.values())
    if len(versions) != 1:
        raise ManifestError(
            "publishable workspace crates must all have the same release version"
        )

    release_roots: dict[str, pathlib.Path] = {}
    for name in manifest.ordered_crates:
        manifest_path = pathlib.Path(
            string_value(packages[name].get("manifest_path"), f"{name}.manifest_path")
        )
        release_roots[name] = manifest_path.parent
        contract = manifest.api_contracts[name]
        kinds = _target_kinds(packages[name], name)
        has_library = "lib" in kinds
        has_binary = "bin" in kinds
        if contract == "binary":
            if not has_binary or has_library:
                raise ManifestError(
                    f"release package {name} has api_contract=binary but must have "
                    "a binary target and no library target"
                )
        elif not has_library:
            raise ManifestError(
                f"release package {name} has api_contract={contract} "
                "but has no library target"
            )

    positions = {crate: index for index, crate in enumerate(manifest.ordered_crates)}
    dependencies_by_package: dict[str, set[str]] = {}
    for crate in manifest.ordered_crates:
        raw_dependencies = packages[crate].get("dependencies")
        if not isinstance(raw_dependencies, list):
            raise ManifestError(
                f"cargo metadata dependencies for {crate} must be an array"
            )
        release_dependencies: set[str] = set()
        for dependency_index, raw_dependency in enumerate(raw_dependencies):
            dependency = object_value(raw_dependency, f"{crate} dependency")
            kind = _dependency_kind(dependency, crate, dependency_index)
            dependency_name = _workspace_release_dependency(
                dependency,
                crate,
                dependency_index,
                release_roots,
            )
            if dependency_name is None:
                continue
            requirement = string_value(
                dependency.get("req"),
                f"cargo metadata package {crate} "
                f"dependency {dependency_name}.req",
            )
            expected_requirement = f"={versions_by_package[dependency_name]}"
            if requirement != expected_requirement:
                raise ManifestError(
                    f"release package {crate} dependency {dependency_name} must use "
                    f"exact release requirement {expected_requirement}; "
                    f"found {requirement}"
                )
            if kind == "dev":
                continue
            release_dependencies.add(dependency_name)
            if positions[dependency_name] >= positions[crate]:
                raise ManifestError(
                    f"release crate {crate} appears before dependency {dependency_name}"
                )
        dependencies_by_package[crate] = release_dependencies
    independent = frozenset(
        crate
        for crate in manifest.ordered_crates
        if not dependencies_by_package[crate]
    )
    return ValidatedRelease(versions.pop(), independent)
