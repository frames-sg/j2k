#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Fail-closed CI change planning and aggregate-result validation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


CUDA_ROOTS = frozenset(
    {
        "j2k-cuda-runtime",
        "j2k-jpeg-cuda",
        "j2k-cuda",
        "j2k-transcode-cuda",
    }
)
METAL_ROOTS = frozenset(
    {
        "j2k-metal-support",
        "j2k-jpeg-metal",
        "j2k-metal",
        "j2k-transcode-metal",
    }
)
SHARED_GPU_PACKAGES = frozenset({"j2k-ml"})
DOCUMENTATION_PREFIXES = ("docs/", "engineering/")
DOCUMENTATION_PATHS = frozenset(
    {
        "CHANGELOG.md",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "LICENSE-APACHE",
        "LICENSE-MIT",
        "NOTICES.md",
        "README.md",
        "SECURITY.md",
    }
)
QUALITY_EVIDENCE_PREFIXES = (
    "docs/stable-api-",
    "engineering/public-api-review-",
    "engineering/reviewed-public-api-diff-",
)
BROAD_PATHS = frozenset(
    {
        "Cargo.lock",
        "Cargo.toml",
        "deny.toml",
        "rust-toolchain.toml",
        "typos.toml",
    }
)
BROAD_PREFIXES = (".cargo/", ".github/workflows/", "scripts/", "xtask/")
CONDITIONAL_JOBS = ("rust-quality", "docs", "metal-compile")


class PlanError(RuntimeError):
    """CI planning input or results are malformed or unsafe."""


@dataclass(frozen=True)
class Plan:
    rust: bool
    docs: bool
    cuda: bool
    metal: bool
    metal_compile: bool
    paths: tuple[str, ...]

    def expectations(self) -> dict[str, bool]:
        return {
            "rust-quality": self.rust,
            "docs": self.docs,
            "metal-compile": self.metal_compile,
        }


def _safe_repo_path(value: str) -> str:
    if not value or "\0" in value or "\n" in value or "\r" in value:
        raise PlanError("changed paths must be non-empty single-line strings")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise PlanError(f"changed path escapes the repository: {value!r}")
    normalized = path.as_posix()
    if normalized in ("", "."):
        raise PlanError(f"invalid changed path: {value!r}")
    return normalized


def parse_name_status(raw: bytes) -> tuple[str, ...]:
    """Parse `git diff --name-status -z`, preserving both rename/copy paths."""

    try:
        fields = raw.decode("utf-8", errors="strict").split("\0")
    except UnicodeDecodeError as error:
        raise PlanError("git diff paths must be valid UTF-8") from error
    if fields and fields[-1] == "":
        fields.pop()
    paths: list[str] = []
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        if not status:
            raise PlanError("git diff contains an empty status")
        kind = status[0]
        if kind in ("R", "C"):
            path_count = 2
        elif kind in ("A", "D", "M", "T", "U"):
            path_count = 1
        else:
            raise PlanError(f"unsupported git diff status: {status!r}")
        if index + path_count > len(fields):
            raise PlanError(f"truncated git diff record for status {status!r}")
        paths.extend(_safe_repo_path(value) for value in fields[index : index + path_count])
        index += path_count
    if not paths:
        raise PlanError("git diff did not report any changed paths")
    return tuple(paths)


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PlanError(f"{context} must be an object")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise PlanError(f"{context} must be a non-empty string")
    return value


def _metadata_graph(
    metadata: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, set[str]]]:
    root = Path(_string(metadata.get("workspace_root"), "metadata workspace_root"))
    packages_value = metadata.get("packages")
    if not isinstance(packages_value, list) or not packages_value:
        raise PlanError("metadata packages must be a non-empty array")

    package_roots: dict[str, str] = {}
    dependencies: dict[str, set[str]] = {}
    raw_packages: list[Mapping[str, Any]] = []
    for index, value in enumerate(packages_value):
        package = _object(value, f"metadata package {index}")
        name = _string(package.get("name"), f"metadata package {index} name")
        manifest = Path(
            _string(
                package.get("manifest_path"),
                f"metadata package {name} manifest_path",
            )
        )
        try:
            relative_root = manifest.parent.relative_to(root).as_posix()
        except ValueError as error:
            raise PlanError(f"workspace package {name} is outside workspace_root") from error
        if name in dependencies or relative_root in package_roots:
            raise PlanError(f"duplicate workspace package metadata for {name}")
        package_roots[relative_root] = name
        dependencies[name] = set()
        raw_packages.append(package)

    workspace_names = set(dependencies)
    for package in raw_packages:
        name = _string(package.get("name"), "metadata package name")
        raw_dependencies = package.get("dependencies")
        if not isinstance(raw_dependencies, list):
            raise PlanError(f"metadata dependencies for {name} must be an array")
        for index, value in enumerate(raw_dependencies):
            dependency = _object(value, f"metadata dependency {name}[{index}]")
            dependency_name = _string(
                dependency.get("name"), f"metadata dependency {name}[{index}] name"
            )
            if dependency_name in workspace_names:
                dependencies[name].add(dependency_name)
    return package_roots, dependencies


def _dependency_closure(roots: Iterable[str], dependencies: Mapping[str, set[str]]) -> set[str]:
    pending = list(roots)
    closure: set[str] = set()
    while pending:
        package = pending.pop()
        if package in closure:
            continue
        if package not in dependencies:
            raise PlanError(f"GPU root package is absent from cargo metadata: {package}")
        closure.add(package)
        pending.extend(dependencies[package] - closure)
    return closure


def _package_for_path(path: str, package_roots: Mapping[str, str]) -> str | None:
    matches = [
        (root, package)
        for root, package in package_roots.items()
        if path == root or path.startswith(f"{root}/")
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: len(item[0]))[1]


def _is_documentation(path: str) -> bool:
    return path in DOCUMENTATION_PATHS or path.startswith(DOCUMENTATION_PREFIXES)


def _is_quality_evidence(path: str) -> bool:
    return path.startswith(QUALITY_EVIDENCE_PREFIXES)


def classify_paths(paths: Iterable[str], metadata: Mapping[str, Any]) -> Plan:
    normalized_paths = tuple(dict.fromkeys(_safe_repo_path(path) for path in paths))
    if not normalized_paths:
        raise PlanError("at least one changed path is required")
    package_roots, dependencies = _metadata_graph(metadata)
    cuda_dependencies = _dependency_closure(CUDA_ROOTS, dependencies)
    metal_dependencies = _dependency_closure(METAL_ROOTS, dependencies)

    rust = False
    docs = False
    cuda = False
    metal = False
    for path in normalized_paths:
        if _is_quality_evidence(path):
            docs = True
            rust = True
            cuda = True
            metal = True
            continue
        if _is_documentation(path):
            docs = True
            continue
        if path in BROAD_PATHS or path.startswith(BROAD_PREFIXES):
            rust = True
            cuda = True
            metal = True
            continue

        package = _package_for_path(path, package_roots)
        if package is not None:
            rust = True
            if package in SHARED_GPU_PACKAGES:
                cuda = True
                metal = True
            else:
                cuda = cuda or package in cuda_dependencies
                metal = metal or package in metal_dependencies
            continue

        if path.startswith("crates/"):
            rust = True
            cuda = True
            metal = True
            continue

        if path.endswith((".rs", ".toml", ".lock", ".py", ".sh", ".yml", ".yaml")):
            rust = True
            cuda = True
            metal = True
            continue

        # Known repository metadata stays hosted-only. Everything else fails broad.
        if path in {".gitignore", ".jscpd.json", ".github/CODEOWNERS"}:
            rust = True
            continue
        rust = True
        cuda = True
        metal = True

    return Plan(
        rust=rust,
        docs=docs,
        cuda=cuda,
        metal=metal,
        metal_compile=metal,
        paths=normalized_paths,
    )


def validate_aggregate(
    expectations: Mapping[str, Any],
    results: Mapping[str, Any],
    *,
    always_required: Iterable[str] = (),
) -> None:
    expected_keys = set(CONDITIONAL_JOBS)
    missing = expected_keys - set(expectations)
    if missing:
        raise PlanError(f"missing conditional jobs: {', '.join(sorted(missing))}")
    unexpected = set(expectations) - expected_keys
    if unexpected:
        raise PlanError(f"unknown conditional jobs: {', '.join(sorted(unexpected))}")
    failures: list[str] = []

    def result_for(job: str) -> str:
        value = results.get(job)
        if not isinstance(value, dict):
            return "missing" if value is None else "malformed"
        result = value.get("result")
        return result if isinstance(result, str) and result else "malformed"

    for job in always_required:
        result = result_for(job)
        if result != "success":
            failures.append(f"always-required job {job}: {result}")

    for job, required in expectations.items():
        if not isinstance(required, bool):
            failures.append(f"expectation for {job}: malformed")
            continue
        result = result_for(job)
        allowed = {"success"} if required else {"success", "skipped"}
        if result not in allowed:
            requirement = "required job" if required else "optional job"
            failures.append(f"{requirement} {job}: {result}")

    if failures:
        raise PlanError("CI aggregate rejected results:\n" + "\n".join(sorted(failures)))


def validate_success_results(results: Mapping[str, Any]) -> None:
    """Require every reported dependency to be present, well formed, and successful."""

    if not results:
        raise PlanError("success-only aggregate did not receive any job results")
    failures: list[str] = []
    for job, value in results.items():
        if not isinstance(job, str) or not job:
            failures.append("malformed job name")
            continue
        if not isinstance(value, dict):
            result = "malformed"
        else:
            raw_result = value.get("result")
            result = raw_result if isinstance(raw_result, str) and raw_result else "malformed"
        if result != "success":
            failures.append(f"required job {job}: {result}")
    if failures:
        raise PlanError("CI aggregate rejected results:\n" + "\n".join(sorted(failures)))


def _load_json(value: str, context: str) -> Mapping[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise PlanError(f"{context} is not valid JSON") from error
    return _object(decoded, context)


def _run(command: Sequence[str]) -> bytes:
    try:
        completed = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as error:
        raise PlanError(f"command failed: {' '.join(command)}") from error
    return completed.stdout


def _write_outputs(plan: Plan, output_path: str | None) -> None:
    values = {
        "rust": plan.rust,
        "docs": plan.docs,
        "cuda": plan.cuda,
        "metal": plan.metal,
        "metal_compile": plan.metal_compile,
        "expectations": plan.expectations(),
        "plan": asdict(plan),
    }
    rendered = {
        key: json.dumps(value, separators=(",", ":"))
        if isinstance(value, (dict, list, tuple))
        else str(value).lower()
        for key, value in values.items()
    }
    if output_path:
        with open(output_path, "a", encoding="utf-8") as output:
            for key, value in rendered.items():
                output.write(f"{key}={value}\n")
    print(json.dumps(asdict(plan), indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="classify a git revision range")
    plan.add_argument("--base", required=True)
    plan.add_argument("--head", required=True)
    plan.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))

    aggregate = subparsers.add_parser("aggregate", help="validate conditional CI results")
    aggregate.add_argument("--expectations-json", required=True)
    aggregate.add_argument("--needs-json", required=True)
    aggregate.add_argument("--always-required", action="append", default=[])

    success_only = subparsers.add_parser(
        "success-only", help="require every supplied CI result to be successful"
    )
    success_only.add_argument("--needs-json", required=True)
    return parser


def run_command(args: argparse.Namespace) -> None:
    if args.command == "plan":
        metadata = _load_json(
            _run(["cargo", "metadata", "--format-version", "1", "--no-deps"]).decode(),
            "cargo metadata",
        )
        paths = parse_name_status(
            _run(["git", "diff", "--name-status", "-z", f"{args.base}...{args.head}"])
        )
        _write_outputs(classify_paths(paths, metadata), args.github_output)
        return
    if args.command == "aggregate":
        validate_aggregate(
            _load_json(args.expectations_json, "expectations"),
            _load_json(args.needs_json, "needs"),
            always_required=args.always_required,
        )
        print("all required CI jobs succeeded and all skips matched the validated plan")
        return
    if args.command == "success-only":
        validate_success_results(_load_json(args.needs_json, "needs"))
        print("all required CI jobs succeeded")
        return
    raise PlanError(f"unsupported command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run_command(build_parser().parse_args(argv))
    except PlanError as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
