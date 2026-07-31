#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Package and publish the ordered workspace release with fail-closed retries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

if __package__:
    from .release_manifest import (
        CRATE_PATTERN,
        ManifestError,
        ReleaseManifest,
        cargo_metadata as _cargo_metadata,
        load_release_manifest as _load_release_manifest,
        object_value as _object,
        string_value as _string,
        validate_release_graph as _validate_release_graph,
    )
else:
    from release_manifest import (
        CRATE_PATTERN,
        ManifestError,
        ReleaseManifest,
        cargo_metadata as _cargo_metadata,
        load_release_manifest as _load_release_manifest,
        object_value as _object,
        string_value as _string,
        validate_release_graph as _validate_release_graph,
    )


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "release-crates.json"
CRATES_IO_API_URL = "https://crates.io/api/v1"
MAX_RESPONSE_BYTES = 1_048_576
REQUEST_TIMEOUT_SECONDS = 15
RETRY_DELAYS_SECONDS = (5, 15, 30)
CHECKSUM_PATTERN = re.compile(r"[0-9a-f]{64}\Z")

PublishError = ManifestError


class TransientPublishError(PublishError):
    """A bounded retry may recover a registry or transport operation."""


@dataclass(frozen=True)
class RegistryRecord:
    published: bool
    checksum: str | None


def load_release_manifest(path: pathlib.Path = DEFAULT_MANIFEST) -> ReleaseManifest:
    return _load_release_manifest(path)


def cargo_metadata() -> Mapping[str, Any]:
    return _cargo_metadata(ROOT)


def validate_release_graph(
    manifest: ReleaseManifest, metadata: Mapping[str, Any]
) -> str:
    return _validate_release_graph(manifest, metadata).version


def registry_independent_crates(
    manifest: ReleaseManifest, metadata: Mapping[str, Any]
) -> tuple[str, ...]:
    independent = _validate_release_graph(manifest, metadata).registry_independent
    return tuple(crate for crate in manifest.ordered_crates if crate in independent)


class CratesIoApi:
    """Unauthenticated exact-version lookup including the registry checksum."""

    def __init__(self, opener: Callable[..., Any] = urllib.request.urlopen) -> None:
        self._opener = opener

    def version_record(self, crate: str, version: str) -> RegistryRecord:
        if not CRATE_PATTERN.fullmatch(crate):
            raise PublishError("crate name is malformed")
        request = urllib.request.Request(
            f"{CRATES_IO_API_URL}/crates/"
            f"{urllib.parse.quote(crate, safe='')}/{urllib.parse.quote(version, safe='')}",
            headers={
                "Accept": "application/json",
                "User-Agent": "j2k-release-publisher/1 (+https://github.com/frames-sg/j2k)",
            },
        )
        try:
            with self._opener(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                status = getattr(response, "status", None)
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            error.close()
            if error.code == 404:
                return RegistryRecord(False, None)
            error_type = (
                TransientPublishError
                if error.code == 429 or 500 <= error.code < 600
                else PublishError
            )
            raise error_type(
                f"crates.io returned HTTP {error.code} for {crate} {version}"
            ) from None
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            reason = getattr(error, "reason", error)
            detail = str(reason).strip() or "no transport detail"
            raise TransientPublishError(
                f"crates.io lookup failed for {crate} {version} "
                f"({type(reason).__name__}: {detail})"
            ) from None

        if status != 200:
            raise PublishError(
                f"crates.io returned unexpected HTTP status {status!r} for {crate} {version}"
            )
        if len(raw) > MAX_RESPONSE_BYTES:
            raise PublishError(f"crates.io response was too large for {crate} {version}")
        try:
            version_payload = _object(
                _object(json.loads(raw), "crates.io response").get("version"),
                "crates.io response.version",
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PublishError(
                f"crates.io returned invalid JSON for {crate} {version}: {error}"
            ) from None
        if (
            _string(version_payload.get("crate"), "version.crate") != crate
            or _string(version_payload.get("num"), "version.num") != version
        ):
            raise PublishError(f"crates.io returned mismatched identity for {crate} {version}")
        checksum = _string(version_payload.get("checksum"), "version.checksum").lower()
        if not CHECKSUM_PATTERN.fullmatch(checksum):
            raise PublishError(f"crates.io returned a malformed checksum for {crate} {version}")
        return RegistryRecord(True, checksum)


def package_archive_checksum(
    crate: str, version: str, extra_arguments: Sequence[str] = ()
) -> str:
    command = [
        "cargo",
        "package",
        "--locked",
        "--no-verify",
        "-p",
        crate,
        *extra_arguments,
    ]
    result = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise PublishError(
            f"cargo package failed for {crate}:\n{result.stderr.strip()}"
        )
    archive = ROOT / "target" / "package" / f"{crate}-{version}.crate"
    try:
        return hashlib.sha256(archive.read_bytes()).hexdigest()
    except OSError as error:
        raise PublishError(f"could not hash packaged archive {archive}: {error}") from None


def is_unpublished_dependency_failure(output: str) -> bool:
    message = output.lower()
    return all(
        marker in message
        for marker in (
            "failed to select a version for the requirement",
            "candidate versions found which didn't match",
            "location searched: crates.io index",
        )
    )


def package_checksums(
    manifest: ReleaseManifest, version: str, metadata: Mapping[str, Any]
) -> dict[str, str]:
    raw_packages = metadata.get("packages")
    if not isinstance(raw_packages, list):
        raise PublishError("cargo metadata packages must be an array")
    packages = {
        _string(package.get("name"), "cargo metadata package.name"): package
        for raw_package in raw_packages
        for package in (_object(raw_package, "cargo metadata package"),)
    }
    root = ROOT.resolve()
    patch_arguments: list[str] = []
    for crate in manifest.ordered_crates:
        package = packages.get(crate)
        if package is None:
            raise PublishError(f"cargo metadata is missing release package {crate}")
        manifest_path = pathlib.Path(
            _string(package.get("manifest_path"), f"{crate}.manifest_path")
        ).resolve()
        package_root = manifest_path.parent
        try:
            package_root.relative_to(root)
        except ValueError:
            raise PublishError(
                f"release package {crate} is outside the workspace root"
            ) from None
        patch_arguments.extend(
            [
                "--config",
                f"patch.crates-io.{crate}.path={json.dumps(str(package_root))}",
            ]
        )

    checksums: dict[str, str] = {}
    for crate in manifest.ordered_crates:
        try:
            checksums[crate] = package_archive_checksum(crate, version)
        except PublishError as error:
            if not is_unpublished_dependency_failure(str(error)):
                raise
            checksums[crate] = package_archive_checksum(
                crate, version, patch_arguments
            )
    return checksums


def validate_registry_state(
    api: Any,
    manifest: ReleaseManifest,
    version: str,
    checksums: Mapping[str, str],
    *,
    allow_published: bool,
) -> int:
    missing_checksums = set(manifest.ordered_crates) - set(checksums)
    if missing_checksums:
        raise PublishError(
            "local checksums are missing for: " + ", ".join(sorted(missing_checksums))
        )

    records: list[tuple[str, RegistryRecord]] = []
    permanent_failures: list[str] = []
    transient_failures: list[str] = []
    for crate in manifest.ordered_crates:
        try:
            records.append((crate, api.version_record(crate, version)))
        except TransientPublishError as error:
            transient_failures.append(f"{crate}: {error}")
        except PublishError as error:
            permanent_failures.append(f"{crate}: {error}")
    if permanent_failures:
        raise PublishError(
            "could not classify every crates.io target version:\n- "
            + "\n- ".join(permanent_failures)
        )
    if transient_failures:
        raise TransientPublishError(
            "transiently could not classify every crates.io target version:\n- "
            + "\n- ".join(transient_failures)
        )

    saw_unpublished = False
    prefix_length = 0
    non_prefix: list[str] = []
    for crate, record in records:
        if not record.published:
            if record.checksum is not None:
                raise PublishError(f"unpublished crate {crate} unexpectedly has a checksum")
            saw_unpublished = True
            continue
        if record.checksum != checksums[crate]:
            raise PublishError(
                f"published checksum for {crate} does not match the local package checksum"
            )
        if saw_unpublished:
            non_prefix.append(crate)
        else:
            prefix_length += 1
    if non_prefix:
        raise PublishError(
            "published crates do not form a dependency-order prefix: "
            + ", ".join(non_prefix)
        )
    if prefix_length and not allow_published:
        published = manifest.ordered_crates[:prefix_length]
        raise PublishError(
            "target version is already published for: "
            + ", ".join(published)
            + "; set CRATES_IO_ALLOW_PUBLISHED_RERUN=true only for an intentional retry"
        )
    return prefix_length


def is_retryable_failure(output: str) -> bool:
    message = output.lower()
    permanent_markers = (
        "http 401",
        "http 403",
        "unauthorized",
        "forbidden",
        "failed to verify",
        "verification failed",
        "already exists",
        "version exists",
        "invalid manifest",
    )
    if any(marker in message for marker in permanent_markers):
        return False
    return bool(
        re.search(r"http\s+(?:429|5\d\d)\b", message)
        or any(
            marker in message
            for marker in (
                "timed out",
                "timeout",
                "connection reset",
                "connection refused",
                "temporary failure",
                "service unavailable",
            )
        )
    )


def validate_registry_state_with_retry(
    api: Any,
    manifest: ReleaseManifest,
    version: str,
    checksums: Mapping[str, str],
    *,
    allow_published: bool,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    for attempt in range(len(RETRY_DELAYS_SECONDS) + 1):
        try:
            return validate_registry_state(
                api,
                manifest,
                version,
                checksums,
                allow_published=allow_published,
            )
        except PublishError as error:
            retryable = isinstance(error, TransientPublishError) or is_retryable_failure(
                str(error)
            )
            if attempt >= len(RETRY_DELAYS_SECONDS) or not retryable:
                raise
            delay = RETRY_DELAYS_SECONDS[attempt]
            print(
                f"retrying crates.io state verification after {delay} seconds",
                file=sys.stderr,
            )
            sleep(delay)
    raise AssertionError("bounded registry-state loop did not return or raise")


def _run_publish(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True
    )


def publish_remaining(
    api: Any,
    manifest: ReleaseManifest,
    version: str,
    checksums: dict[str, str],
    start_index: int,
    *,
    run: Callable[[list[str]], subprocess.CompletedProcess[str]] = _run_publish,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    if not 0 <= start_index <= len(manifest.ordered_crates):
        raise PublishError("publish start index is outside the release manifest")

    for index in range(start_index, len(manifest.ordered_crates)):
        crate = manifest.ordered_crates[index]
        command = ["cargo", "publish", "--locked", "-p", crate]
        for attempt in range(len(RETRY_DELAYS_SECONDS) + 1):
            result = run(command)
            if result.stdout:
                print(result.stdout.rstrip())
            if result.stderr:
                print(result.stderr.rstrip(), file=sys.stderr)
            checksums[crate] = package_archive_checksum(crate, version)
            if result.returncode == 0:
                break

            prefix = validate_registry_state_with_retry(
                api,
                manifest,
                version,
                checksums,
                allow_published=True,
                sleep=sleep,
            )
            if prefix == index + 1:
                print(
                    f"crates.io confirms {crate} {version} despite the local failure; continuing"
                )
                break
            if prefix != index:
                raise PublishError(
                    f"registry prefix advanced unexpectedly while publishing {crate}"
                )
            combined_output = f"{result.stdout}\n{result.stderr}"
            if attempt >= len(RETRY_DELAYS_SECONDS) or not is_retryable_failure(
                combined_output
            ):
                raise PublishError(
                    f"cargo publish failed permanently for {crate} (exit {result.returncode})"
                )
            delay = RETRY_DELAYS_SECONDS[attempt]
            print(f"retrying {crate} after {delay} seconds", file=sys.stderr)
            sleep(delay)


def _allow_published_rerun() -> bool:
    raw = os.environ.get("CRATES_IO_ALLOW_PUBLISHED_RERUN", "false").lower()
    if raw in {"1", "true"}:
        return True
    if raw in {"0", "false", ""}:
        return False
    raise PublishError("CRATES_IO_ALLOW_PUBLISHED_RERUN must be true or false")


def run(command: str, manifest_path: pathlib.Path) -> None:
    manifest = load_release_manifest(manifest_path)
    metadata = cargo_metadata()
    version = validate_release_graph(manifest, metadata)
    checksums = package_checksums(manifest, version, metadata)
    if command == "preflight":
        print(
            f"packaged {len(checksums)} crates for {version}; release manifest is valid"
        )
        return
    if command != "publish":
        raise PublishError(f"unknown command {command}")
    if not os.environ.get("CARGO_REGISTRY_TOKEN"):
        raise PublishError("CARGO_REGISTRY_TOKEN is required for publication")
    api = CratesIoApi()
    prefix = validate_registry_state_with_retry(
        api,
        manifest,
        version,
        checksums,
        allow_published=_allow_published_rerun(),
    )
    print(
        f"validated {len(manifest.ordered_crates)} registry targets; "
        f"resuming after {prefix} published crates"
    )
    publish_remaining(api, manifest, version, checksums, prefix)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("manifest", "preflight", "publish"))
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--field", choices=("ordered-crates", "registry-independent")
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "manifest":
            if args.field is None:
                raise PublishError("manifest command requires --field")
            manifest = load_release_manifest(args.manifest.resolve())
            if args.field == "ordered-crates":
                selected = manifest.ordered_crates
            else:
                selected = registry_independent_crates(manifest, cargo_metadata())
            print("\n".join(selected))
            return 0
        if args.field is not None:
            raise PublishError("--field is only valid with the manifest command")
        run(args.command, args.manifest.resolve())
    except PublishError as error:
        print(f"publish-release: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
