#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Check the j2k-c release archives, and write or verify their SHA-256 manifest.

`.github/workflows/c-packages.yml` writes the manifest once every archive has
passed its consumer tests. `.github/workflows/publish.yml` verifies the same
files against that manifest's recorded digest before attaching them to the
draft GitHub Release.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import re
import sys
from typing import Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
# The platforms c-packages.yml builds, with each archive's format.
PLATFORMS = (
    ("linux-x86_64", "tar.gz"),
    ("linux-aarch64", "tar.gz"),
    ("macos-aarch64", "tar.gz"),
    ("windows-x86_64", "zip"),
)
VERSION_PATTERN = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?")
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
CHECKSUM_LINE = re.compile(r"([0-9a-f]{64})  (\S+)")


class AssetError(RuntimeError):
    """The release files are incomplete or unexpected, or fail their checksums."""


def _require_version(version: str) -> str:
    if not VERSION_PATTERN.fullmatch(version):
        raise AssetError(f"version must be X.Y.Z with an optional pre-release: {version!r}")
    return version


def archive_names(version: str) -> list[str]:
    _require_version(version)
    return [f"j2k-c-{version}-{platform}.{suffix}" for platform, suffix in PLATFORMS]


def manifest_name(version: str) -> str:
    return f"j2k-c-{_require_version(version)}-SHA256SUMS"


def release_files(version: str) -> list[str]:
    """Every file the release gets: archives, their .sha256 files, and the manifest."""
    archives = archive_names(version)
    return sorted([*archives, *(f"{name}.sha256" for name in archives), manifest_name(version)])


def sha256_file(path: pathlib.Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as binary:
        while True:
            block = binary.read(1 << 20)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def read_checksums(path: pathlib.Path) -> dict[str, str]:
    """Parse `sha256sum` lines into {file name: digest}, rejecting anything else."""
    try:
        text = path.read_text(encoding="ascii")
    except UnicodeDecodeError:
        raise AssetError(f"{path.name} is not ASCII") from None
    if not text.endswith("\n"):
        raise AssetError(f"{path.name} must end with a newline")
    entries: dict[str, str] = {}
    for line in text[:-1].split("\n"):
        match = CHECKSUM_LINE.fullmatch(line)
        if not match or match.group(2) in entries:
            raise AssetError(f"{path.name} has a malformed or repeated line: {line!r}")
        entries[match.group(2)] = match.group(1)
    return entries


def require_exact_files(directory: pathlib.Path, expected: Sequence[str]) -> None:
    actual = sorted(entry.name for entry in directory.iterdir())
    if actual != sorted(expected):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise AssetError(
            f"{directory} must hold exactly the C package files; "
            f"missing {missing}, unexpected {unexpected}"
        )
    for name in expected:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise AssetError(f"{name} must be a regular file")


def verified_archives(directory: pathlib.Path, version: str) -> dict[str, str]:
    """Check each archive against its own .sha256 file; return {name: digest}."""
    digests: dict[str, str] = {}
    for name in archive_names(version):
        recorded = read_checksums(directory / f"{name}.sha256")
        if list(recorded) != [name]:
            raise AssetError(f"{name}.sha256 must name only {name}")
        actual = sha256_file(directory / name)
        if recorded[name] != actual:
            raise AssetError(f"{name} has SHA-256 {actual}, not the one in {name}.sha256")
        digests[name] = actual
    return digests


def write_manifest(directory: pathlib.Path, version: str) -> tuple[pathlib.Path, str]:
    """Write the manifest for a complete, self-consistent set of archives."""
    archives = archive_names(version)
    require_exact_files(directory, [*archives, *(f"{name}.sha256" for name in archives)])
    digests = verified_archives(directory, version)
    manifest = directory / manifest_name(version)
    with open(manifest, "x", encoding="ascii", newline="\n") as text:
        text.write("".join(f"{digests[name]}  {name}\n" for name in sorted(digests)))
    return manifest, sha256_file(manifest)


def verify_release(
    directory: pathlib.Path, version: str, manifest_sha256: str, tag: str | None = None
) -> list[pathlib.Path]:
    """Require exactly the release files, matching the recorded manifest digest."""
    if tag is not None and tag != f"v{_require_version(version)}":
        raise AssetError(f"release tag {tag} does not match package version {version}")
    if not DIGEST_PATTERN.fullmatch(manifest_sha256):
        raise AssetError("the recorded manifest SHA-256 must be 64 lowercase hex digits")
    names = release_files(version)
    require_exact_files(directory, names)
    manifest = directory / manifest_name(version)
    if sha256_file(manifest) != manifest_sha256:
        raise AssetError(f"{manifest.name} does not match the SHA-256 recorded when it was written")
    recorded = read_checksums(manifest)
    archives = archive_names(version)
    if sorted(recorded) != sorted(archives):
        raise AssetError(f"{manifest.name} must list exactly {archives}")
    digests = verified_archives(directory, version)
    changed = [name for name in archives if recorded[name] != digests[name]]
    if changed:
        raise AssetError(f"archives differ from {manifest.name}: {changed}")
    return [directory / name for name in names]


def workspace_version(cargo_toml: pathlib.Path = ROOT / "Cargo.toml") -> str:
    """Read [workspace.package].version, which every crate and package shares."""
    try:
        import tomllib
    except ModuleNotFoundError:
        raise AssetError("reading Cargo.toml needs Python 3.11 or newer") from None
    with open(cargo_toml, "rb") as binary:
        manifest = tomllib.load(binary)
    version = manifest.get("workspace", {}).get("package", {}).get("version")
    if not isinstance(version, str):
        raise AssetError(f"{cargo_toml.name} has no [workspace.package].version")
    return _require_version(version)


def _append(path: str | None, text: str) -> None:
    if path:
        with open(path, "a", encoding="utf-8") as output:
            output.write(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    version = subparsers.add_parser("version", help="print the workspace package version")
    version.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))

    manifest = subparsers.add_parser("manifest", help="check the archives and write the manifest")
    manifest.add_argument("--version", required=True)
    manifest.add_argument("--dir", type=pathlib.Path, required=True)
    manifest.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    manifest.add_argument("--summary", default=os.environ.get("GITHUB_STEP_SUMMARY"))

    verify = subparsers.add_parser("verify", help="check the release files against the manifest")
    verify.add_argument("--version", required=True)
    verify.add_argument("--dir", type=pathlib.Path, required=True)
    verify.add_argument("--manifest-sha256", required=True)
    verify.add_argument("--tag")
    return parser


def run_command(args: argparse.Namespace) -> None:
    if args.command == "version":
        version = workspace_version()
        _append(args.github_output, f"version={version}\n")
        print(version)
    elif args.command == "manifest":
        manifest, digest = write_manifest(args.dir, args.version)
        contents = manifest.read_text(encoding="ascii")
        _append(args.github_output, f"sha256={digest}\n")
        _append(args.summary, f"### {manifest.name}\n\nSHA-256 `{digest}`\n\n```text\n{contents}```\n")
        print(f"{contents}{manifest.name}: {digest}")
    else:
        for path in verify_release(args.dir, args.version, args.manifest_sha256, args.tag):
            print(path.name)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run_command(build_parser().parse_args(argv))
    except (AssetError, OSError) as error:
        print(f"c-release-assets: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
