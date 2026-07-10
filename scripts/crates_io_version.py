#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Fail-closed crates.io version-state checks for staged publication."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from enum import Enum
from typing import Any, Callable, Mapping, Sequence


CRATES_IO_API_URL = "https://crates.io/api/v1"
MAX_RESPONSE_BYTES = 1_048_576
REQUEST_TIMEOUT_SECONDS = 15
CRATE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
VERSION_PATTERN = re.compile(r"[0-9A-Za-z][0-9A-Za-z.+-]*\Z")


class VersionCheckError(RuntimeError):
    """A target version could not be classified safely."""


class VersionState(str, Enum):
    AVAILABLE = "available"
    PUBLISHED = "published"


def _validated_crate(crate: str) -> str:
    if not CRATE_PATTERN.fullmatch(crate):
        raise VersionCheckError("crate name is malformed")
    return crate


def _validated_version(version: str) -> str:
    if not VERSION_PATTERN.fullmatch(version):
        raise VersionCheckError("crate version is malformed")
    return version


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise VersionCheckError(
            f"malformed crates.io response: {context} must be an object"
        )
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise VersionCheckError(f"malformed crates.io response: {context} must be a string")
    return value


class CratesIoApi:
    """Small unauthenticated crates.io client with explicit 200/404 semantics."""

    def __init__(
        self,
        *,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        self._opener = opener

    def version_state(self, crate: str, version: str) -> VersionState:
        exact_crate = _validated_crate(crate)
        exact_version = _validated_version(version)
        encoded_crate = urllib.parse.quote(exact_crate, safe="")
        encoded_version = urllib.parse.quote(exact_version, safe="")
        request = urllib.request.Request(
            f"{CRATES_IO_API_URL}/crates/{encoded_crate}/{encoded_version}",
            headers={
                "Accept": "application/json",
                "User-Agent": "j2k-release-preflight/1 (+https://github.com/frames-sg/j2k)",
            },
        )
        try:
            with self._opener(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                status = getattr(response, "status", None)
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            error.close()
            if error.code == 404:
                return VersionState.AVAILABLE
            raise VersionCheckError(
                f"crates.io returned HTTP {error.code} for {exact_crate} {exact_version}"
            ) from None
        except urllib.error.URLError as error:
            raise VersionCheckError(
                f"crates.io request failed for {exact_crate} {exact_version} "
                f"({type(error.reason).__name__})"
            ) from None
        except (OSError, TimeoutError) as error:
            raise VersionCheckError(
                f"crates.io request failed for {exact_crate} {exact_version} "
                f"({type(error).__name__})"
            ) from None

        if status != 200:
            raise VersionCheckError(
                f"crates.io returned unexpected HTTP status {status!r} "
                f"for {exact_crate} {exact_version}"
            )
        if len(raw) > MAX_RESPONSE_BYTES:
            raise VersionCheckError(
                f"crates.io response was too large for {exact_crate} {exact_version}"
            )
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise VersionCheckError(
                f"crates.io returned invalid JSON for {exact_crate} {exact_version}"
            ) from None

        version_payload = _object(
            _object(payload, "root").get("version"), "version"
        )
        response_crate = _string(version_payload.get("crate"), "version.crate")
        response_version = _string(version_payload.get("num"), "version.num")
        if response_crate != exact_crate or response_version != exact_version:
            raise VersionCheckError(
                f"crates.io returned mismatched identity for {exact_crate} {exact_version}"
            )
        return VersionState.PUBLISHED


def verify_version_set(
    api: CratesIoApi,
    crates: Sequence[str],
    version: str,
    *,
    allow_published_rerun: bool,
) -> tuple[str, ...]:
    """Check every crate and return the already-published dependency-order prefix."""

    exact_version = _validated_version(version)
    exact_crates = tuple(_validated_crate(crate) for crate in crates)
    if not exact_crates:
        raise VersionCheckError("at least one crate is required")
    if len(set(exact_crates)) != len(exact_crates):
        raise VersionCheckError("crate list contains duplicates")

    states: list[tuple[str, VersionState]] = []
    failures: list[str] = []
    for crate in exact_crates:
        try:
            state = api.version_state(crate, exact_version)
            if state not in (VersionState.AVAILABLE, VersionState.PUBLISHED):
                raise VersionCheckError("version lookup returned an unknown state")
            states.append((crate, state))
        except VersionCheckError as error:
            failures.append(f"{crate}: {error}")
    if failures:
        raise VersionCheckError(
            "could not classify every crates.io target version:\n- "
            + "\n- ".join(failures)
        )

    published = tuple(
        crate for crate, state in states if state is VersionState.PUBLISHED
    )
    if published and not allow_published_rerun:
        raise VersionCheckError(
            "target version is already published for: "
            + ", ".join(published)
            + "; set CRATES_IO_ALLOW_PUBLISHED_RERUN=true only for an intentional retry"
        )

    saw_available = False
    non_prefix: list[str] = []
    prefix: list[str] = []
    for crate, state in states:
        if state is VersionState.AVAILABLE:
            saw_available = True
        elif saw_available:
            non_prefix.append(crate)
        else:
            prefix.append(crate)
    if non_prefix:
        raise VersionCheckError(
            "published crates do not form a dependency-order prefix; "
            "unexpected published crates: "
            + ", ".join(non_prefix)
        )
    return tuple(prefix)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    state_parser = subparsers.add_parser(
        "state", help="print available or published for one exact crate version"
    )
    state_parser.add_argument("--crate", required=True)
    state_parser.add_argument("--version", required=True)

    set_parser = subparsers.add_parser(
        "verify-set", help="verify every target version before publication begins"
    )
    set_parser.add_argument("--version", required=True)
    set_parser.add_argument("--crate", action="append", required=True)
    set_parser.add_argument("--allow-published-rerun", action="store_true")
    return parser


def run_command(args: argparse.Namespace, api: CratesIoApi | None = None) -> None:
    client = api or CratesIoApi()
    if args.command == "state":
        print(client.version_state(args.crate, args.version).value)
        return
    if args.command == "verify-set":
        prefix = verify_version_set(
            client,
            args.crate,
            args.version,
            allow_published_rerun=args.allow_published_rerun,
        )
        available = len(args.crate) - len(prefix)
        print(
            f"verified {len(args.crate)} crates.io target versions: "
            f"{len(prefix)} published retry prefix, {available} available"
        )
        return
    raise VersionCheckError(f"unsupported command {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_command(args)
    except VersionCheckError as error:
        print(f"crates.io version check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
