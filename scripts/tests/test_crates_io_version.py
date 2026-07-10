# SPDX-License-Identifier: MIT OR Apache-2.0

from __future__ import annotations

import json
import urllib.error
import unittest
from typing import Any

from scripts import crates_io_version as version_check


class FakeResponse:
    def __init__(self, payload: Any, status: int = 200) -> None:
        self.status = status
        self._payload = (
            payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        )

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        del args

    def read(self, limit: int) -> bytes:
        return self._payload[:limit]


class FakeVersionApi:
    def __init__(self, states: dict[str, version_check.VersionState | Exception]) -> None:
        self.states = states
        self.calls: list[tuple[str, str]] = []

    def version_state(self, crate: str, version: str) -> version_check.VersionState:
        self.calls.append((crate, version))
        state = self.states[crate]
        if isinstance(state, Exception):
            raise state
        return state


class ExactVersionStateTests(unittest.TestCase):
    def test_http_404_is_available(self) -> None:
        def missing(request: Any, timeout: int) -> Any:
            del timeout
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

        api = version_check.CratesIoApi(opener=missing)
        self.assertIs(
            api.version_state("j2k-core", "0.7.0"),
            version_check.VersionState.AVAILABLE,
        )

    def test_exact_http_200_payload_is_published_without_authorization(self) -> None:
        requests: list[Any] = []

        def published(request: Any, timeout: int) -> FakeResponse:
            self.assertEqual(timeout, version_check.REQUEST_TIMEOUT_SECONDS)
            requests.append(request)
            return FakeResponse(
                {"version": {"crate": "j2k-core", "num": "0.7.0"}}
            )

        api = version_check.CratesIoApi(opener=published)
        self.assertIs(
            api.version_state("j2k-core", "0.7.0"),
            version_check.VersionState.PUBLISHED,
        )
        headers = {key.lower(): value for key, value in requests[0].header_items()}
        self.assertNotIn("authorization", headers)
        self.assertIn("user-agent", headers)

    def test_non_404_http_failures_are_not_treated_as_available(self) -> None:
        for status in (401, 403, 429, 500):
            with self.subTest(status=status):
                def reject(request: Any, timeout: int, status: int = status) -> Any:
                    del timeout
                    raise urllib.error.HTTPError(
                        request.full_url, status, "failure", {}, None
                    )

                api = version_check.CratesIoApi(opener=reject)
                with self.assertRaisesRegex(
                    version_check.VersionCheckError, f"HTTP {status}"
                ):
                    api.version_state("j2k-core", "0.7.0")

    def test_network_failures_are_not_treated_as_available(self) -> None:
        failures = (
            urllib.error.URLError(TimeoutError()),
            TimeoutError(),
            OSError("network unavailable"),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                def reject(
                    request: Any, timeout: int, failure: Exception = failure
                ) -> Any:
                    del request, timeout
                    raise failure

                api = version_check.CratesIoApi(opener=reject)
                with self.assertRaisesRegex(
                    version_check.VersionCheckError, "request failed"
                ):
                    api.version_state("j2k-core", "0.7.0")

    def test_malformed_and_mismatched_payloads_fail_closed(self) -> None:
        payloads = (
            b"not json",
            {},
            {"version": {"crate": "other", "num": "0.7.0"}},
            {"version": {"crate": "j2k-core", "num": "0.6.2"}},
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                api = version_check.CratesIoApi(
                    opener=lambda request, timeout, payload=payload: FakeResponse(payload)
                )
                with self.assertRaises(version_check.VersionCheckError):
                    api.version_state("j2k-core", "0.7.0")


class ReleaseSetTests(unittest.TestCase):
    def test_initial_publish_rejects_any_existing_version_after_checking_all(self) -> None:
        api = FakeVersionApi(
            {
                "a": version_check.VersionState.AVAILABLE,
                "b": version_check.VersionState.PUBLISHED,
                "c": version_check.VersionState.AVAILABLE,
            }
        )
        with self.assertRaisesRegex(version_check.VersionCheckError, "already published"):
            version_check.verify_version_set(
                api,  # type: ignore[arg-type]
                ["a", "b", "c"],
                "0.7.0",
                allow_published_rerun=False,
            )
        self.assertEqual(api.calls, [("a", "0.7.0"), ("b", "0.7.0"), ("c", "0.7.0")])

    def test_idempotent_retry_accepts_only_a_published_prefix(self) -> None:
        api = FakeVersionApi(
            {
                "a": version_check.VersionState.PUBLISHED,
                "b": version_check.VersionState.PUBLISHED,
                "c": version_check.VersionState.AVAILABLE,
            }
        )
        prefix = version_check.verify_version_set(
            api,  # type: ignore[arg-type]
            ["a", "b", "c"],
            "0.7.0",
            allow_published_rerun=True,
        )
        self.assertEqual(prefix, ("a", "b"))

    def test_non_prefix_publication_is_rejected(self) -> None:
        api = FakeVersionApi(
            {
                "a": version_check.VersionState.PUBLISHED,
                "b": version_check.VersionState.AVAILABLE,
                "c": version_check.VersionState.PUBLISHED,
            }
        )
        with self.assertRaisesRegex(
            version_check.VersionCheckError, "dependency-order prefix"
        ):
            version_check.verify_version_set(
                api,  # type: ignore[arg-type]
                ["a", "b", "c"],
                "0.7.0",
                allow_published_rerun=True,
            )

    def test_every_lookup_failure_is_reported_and_duplicates_are_rejected(self) -> None:
        api = FakeVersionApi(
            {
                "a": version_check.VersionCheckError("timeout"),
                "b": version_check.VersionCheckError("HTTP 429"),
            }
        )
        with self.assertRaisesRegex(version_check.VersionCheckError, "a: timeout") as error:
            version_check.verify_version_set(
                api,  # type: ignore[arg-type]
                ["a", "b"],
                "0.7.0",
                allow_published_rerun=False,
            )
        self.assertIn("b: HTTP 429", str(error.exception))
        self.assertEqual(api.calls, [("a", "0.7.0"), ("b", "0.7.0")])

        with self.assertRaisesRegex(version_check.VersionCheckError, "duplicates"):
            version_check.verify_version_set(
                api,  # type: ignore[arg-type]
                ["a", "a"],
                "0.7.0",
                allow_published_rerun=False,
            )

        unknown = FakeVersionApi({"a": "unknown"})
        with self.assertRaisesRegex(version_check.VersionCheckError, "unknown state"):
            version_check.verify_version_set(
                unknown,  # type: ignore[arg-type]
                ["a"],
                "0.7.0",
                allow_published_rerun=False,
            )

    def test_parser_smoke(self) -> None:
        args = version_check.build_parser().parse_args(
            [
                "verify-set",
                "--version",
                "0.7.0",
                "--crate",
                "j2k-core",
                "--allow-published-rerun",
            ]
        )
        self.assertEqual(args.command, "verify-set")
        self.assertTrue(args.allow_published_rerun)


if __name__ == "__main__":
    unittest.main()
