# SPDX-License-Identifier: MIT OR Apache-2.0

from __future__ import annotations

import subprocess
import unittest
import urllib.error

from scripts import publish_release


def package(name: str, dependencies: list[tuple[str, str | None]], *, publish: object = None) -> dict[str, object]:
    return {
        "name": name,
        "version": "0.7.3",
        "publish": publish,
        "dependencies": [
            {"name": dependency, "kind": kind, "target": None}
            for dependency, kind in dependencies
        ],
    }


class FakeApi:
    def __init__(self, records: dict[str, publish_release.RegistryRecord]) -> None:
        self.records = records
        self.calls: list[tuple[str, str]] = []

    def version_record(self, crate: str, version: str) -> publish_release.RegistryRecord:
        self.calls.append((crate, version))
        return self.records[crate]


class ManifestTests(unittest.TestCase):
    def test_metadata_validation_requires_complete_dependency_ordered_publish_set(self) -> None:
        manifest = publish_release.ReleaseManifest(
            ordered_crates=("base", "consumer"),
            registry_independent=frozenset({"base"}),
        )
        metadata = {
            "packages": [
                package("base", []),
                package("consumer", [("base", None), ("private-tests", "dev")]),
                package("private-tests", [], publish=[]),
            ]
        }

        self.assertEqual(
            publish_release.validate_release_graph(manifest, metadata), "0.7.3"
        )

        reversed_manifest = publish_release.ReleaseManifest(
            ordered_crates=("consumer", "base"),
            registry_independent=frozenset({"base"}),
        )
        with self.assertRaisesRegex(publish_release.PublishError, "before dependency"):
            publish_release.validate_release_graph(reversed_manifest, metadata)

        incomplete = publish_release.ReleaseManifest(
            ordered_crates=("base",), registry_independent=frozenset({"base"})
        )
        with self.assertRaisesRegex(publish_release.PublishError, "publishable workspace crates"):
            publish_release.validate_release_graph(incomplete, metadata)


class RegistryTests(unittest.TestCase):
    def test_registry_transport_errors_are_typed_and_retried(self) -> None:
        attempts = 0

        def opener(_request: object, *, timeout: int) -> object:
            nonlocal attempts
            attempts += 1
            self.assertEqual(timeout, publish_release.REQUEST_TIMEOUT_SECONDS)
            raise urllib.error.URLError(OSError("Temporary failure in name resolution"))

        api = publish_release.CratesIoApi(opener)
        with self.assertRaisesRegex(
            publish_release.TransientPublishError,
            "Temporary failure in name resolution",
        ):
            api.version_record("a", "0.7.3")

        manifest = publish_release.ReleaseManifest(
            ordered_crates=("a",), registry_independent=frozenset()
        )
        delays: list[int] = []
        with self.assertRaises(publish_release.TransientPublishError):
            publish_release.validate_registry_state_with_retry(
                api,
                manifest,
                "0.7.3",
                {"a": "aaa"},
                allow_published=True,
                sleep=delays.append,
            )
        self.assertEqual(delays, [5, 15, 30])
        self.assertEqual(attempts, 5)

    def test_registry_retry_requires_prefix_and_matching_local_checksums(self) -> None:
        manifest = publish_release.ReleaseManifest(
            ordered_crates=("a", "b", "c"), registry_independent=frozenset()
        )
        api = FakeApi(
            {
                "a": publish_release.RegistryRecord(True, "aaa"),
                "b": publish_release.RegistryRecord(False, None),
                "c": publish_release.RegistryRecord(False, None),
            }
        )

        self.assertEqual(
            publish_release.validate_registry_state(
                api, manifest, "0.7.3", {"a": "aaa", "b": "bbb", "c": "ccc"}, allow_published=True
            ),
            1,
        )

        api.records["a"] = publish_release.RegistryRecord(True, "wrong")
        with self.assertRaisesRegex(publish_release.PublishError, "checksum"):
            publish_release.validate_registry_state(
                api, manifest, "0.7.3", {"a": "aaa", "b": "bbb", "c": "ccc"}, allow_published=True
            )

        api.records = {
            "a": publish_release.RegistryRecord(True, "aaa"),
            "b": publish_release.RegistryRecord(False, None),
            "c": publish_release.RegistryRecord(True, "ccc"),
        }
        with self.assertRaisesRegex(publish_release.PublishError, "prefix"):
            publish_release.validate_registry_state(
                api, manifest, "0.7.3", {"a": "aaa", "b": "bbb", "c": "ccc"}, allow_published=True
            )

    def test_transient_publish_retries_are_bounded_and_requery_registry(self) -> None:
        manifest = publish_release.ReleaseManifest(
            ordered_crates=("a",), registry_independent=frozenset()
        )
        api = FakeApi({"a": publish_release.RegistryRecord(False, None)})
        attempts = 0
        delays: list[int] = []

        def run(command: list[str]) -> subprocess.CompletedProcess[str]:
            nonlocal attempts
            attempts += 1
            self.assertEqual(command, ["cargo", "publish", "--locked", "-p", "a"])
            if attempts == 1:
                return subprocess.CompletedProcess(command, 1, "", "HTTP 503 unavailable")
            api.records["a"] = publish_release.RegistryRecord(True, "aaa")
            return subprocess.CompletedProcess(command, 0, "published", "")

        publish_release.publish_remaining(
            api,
            manifest,
            "0.7.3",
            {"a": "aaa"},
            0,
            run=run,
            sleep=delays.append,
        )

        self.assertEqual(attempts, 2)
        self.assertEqual(delays, [5])
        self.assertGreaterEqual(len(api.calls), 1)

    def test_authentication_and_validation_failures_are_never_retried(self) -> None:
        for output in (
            "HTTP 403 forbidden",
            "unauthorized token",
            "failed to verify package",
            "version already exists",
        ):
            with self.subTest(output=output):
                self.assertFalse(publish_release.is_retryable_failure(output))
        for output in ("HTTP 429", "HTTP 502", "timed out", "connection reset"):
            with self.subTest(output=output):
                self.assertTrue(publish_release.is_retryable_failure(output))


if __name__ == "__main__":
    unittest.main()
