# SPDX-License-Identifier: MIT OR Apache-2.0

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from scripts import publish_release


def release_manifest(*crates: str) -> publish_release.ReleaseManifest:
    return publish_release.ReleaseManifest(
        ordered_crates=crates,
        api_contracts={crate: "stable" for crate in crates},
    )


def package(
    name: str,
    dependencies: list[tuple[str, str | None]],
    *,
    publish: object = None,
    target_kinds: tuple[str, ...] = ("lib",),
) -> dict[str, object]:
    return {
        "id": name,
        "name": name,
        "version": "0.7.3",
        "publish": publish,
        "manifest_path": f"/workspace/{name}/Cargo.toml",
        "dependencies": [
            {
                "name": dependency,
                "kind": kind,
                "target": None,
                "optional": False,
                "path": f"/workspace/{dependency}",
                "req": "=0.7.3",
                "source": None,
            }
            for dependency, kind in dependencies
        ],
        "targets": [{"kind": list(target_kinds)}],
    }


class FakeApi:
    def __init__(self, records: dict[str, publish_release.RegistryRecord]) -> None:
        self.records = records
        self.calls: list[tuple[str, str]] = []

    def version_record(self, crate: str, version: str) -> publish_release.RegistryRecord:
        self.calls.append((crate, version))
        return self.records[crate]


class ManifestTests(unittest.TestCase):
    def test_schema_two_manifest_records_order_and_api_contracts(self) -> None:
        payload = {
            "schema": 2,
            "crates": [
                {"name": "stable-api", "api_contract": "stable"},
                {
                    "name": "experimental-api",
                    "api_contract": "experimental",
                },
                {
                    "name": "implementation-spi",
                    "api_contract": "implementation",
                },
                {"name": "cli", "api_contract": "binary"},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "release-crates.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = publish_release.load_release_manifest(manifest_path)

        self.assertEqual(
            manifest.ordered_crates,
            ("stable-api", "experimental-api", "implementation-spi", "cli"),
        )
        self.assertEqual(
            manifest.api_contracts,
            {
                "stable-api": "stable",
                "experimental-api": "experimental",
                "implementation-spi": "implementation",
                "cli": "binary",
            },
        )

        invalid_payloads = (
            ({"schema": 1, "crates": payload["crates"]}, "schema must be exactly 2"),
            ({"schema": 2.0, "crates": payload["crates"]}, "schema must be exactly 2"),
            ({"schema": True, "crates": payload["crates"]}, "schema must be exactly 2"),
            (
                {"schema": 2, "crates": payload["crates"], "extra": True},
                "unexpected fields",
            ),
            ({"schema": 2, "crates": []}, "non-empty array"),
            (
                {
                    "schema": 2,
                    "crates": [
                        {"name": "duplicate", "api_contract": "stable"},
                        {"name": "duplicate", "api_contract": "experimental"},
                    ],
                },
                "duplicate",
            ),
            (
                {
                    "schema": 2,
                    "crates": [
                        {"name": "crate", "api_contract": "unsupported"},
                    ],
                },
                "api_contract",
            ),
            (
                {
                    "schema": 2,
                    "crates": [
                        {
                            "name": "crate",
                            "api_contract": "stable",
                            "extra": True,
                        },
                    ],
                },
                "unexpected fields",
            ),
        )
        for index, (invalid, expected) in enumerate(invalid_payloads):
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    manifest_path = (
                        Path(temporary_directory) / f"invalid-{index}.json"
                    )
                    manifest_path.write_text(
                        json.dumps(invalid), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(
                        publish_release.PublishError, expected
                    ):
                        publish_release.load_release_manifest(manifest_path)

        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "duplicate-field.json"
            manifest_path.write_text(
                '{"schema":2,"schema":2,"crates":'
                '[{"name":"crate","api_contract":"stable"}]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                publish_release.PublishError, "duplicate field"
            ):
                publish_release.load_release_manifest(manifest_path)

    def test_metadata_validation_requires_complete_dependency_ordered_publish_set(self) -> None:
        manifest = release_manifest("base", "consumer")
        metadata = {
            "workspace_members": ["base", "consumer", "private-tests"],
            "packages": [
                package("base", []),
                package("consumer", [("base", None), ("private-tests", "dev")]),
                package("private-tests", [], publish=[]),
            ]
        }

        self.assertEqual(
            publish_release.validate_release_graph(manifest, metadata), "0.7.3"
        )

        reversed_manifest = release_manifest("consumer", "base")
        with self.assertRaisesRegex(publish_release.PublishError, "before dependency"):
            publish_release.validate_release_graph(reversed_manifest, metadata)

        incomplete = release_manifest("base")
        with self.assertRaisesRegex(publish_release.PublishError, "publishable workspace crates"):
            publish_release.validate_release_graph(incomplete, metadata)

        self.assertEqual(
            publish_release.registry_independent_crates(manifest, metadata),
            ("base",),
        )

    def test_metadata_validation_enforces_tiers_sources_and_exact_requirements(
        self,
    ) -> None:
        manifest = release_manifest("base", "consumer")
        metadata = {
            "workspace_members": ["base", "consumer"],
            "packages": [
                package("base", []),
                package("consumer", [("base", "build")]),
            ],
        }

        with self.assertRaisesRegex(publish_release.PublishError, "before dependency"):
            publish_release.validate_release_graph(
                release_manifest("consumer", "base"), metadata
            )

        metadata["packages"][1]["dependencies"][0]["kind"] = "dev"
        publish_release.validate_release_graph(
            release_manifest("consumer", "base"), metadata
        )

        metadata["packages"][1]["dependencies"][0]["kind"] = None
        metadata["packages"][1]["dependencies"][0]["req"] = "^0.7.3"
        with self.assertRaisesRegex(publish_release.PublishError, "exact release"):
            publish_release.validate_release_graph(manifest, metadata)

        metadata["packages"][1]["dependencies"][0]["req"] = "=0.7.3"
        metadata["packages"][1]["dependencies"][0]["source"] = (
            "registry+https://github.com/rust-lang/crates.io-index"
        )
        metadata["packages"][1]["dependencies"][0]["path"] = None
        with self.assertRaisesRegex(
            publish_release.PublishError, "workspace/path sourced"
        ):
            publish_release.validate_release_graph(manifest, metadata)

        metadata["packages"][1]["dependencies"][0]["source"] = None
        metadata["packages"][1]["dependencies"][0]["path"] = "/workspace/base"
        binary_manifest = publish_release.ReleaseManifest(
            ordered_crates=manifest.ordered_crates,
            api_contracts={"base": "stable", "consumer": "binary"},
        )
        with self.assertRaisesRegex(publish_release.PublishError, "binary target"):
            publish_release.validate_release_graph(binary_manifest, metadata)

    def test_metadata_validation_requires_crates_io_publication_eligibility(
        self,
    ) -> None:
        manifest = release_manifest("base")
        metadata = {
            "workspace_members": ["base"],
            "packages": [package("base", [], publish=["private-registry"])],
        }

        with self.assertRaisesRegex(
            publish_release.PublishError, "publishable workspace crates"
        ):
            publish_release.validate_release_graph(manifest, metadata)

        metadata["packages"][0]["publish"] = ["crates-io"]
        publish_release.validate_release_graph(manifest, metadata)


class PackagingTests(unittest.TestCase):
    def test_workspace_packages_resolve_unpublished_exact_dependencies_locally(self) -> None:
        manifest = release_manifest("base", "consumer")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metadata = {
                "packages": [
                    {
                        "name": crate,
                        "manifest_path": str(root / "crates" / crate / "Cargo.toml"),
                    }
                    for crate in manifest.ordered_crates
                ]
            }
            package_dir = root / "target" / "package"
            package_dir.mkdir(parents=True)
            for crate in manifest.ordered_crates:
                (package_dir / f"{crate}-0.7.3.crate").write_bytes(crate.encode())

            commands: list[list[str]] = []

            def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                commands.append(command)
                self.assertEqual(kwargs["cwd"], root)
                crate = command[command.index("-p") + 1]
                if crate == "consumer" and "--config" not in command:
                    return subprocess.CompletedProcess(
                        command,
                        1,
                        "",
                        "failed to select a version for the requirement "
                        "`base = \"=0.7.3\"`; candidate versions found which "
                        "didn't match; location searched: crates.io index",
                    )
                return subprocess.CompletedProcess(command, 0, "", "")

            with (
                mock.patch.object(publish_release, "ROOT", root),
                mock.patch.object(publish_release.subprocess, "run", side_effect=run),
            ):
                publish_release.package_checksums(manifest, "0.7.3", metadata)

        expected_patches = {
            f'patch.crates-io.{crate}.path="{root.resolve() / "crates" / crate}"'
            for crate in manifest.ordered_crates
        }
        self.assertEqual(len(commands), 3)
        self.assertNotIn("--config", commands[0])
        self.assertNotIn("--config", commands[1])
        configs = {
            commands[2][index + 1]
            for index, argument in enumerate(commands[2])
            if argument == "--config"
        }
        self.assertEqual(configs, expected_patches)


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

        manifest = release_manifest("a")
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
        manifest = release_manifest("a", "b", "c")
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
        manifest = release_manifest("a")
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

        refreshed: list[tuple[str, str]] = []

        def package_archive_checksum(crate: str, version: str) -> str:
            refreshed.append((crate, version))
            return "aaa"

        checksums = {"a": "staged-checksum"}
        with mock.patch.object(
            publish_release,
            "package_archive_checksum",
            side_effect=package_archive_checksum,
        ):
            publish_release.publish_remaining(
                api,
                manifest,
                "0.7.3",
                checksums,
                0,
                run=run,
                sleep=delays.append,
            )

        self.assertEqual(attempts, 2)
        self.assertEqual(delays, [5])
        self.assertGreaterEqual(len(api.calls), 1)
        self.assertEqual(refreshed, [("a", "0.7.3"), ("a", "0.7.3")])
        self.assertEqual(checksums, {"a": "aaa"})

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
