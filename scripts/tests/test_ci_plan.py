# SPDX-License-Identifier: MIT OR Apache-2.0

from __future__ import annotations

import unittest

from scripts import ci_plan


def metadata() -> dict[str, object]:
    def package(name: str, dependencies: list[str]) -> dict[str, object]:
        return {
            "name": name,
            "manifest_path": f"/repo/crates/{name}/Cargo.toml",
            "dependencies": [
                {"name": dependency, "kind": None, "target": None}
                for dependency in dependencies
            ],
        }

    return {
        "workspace_root": "/repo",
        "packages": [
            package("j2k-core", []),
            package("j2k-profile", []),
            package("j2k-cuda-runtime", ["j2k-core", "j2k-profile"]),
            package("j2k-jpeg-cuda", ["j2k-cuda-runtime", "j2k-core"]),
            package("j2k-cuda", ["j2k-cuda-runtime", "j2k-core"]),
            package("j2k-transcode-cuda", ["j2k-cuda-runtime", "j2k-core"]),
            package("j2k-metal-support", ["j2k-core"]),
            package("j2k-jpeg-metal", ["j2k-metal-support", "j2k-core"]),
            package("j2k-metal", ["j2k-metal-support", "j2k-core", "j2k-profile"]),
            package("j2k-transcode-metal", ["j2k-metal-support", "j2k-core"]),
            package("j2k-ml", ["j2k-cuda", "j2k-metal"]),
            package("j2k-cli", ["j2k-core"]),
        ],
    }


class DiffParserTests(unittest.TestCase):
    def test_name_status_parser_keeps_both_rename_paths_and_deletions(self) -> None:
        raw = (
            b"R100\0crates/j2k-cuda/src/old.rs\0"
            b"crates/j2k-metal/src/new.rs\0"
            b"D\0scripts/removed.py\0M\0docs/guide.md\0"
        )

        self.assertEqual(
            ci_plan.parse_name_status(raw),
            (
                "crates/j2k-cuda/src/old.rs",
                "crates/j2k-metal/src/new.rs",
                "scripts/removed.py",
                "docs/guide.md",
            ),
        )

    def test_name_status_parser_rejects_truncated_or_unsafe_records(self) -> None:
        for raw in (
            b"R100\0old.rs\0",
            b"M\0../outside\0",
            b"M\0/absolute\0",
            b"X\0path\0",
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ci_plan.PlanError):
                    ci_plan.parse_name_status(raw)


class ClassificationTests(unittest.TestCase):
    def classify(self, *paths: str) -> ci_plan.Plan:
        return ci_plan.classify_paths(paths, metadata())

    def test_cuda_and_metal_crates_are_isolated(self) -> None:
        cuda = self.classify("crates/j2k-cuda/src/lib.rs")
        metal = self.classify("crates/j2k-metal/src/lib.rs")

        self.assertTrue(cuda.rust)
        self.assertTrue(cuda.cuda)
        self.assertFalse(cuda.metal)
        self.assertTrue(metal.rust)
        self.assertTrue(metal.metal)
        self.assertFalse(metal.cuda)

    def test_shared_dependency_and_ml_changes_require_both_lanes(self) -> None:
        for path in (
            "crates/j2k-core/src/lib.rs",
            "crates/j2k-profile/src/lib.rs",
            "crates/j2k-ml/src/lib.rs",
        ):
            with self.subTest(path=path):
                plan = self.classify(path)
                self.assertTrue(plan.cuda)
                self.assertTrue(plan.metal)

    def test_docs_only_change_stays_cheap(self) -> None:
        plan = self.classify("docs/architecture.md")

        self.assertTrue(plan.docs)
        self.assertFalse(plan.rust)
        self.assertFalse(plan.cuda)
        self.assertFalse(plan.metal)
        self.assertFalse(plan.metal_compile)

    def test_machine_readable_api_evidence_requires_fail_closed_quality_lanes(self) -> None:
        for path in (
            "docs/stable-api-1.0.public-api.txt",
            "docs/stable-api-1.0.implementation-public-api.txt",
            "engineering/public-api-review-0.7.3.yml",
            "engineering/reviewed-public-api-diff-0.7.3.md",
        ):
            with self.subTest(path=path):
                plan = self.classify(path)
                self.assertTrue(plan.docs)
                self.assertTrue(plan.rust)
                self.assertTrue(plan.cuda)
                self.assertTrue(plan.metal)
                self.assertTrue(plan.metal_compile)

    def test_root_metadata_unknown_infrastructure_and_planner_fail_closed(self) -> None:
        for path in (
            "Cargo.lock",
            ".cargo/config.toml",
            "build-support/unknown-tool.cfg",
            "scripts/ci_plan.py",
        ):
            with self.subTest(path=path):
                plan = self.classify(path)
                self.assertTrue(plan.rust)
                self.assertTrue(plan.cuda)
                self.assertTrue(plan.metal)
                self.assertTrue(plan.metal_compile)


class AggregateTests(unittest.TestCase):
    def test_required_jobs_must_succeed_and_optional_jobs_may_skip(self) -> None:
        ci_plan.validate_aggregate(
            {"rust-quality": True, "docs": False, "metal-compile": False},
            {
                "planner": {"result": "success"},
                "gpu-evidence": {"result": "success"},
                "rust-quality": {"result": "success"},
                "docs": {"result": "skipped"},
                "metal-compile": {"result": "skipped"},
            },
            always_required=("planner", "gpu-evidence"),
        )

    def test_expectation_map_must_name_every_conditional_job(self) -> None:
        with self.assertRaisesRegex(ci_plan.PlanError, "missing conditional jobs"):
            ci_plan.validate_aggregate(
                {"rust-quality": True, "metal-compile": False},
                {
                    "rust-quality": {"result": "success"},
                    "docs": {"result": "skipped"},
                    "metal-compile": {"result": "skipped"},
                },
            )

    def test_missing_malformed_failed_and_unexpected_skips_fail_closed(self) -> None:
        cases = (
            (
                {"rust-quality": True, "docs": False, "metal-compile": False},
                {},
            ),
            (
                {"rust-quality": True, "docs": False, "metal-compile": False},
                {"rust-quality": "success"},
            ),
            (
                {"rust-quality": True, "docs": False, "metal-compile": False},
                {"rust-quality": {"result": "failure"}},
            ),
            (
                {"rust-quality": True, "docs": False, "metal-compile": False},
                {"rust-quality": {"result": "skipped"}},
            ),
            (
                {"rust-quality": False, "docs": False, "metal-compile": False},
                {"rust-quality": {"result": "success"}, "planner": {"result": "cancelled"}},
            ),
        )
        for expectations, results in cases:
            with self.subTest(expectations=expectations, results=results):
                with self.assertRaises(ci_plan.PlanError):
                    ci_plan.validate_aggregate(
                        expectations,
                        results,
                        always_required=("planner",) if "planner" in results else (),
                    )

    def test_success_only_aggregate_rejects_missing_malformed_and_skipped_jobs(self) -> None:
        ci_plan.validate_success_results(
            {"fmt": {"result": "success"}, "test": {"result": "success"}}
        )

        for results in (
            {},
            {"fmt": "success"},
            {"fmt": {"result": "skipped"}},
            {"fmt": {"result": "cancelled"}},
        ):
            with self.subTest(results=results):
                with self.assertRaises(ci_plan.PlanError):
                    ci_plan.validate_success_results(results)


if __name__ == "__main__":
    unittest.main()
