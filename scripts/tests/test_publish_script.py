# SPDX-License-Identifier: MIT OR Apache-2.0

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLISH_SCRIPT = REPO_ROOT / "scripts" / "publish-crate.sh"


class PublishScriptEnvironmentTests(unittest.TestCase):
    def run_with(self, **overrides: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(overrides)
        return subprocess.run(
            ["bash", str(PUBLISH_SCRIPT), "j2k-core"],
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_publish_attempts_must_be_a_positive_decimal(self) -> None:
        for value in ("0", "08", "-1", "1.5", "three"):
            with self.subTest(value=value):
                result = self.run_with(CRATES_IO_PUBLISH_ATTEMPTS=value)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("positive decimal integer", result.stderr)

    def test_retry_and_settle_seconds_must_be_nonnegative_decimals(self) -> None:
        for name in (
            "CRATES_IO_RATE_LIMIT_RETRY_SECONDS",
            "CRATES_IO_INDEX_SETTLE_SECONDS",
        ):
            for value in ("08", "-1", "1.5", "soon"):
                with self.subTest(name=name, value=value):
                    result = self.run_with(**{name: value})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("nonnegative decimal integer", result.stderr)

    def test_boolean_controls_reject_unknown_values(self) -> None:
        result = self.run_with(CRATES_IO_ALLOW_PUBLISHED_RERUN="yes")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be true, false, 1, or 0", result.stderr)

        result = self.run_with(DRY_RUN_ONLY="yes")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be true or false", result.stderr)

    def test_exactly_one_command_argument_is_required(self) -> None:
        for arguments in ([], ["j2k-core", "unexpected"]):
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    ["bash", str(PUBLISH_SCRIPT), *arguments],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("usage: publish-crate.sh", result.stderr)


if __name__ == "__main__":
    unittest.main()
