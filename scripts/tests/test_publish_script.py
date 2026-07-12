# SPDX-License-Identifier: MIT OR Apache-2.0

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
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


class PublishScriptTagProofTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repository = Path(temporary.name) / "repository"
        self.repository.mkdir()
        self.remote = Path(temporary.name) / "remote.git"
        subprocess.run(
            ["git", "init", "--bare", "--quiet", str(self.remote)],
            check=True,
            capture_output=True,
            text=True,
        )
        self.fake_bin = Path(temporary.name) / "fake-bin"
        self.fake_bin.mkdir()
        self.cargo_marker = Path(temporary.name) / "cargo-called"
        (self.repository / "Cargo.toml").write_text(
            '[workspace.package]\n'
            'version = "0.7.0"\n'
            'repository = "https://github.com/frames-sg/j2k"\n',
            encoding="utf-8",
        )
        (self.repository / "tracked.txt").write_text("initial\n", encoding="utf-8")
        self.git("init", "--quiet")
        self.git("config", "user.name", "Release Test")
        self.git("config", "user.email", "release-test@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "tag.gpgSign", "false")
        self.git("add", "Cargo.toml", "tracked.txt")
        self.git("commit", "--quiet", "-m", "initial")
        self.origin_url = "https://github.com/frames-sg/j2k.git"
        self.git("remote", "add", "origin", self.origin_url)
        fake_cargo = self.fake_bin / "cargo"
        fake_cargo.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "if [[ -n \"${FAKE_CARGO_MARKER:-}\" ]]; then\n"
            "  : > \"${FAKE_CARGO_MARKER}\"\n"
            "fi\n"
            "if [[ \"${1:-} ${2:-} ${3:-}\" == \"xtask release-integrity --publish\" ]]; then\n"
            "  exit 0\n"
            "fi\n"
            "if [[ \"${1:-}\" == \"pkgid\" ]]; then\n"
            "  printf 'path+file:///release-fixture#0.7.0\\n'\n"
            "  exit 0\n"
            "fi\n"
            "printf 'unexpected cargo invocation: %s\\n' \"$*\" >&2\n"
            "exit 91\n",
            encoding="utf-8",
        )
        fake_cargo.chmod(0o755)
        real_git = shutil.which("git")
        if real_git is None:
            raise RuntimeError("git is required for publish-script tests")
        fake_git = self.fake_bin / "git"
        fake_git.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            ': "${REAL_GIT:?}"\n'
            "if [[ \"${1:-}\" == \"ls-remote\" "
            "&& \"${2:-}\" == \"--tags\" "
            "&& \"${3:-}\" == \"origin\" ]]; then\n"
            '  exec "$REAL_GIT" ls-remote --tags "$FAKE_GIT_REMOTE" "${@:4}"\n'
            "fi\n"
            'exec "$REAL_GIT" "$@"\n',
            encoding="utf-8",
        )
        fake_git.chmod(0o755)
        self.real_git = real_git

    def git(self, *arguments: str) -> None:
        subprocess.run(
            ["git", *arguments],
            cwd=self.repository,
            check=True,
            capture_output=True,
            text=True,
        )

    def commit_change(self) -> None:
        (self.repository / "tracked.txt").write_text("changed\n", encoding="utf-8")
        self.git("add", "tracked.txt")
        self.git("commit", "--quiet", "-m", "change")

    def set_origin(self, url: str) -> None:
        self.origin_url = url
        self.git("remote", "set-url", "origin", url)

    def push_release_tag(self, source: str = "refs/tags/v0.7.0") -> None:
        self.git(
            "push",
            "--quiet",
            str(self.remote),
            f"{source}:refs/tags/v0.7.0",
        )

    def run_publish(
        self,
        *,
        workflow_tag: str | None = None,
        environment_overrides: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PATH"] = f"{self.fake_bin}{os.pathsep}{environment['PATH']}"
        environment["CRATES_IO_ALLOW_PUBLISHED_RERUN"] = "false"
        environment["CRATES_IO_PUBLISH_ATTEMPTS"] = "3"
        environment["CRATES_IO_RATE_LIMIT_RETRY_SECONDS"] = "0"
        environment["CRATES_IO_INDEX_SETTLE_SECONDS"] = "0"
        environment["DRY_RUN_ONLY"] = "false"
        environment["FAKE_CARGO_MARKER"] = str(self.cargo_marker)
        environment["FAKE_GIT_REMOTE"] = str(self.remote)
        environment["REAL_GIT"] = self.real_git
        environment.pop("CRATES_IO_API_TOKEN", None)
        environment.pop("GITHUB_REF_NAME", None)
        self.cargo_marker.unlink(missing_ok=True)
        if workflow_tag is not None:
            environment["GITHUB_REF_NAME"] = workflow_tag
        if environment_overrides is not None:
            environment.update(environment_overrides)
        return subprocess.run(
            ["bash", str(PUBLISH_SCRIPT), "j2k-core"],
            cwd=self.repository,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_workflow_ref_cannot_replace_a_missing_git_tag(self) -> None:
        result = self.run_publish(workflow_tag="v0.7.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("tag does not exist in this checkout", result.stderr)

    def test_lightweight_or_stale_annotated_tags_are_rejected(self) -> None:
        self.git("tag", "v0.7.0")
        lightweight = self.run_publish()
        self.assertNotEqual(lightweight.returncode, 0)
        self.assertIn("to be an annotated tag", lightweight.stderr)

        self.git("tag", "--delete", "v0.7.0")
        self.git("tag", "--annotate", "v0.7.0", "--message", "release")
        self.commit_change()
        stale = self.run_publish()
        self.assertNotEqual(stale.returncode, 0)
        self.assertIn("peel exactly to HEAD", stale.stderr)

    def test_verified_tag_must_also_match_workflow_ref(self) -> None:
        self.git("tag", "--annotate", "v0.7.0", "--message", "release")
        result = self.run_publish(workflow_tag="v0.7.1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match verified tag v0.7.0", result.stderr)

    def test_annotated_tag_at_head_passes_tag_proof(self) -> None:
        self.git("tag", "--annotate", "v0.7.0", "--message", "release")
        self.push_release_tag()
        result = self.run_publish(workflow_tag="v0.7.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CRATES_IO_API_TOKEN is required for a real publish", result.stderr)
        self.assertTrue(self.cargo_marker.exists())

    def test_supported_origin_url_forms_accept_valid_remote_tag(self) -> None:
        self.git("tag", "--annotate", "v0.7.0", "--message", "release")
        self.push_release_tag()
        for origin_url in (
            "https://github.com/frames-sg/j2k.git",
            "git@github.com:frames-sg/j2k.git",
            "ssh://git@github.com/frames-sg/j2k.git",
        ):
            with self.subTest(origin_url=origin_url):
                self.set_origin(origin_url)
                result = self.run_publish(workflow_tag="v0.7.0")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "CRATES_IO_API_TOKEN is required for a real publish",
                    result.stderr,
                )
                self.assertTrue(self.cargo_marker.exists())

    def test_wrong_origin_is_rejected_before_cargo(self) -> None:
        self.git("tag", "--annotate", "v0.7.0", "--message", "release")
        self.push_release_tag()
        self.set_origin("https://github.com/attacker/j2k.git")
        result = self.run_publish(
            workflow_tag="v0.7.0",
            environment_overrides={
                "GITHUB_REPOSITORY": "frames-sg/j2k",
                "GITHUB_SERVER_URL": "https://github.com",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match the canonical workspace repository", result.stderr)
        self.assertNotIn("attacker", result.stderr)
        self.assertFalse(self.cargo_marker.exists())

    def test_origin_url_rewrite_cannot_redirect_remote_proof(self) -> None:
        self.git("tag", "--annotate", "v0.7.0", "--message", "release")
        self.push_release_tag()
        self.git(
            "config",
            f"url.{self.remote.as_uri()}.insteadOf",
            self.origin_url,
        )
        result = self.run_publish(workflow_tag="v0.7.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("resolves outside", result.stderr)
        self.assertNotIn(str(self.remote), result.stderr)
        self.assertFalse(self.cargo_marker.exists())

    def test_missing_remote_tag_is_rejected_before_cargo(self) -> None:
        self.git("tag", "--annotate", "v0.7.0", "--message", "release")
        result = self.run_publish(workflow_tag="v0.7.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("is missing from canonical origin", result.stderr)
        self.assertFalse(self.cargo_marker.exists())

    def test_lightweight_remote_tag_is_rejected_before_cargo(self) -> None:
        self.git("tag", "--annotate", "v0.7.0", "--message", "release")
        self.push_release_tag("HEAD")
        result = self.run_publish(workflow_tag="v0.7.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("on canonical origin must be annotated", result.stderr)
        self.assertFalse(self.cargo_marker.exists())

    def test_stale_remote_tag_is_rejected_before_cargo(self) -> None:
        self.git("tag", "--annotate", "v0.7.0", "--message", "old release")
        self.push_release_tag()
        self.git("tag", "--delete", "v0.7.0")
        self.commit_change()
        self.git("tag", "--annotate", "v0.7.0", "--message", "new release")
        result = self.run_publish(workflow_tag="v0.7.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("object differs between canonical origin and checkout", result.stderr)
        self.assertFalse(self.cargo_marker.exists())

    def test_origin_errors_do_not_expose_embedded_credentials(self) -> None:
        self.git("tag", "--annotate", "v0.7.0", "--message", "release")
        secret = "release-token-that-must-not-leak"
        self.set_origin(f"https://{secret}@github.com/frames-sg/j2k.git")
        result = self.run_publish(workflow_tag="v0.7.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not a supported secure repository URL", result.stderr)
        self.assertNotIn(secret, result.stderr)
        self.assertFalse(self.cargo_marker.exists())

    def test_dirty_tracked_worktree_fails_before_cargo(self) -> None:
        self.git("tag", "--annotate", "v0.7.0", "--message", "release")
        (self.repository / "tracked.txt").write_text("dirty\n", encoding="utf-8")
        result = self.run_publish(workflow_tag="v0.7.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("clean worktree", result.stderr)
        self.assertFalse(self.cargo_marker.exists())

    def test_untracked_worktree_fails_before_cargo(self) -> None:
        self.git("tag", "--annotate", "v0.7.0", "--message", "release")
        (self.repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")
        result = self.run_publish(workflow_tag="v0.7.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("clean worktree", result.stderr)
        self.assertFalse(self.cargo_marker.exists())


if __name__ == "__main__":
    unittest.main()
