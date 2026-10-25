# SPDX-License-Identifier: MIT OR Apache-2.0
from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts import rerun_gpu_evidence as rerun
from scripts.tests.test_github_actions_verify import (
    FakeApi, STALE_SHA, add_jobs, add_runs, workflow_job,
    workflow_metadata, workflow_run,
)


class RerunGpuEvidenceTests(unittest.TestCase):
    def test_only_latest_exact_sha_failed_evidence_is_retried(self) -> None:
        for case in ("retry", "fork", "stale", "wrong-workflow", "push", "success",
                     "cancelled", "missing", "already-retried", "newer-success",
                     "gpu-failed", "gpu-push", "gpu-path", "wait"):
            with self.subTest(case=case), patch.object(rerun.subprocess, "run") as post, \
                    patch.object(rerun.time, "sleep") as sleep:
                api = FakeApi()
                workflow_metadata(api, "ci.yml", 77)
                event = workflow_run(100, head_branch="review/197" if case == "fork" else "feature")
                event["updated_at"] = "2026-10-25T10:00:00Z"
                run = workflow_run(2, path=".github/workflows/ci.yml", event="pull_request",
                                   conclusion="failure", head_branch="fork-feature")
                job = workflow_job("GPU evidence policy", conclusion="failure")
                job["started_at"] = "2026-10-25T09:00:00Z"
                if case == "stale":
                    run["head_sha"] = STALE_SHA
                if case == "wrong-workflow":
                    run["workflow_id"] = 88
                if case == "push":
                    run["event"] = "push"
                if case in ("success", "cancelled"):
                    job["conclusion"] = case
                if case == "already-retried":
                    job["started_at"] = "2026-10-25T10:01:00Z"
                if case.startswith("gpu-"):
                    event[{"gpu-failed": "conclusion", "gpu-push": "event", "gpu-path": "path"}[case]] = "other"
                newer = dict(run, id=3, conclusion="success")
                add_runs(api, 77, [run, newer] if case == "newer-success" else [run])
                add_jobs(api, 2, [] if case == "missing" else [job])
                if case == "wait":
                    run["status"] = "in_progress"
                    sleep.side_effect = lambda _: run.update(status="completed")
                rerun.retry(api, event, "frames-sg/j2k")
                if case in ("retry", "fork", "wait"):
                    post.assert_called_once_with(
                        ["gh", "api", "--method", "POST",
                         "repos/frames-sg/j2k/actions/runs/2/rerun-failed-jobs"], check=True)
                else:
                    post.assert_not_called()
                self.assertEqual(sleep.call_count, int(case == "wait"))
