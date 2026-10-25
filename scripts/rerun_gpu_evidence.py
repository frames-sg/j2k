# SPDX-License-Identifier: MIT OR Apache-2.0
"""Retry failed PR checks after successful exact-SHA GPU evidence arrives."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

from scripts.github_actions_verify import (
    GitHubApi, fetch_run_jobs, fetch_workflow_identity, fetch_workflow_runs, normalize_sha,
)


def retry(api: GitHubApi, event: dict, repository: str) -> None:
    if (event["conclusion"] != "success" or event["event"] != "workflow_dispatch"
            or event["path"] != ".github/workflows/gpu-validation.yml"):
        return
    sha = normalize_sha(event["head_sha"])
    identity = fetch_workflow_identity(api, "ci.yml")
    while True:
        runs = [run for run in fetch_workflow_runs(api, identity, sha)
                if run["head_sha"] == sha and run["event"] == "pull_request"
                and run["workflow_id"] == identity.workflow_id and run["path"] == identity.path]
        run = max(runs, key=lambda item: item["id"], default=None)
        if run is None:
            return
        # GPU validation can finish before the other PR jobs. Recheck the latest run.
        if run["status"] in {"queued", "in_progress", "pending", "waiting", "requested"}:
            time.sleep(30)
            continue
        if run["status"] == "completed" and run["conclusion"] == "failure":
            for job in fetch_run_jobs(api, run["id"]):
                if (job["name"] == "GPU evidence policy" and job["conclusion"] == "failure"
                        and job["status"] == "completed"
                        and job["started_at"] < event["updated_at"]):
                    # A retry started after this evidence arrived must never be retried again.
                    subprocess.run(["gh", "api", "--method", "POST",
                                    f"repos/{repository}/actions/runs/{run['id']}/rerun-failed-jobs"],
                                   check=True)
                    return
        return


if __name__ == "__main__":
    repository = os.environ["GITHUB_REPOSITORY"]
    api = GitHubApi(os.environ["GITHUB_API_URL"], repository, os.environ["GH_TOKEN"])
    retry(api, json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())["workflow_run"],
          repository)
