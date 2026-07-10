#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Fail-closed GitHub Actions and release-evidence verification."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence


API_VERSION = "2022-11-28"
PAGE_SIZE = 100
MAX_PAGES = 1_000
MAX_TAG_DEPTH = 16
SHA_PATTERN = re.compile(r"[0-9a-fA-F]{40}\Z")

CUDA_PREFIXES = (
    "crates/j2k-cuda-runtime/",
    "crates/j2k-jpeg-cuda/",
    "crates/j2k-cuda/",
    "crates/j2k-transcode-cuda/",
)
METAL_PREFIXES = (
    "crates/j2k-metal-support/",
    "crates/j2k-jpeg-metal/",
    "crates/j2k-metal/",
    "crates/j2k-transcode-metal/",
)
SHARED_GPU_PREFIXES = ("crates/j2k-profile/",)
SHARED_GPU_EXACT_PATHS = frozenset(
    {
        ".github/CODEOWNERS",
        ".github/workflows/ci.yml",
        ".github/workflows/gpu-validation.yml",
        ".github/workflows/publish.yml",
        "scripts/github_actions_verify.py",
    }
)
CUDA_JOB = "CUDA API compatibility on x86_64"
METAL_JOB = "Metal validation on Apple Silicon"
RELEASE_CANDIDATE_JOB = "Release candidate aggregate"


class VerificationError(RuntimeError):
    """An expected verification condition was not met."""


def _dict(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise VerificationError(f"malformed GitHub response: {context} must be an object")
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise VerificationError(f"malformed GitHub response: {context} must be an array")
    return value


def _string(value: Any, context: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise VerificationError(f"malformed GitHub response: {context} must be a string")
    return value


def _integer(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise VerificationError(f"malformed GitHub response: {context} must be an integer")
    return value


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise VerificationError(
            f"malformed GitHub response: {context} must be a boolean"
        )
    return value


def normalize_sha(value: str, context: str = "SHA") -> str:
    if not SHA_PATTERN.fullmatch(value):
        raise VerificationError(f"{context} must be exactly 40 hexadecimal characters")
    return value.lower()


def _required_names(values: Iterable[str]) -> tuple[str, ...]:
    names = tuple(values)
    if not names or any(not name.strip() for name in names):
        raise VerificationError("at least one non-empty exact required job name is required")
    if len(set(names)) != len(names):
        raise VerificationError("exact required job names must not contain duplicates")
    return names


@dataclass(frozen=True)
class OptionalJsonResponse:
    found: bool
    payload: Any


class GitHubApi:
    """Small stdlib-only GitHub REST client whose errors never expose its token."""

    def __init__(
        self,
        api_url: str,
        repository: str,
        token: str,
        *,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        if not token:
            raise VerificationError("GitHub API token is not configured")
        repo_parts = repository.split("/")
        if len(repo_parts) != 2 or any(not part for part in repo_parts):
            raise VerificationError("repository must use owner/name form")
        self._base_url = api_url.rstrip("/")
        self._repository = "/".join(urllib.parse.quote(part, safe="") for part in repo_parts)
        self._token = token
        self._opener = opener

    def get_json(
        self, path: str, params: Mapping[str, str | int] | None = None
    ) -> Any:
        response = self._get_json(path, params, allow_not_found=False)
        if not response.found:
            raise VerificationError("internal GitHub API state error")
        return response.payload

    def get_optional_json(
        self, path: str, params: Mapping[str, str | int] | None = None
    ) -> OptionalJsonResponse:
        """Return found=false only for an authenticated HTTP 404 response."""

        return self._get_json(path, params, allow_not_found=True)

    def _get_json(
        self,
        path: str,
        params: Mapping[str, str | int] | None,
        *,
        allow_not_found: bool,
    ) -> OptionalJsonResponse:
        if not path.startswith("/"):
            raise VerificationError("internal API path must begin with a slash")
        query = urllib.parse.urlencode(params or {})
        url = f"{self._base_url}/repos/{self._repository}{path}"
        if query:
            url = f"{url}?{query}"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "X-GitHub-Api-Version": API_VERSION,
            },
        )
        try:
            with self._opener(request, timeout=30) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            error.close()
            if allow_not_found and error.code == 404:
                return OptionalJsonResponse(found=False, payload=None)
            raise VerificationError(
                f"GitHub API request failed with HTTP {error.code} for {path}"
            ) from None
        except urllib.error.URLError as error:
            reason = type(error.reason).__name__
            raise VerificationError(
                f"GitHub API request failed for {path} ({reason})"
            ) from None
        except (OSError, TimeoutError) as error:
            raise VerificationError(
                f"GitHub API request failed for {path} ({type(error).__name__})"
            ) from None

        try:
            return OptionalJsonResponse(found=True, payload=json.loads(raw))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise VerificationError(
                f"malformed GitHub response: {path} did not return valid JSON"
            ) from None


def fetch_pull_request_paths(api: GitHubApi, pr_number: int) -> tuple[str, ...]:
    if pr_number <= 0:
        raise VerificationError("pull request number must be positive")
    paths: set[str] = set()
    for page in range(1, MAX_PAGES + 1):
        payload = _list(
            api.get_json(
                f"/pulls/{pr_number}/files",
                {"per_page": PAGE_SIZE, "page": page},
            ),
            "pull request files",
        )
        for index, raw_file in enumerate(payload):
            file_entry = _dict(raw_file, f"pull request file {index}")
            paths.add(_string(file_entry.get("filename"), "pull request filename"))
            previous = file_entry.get("previous_filename")
            if previous is not None:
                paths.add(_string(previous, "pull request previous filename"))
        if len(payload) < PAGE_SIZE:
            return tuple(sorted(paths))
    raise VerificationError("pull request file pagination exceeded the safety limit")


@dataclass(frozen=True)
class GpuPolicyDecision:
    changed_gpu_paths: tuple[str, ...]
    required_jobs: tuple[str, ...]


def classify_gpu_paths(paths: Iterable[str]) -> GpuPolicyDecision:
    cuda_paths: set[str] = set()
    metal_paths: set[str] = set()
    shared_paths: set[str] = set()
    for path in paths:
        if path.startswith(CUDA_PREFIXES):
            cuda_paths.add(path)
        if path.startswith(METAL_PREFIXES):
            metal_paths.add(path)
        if path in SHARED_GPU_EXACT_PATHS or path.startswith(SHARED_GPU_PREFIXES):
            shared_paths.add(path)

    required: list[str] = []
    if cuda_paths or shared_paths:
        required.append(CUDA_JOB)
    if metal_paths or shared_paths:
        required.append(METAL_JOB)
    return GpuPolicyDecision(
        tuple(sorted(cuda_paths | metal_paths | shared_paths)), tuple(required)
    )


@dataclass(frozen=True)
class WorkflowIdentity:
    workflow_id: int
    path: str


def fetch_workflow_identity(api: GitHubApi, workflow: str) -> WorkflowIdentity:
    if not workflow or "/" in workflow or "\\" in workflow:
        raise VerificationError("workflow must be an exact workflow filename or numeric ID")
    encoded = urllib.parse.quote(workflow, safe="")
    payload = _dict(
        api.get_json(f"/actions/workflows/{encoded}"), "workflow metadata"
    )
    workflow_id = _integer(payload.get("id"), "workflow id")
    path = _string(payload.get("path"), "workflow path")
    if workflow.isdigit():
        if workflow_id != int(workflow):
            raise VerificationError(
                f"workflow ID mismatch: requested {workflow}, received {workflow_id}"
            )
    else:
        expected_path = f".github/workflows/{workflow}"
        if path != expected_path:
            raise VerificationError(
                f"workflow path mismatch: expected {expected_path}, received {path}"
            )
    return WorkflowIdentity(workflow_id, path)


def fetch_workflow_runs(
    api: GitHubApi, identity: WorkflowIdentity, sha: str
) -> list[Mapping[str, Any]]:
    runs: list[Mapping[str, Any]] = []
    for page in range(1, MAX_PAGES + 1):
        payload = _dict(
            api.get_json(
                f"/actions/workflows/{identity.workflow_id}/runs",
                {"head_sha": sha, "per_page": PAGE_SIZE, "page": page},
            ),
            "workflow runs",
        )
        raw_runs = _list(payload.get("workflow_runs"), "workflow_runs")
        runs.extend(
            _dict(run, f"workflow run on page {page}") for run in raw_runs
        )
        if len(raw_runs) < PAGE_SIZE:
            return runs
    raise VerificationError("workflow run pagination exceeded the safety limit")


def fetch_run_jobs(api: GitHubApi, run_id: int) -> list[Mapping[str, Any]]:
    jobs: list[Mapping[str, Any]] = []
    for page in range(1, MAX_PAGES + 1):
        payload = _dict(
            api.get_json(
                f"/actions/runs/{run_id}/jobs",
                {"per_page": PAGE_SIZE, "page": page},
            ),
            "workflow jobs",
        )
        raw_jobs = _list(payload.get("jobs"), "workflow jobs list")
        jobs.extend(_dict(job, f"workflow job on page {page}") for job in raw_jobs)
        if len(raw_jobs) < PAGE_SIZE:
            return jobs
    raise VerificationError("workflow job pagination exceeded the safety limit")


def verify_workflow_run(
    api: GitHubApi,
    workflow: str,
    sha: str,
    required_jobs: Sequence[str],
    *,
    required_event: str | None = None,
    required_head_branch: str | None = None,
) -> int:
    exact_sha = normalize_sha(sha, "workflow head SHA")
    names = _required_names(required_jobs)
    identity = fetch_workflow_identity(api, workflow)
    runs = fetch_workflow_runs(api, identity, exact_sha)
    failures: list[str] = []
    matching_run_count = 0

    for raw_run in runs:
        run_id = _integer(raw_run.get("id"), "workflow run id")
        run_sha = normalize_sha(
            _string(raw_run.get("head_sha"), "workflow run head_sha"),
            "workflow run head_sha",
        )
        run_workflow_id = _integer(
            raw_run.get("workflow_id"), "workflow run workflow_id"
        )
        run_path = _string(raw_run.get("path"), "workflow run path")
        status = _string(raw_run.get("status"), "workflow run status")
        event = _string(raw_run.get("event"), "workflow run event")
        head_branch_value = raw_run.get("head_branch")
        head_branch = (
            None
            if head_branch_value is None
            else _string(head_branch_value, "workflow run head_branch")
        )
        conclusion_value = raw_run.get("conclusion")
        conclusion = (
            None
            if conclusion_value is None
            else _string(conclusion_value, "workflow run conclusion")
        )

        if run_sha != exact_sha:
            failures.append(f"run {run_id} is stale ({run_sha})")
            continue
        matching_run_count += 1
        if run_workflow_id != identity.workflow_id or run_path != identity.path:
            failures.append(f"run {run_id} has a mismatched workflow identity")
            continue
        if required_event is not None and event != required_event:
            failures.append(
                f"run {run_id} has event {event}, expected {required_event}"
            )
            continue
        if required_head_branch is not None and head_branch != required_head_branch:
            failures.append(
                f"run {run_id} has head branch {head_branch or 'none'}, "
                f"expected {required_head_branch}"
            )
            continue
        if status != "completed" or conclusion != "success":
            failures.append(
                f"run {run_id} is {status}/{conclusion or 'no-conclusion'}"
            )
            continue

        jobs = fetch_run_jobs(api, run_id)
        required_observations: dict[str, list[tuple[str, str | None]]] = {
            name: [] for name in names
        }
        for job in jobs:
            name = _string(job.get("name"), "workflow job name")
            job_status = _string(job.get("status"), f"workflow job {name} status")
            job_conclusion_value = job.get("conclusion")
            job_conclusion = (
                None
                if job_conclusion_value is None
                else _string(
                    job_conclusion_value, f"workflow job {name} conclusion"
                )
            )
            if name in required_observations:
                required_observations[name].append((job_status, job_conclusion))

        duplicate_names = sorted(
            name for name, observations in required_observations.items() if len(observations) > 1
        )
        if duplicate_names:
            failures.append(
                f"run {run_id} contains duplicate required jobs: {', '.join(duplicate_names)}"
            )
            continue
        missing_names = sorted(
            name for name, observations in required_observations.items() if not observations
        )
        if missing_names:
            failures.append(
                f"run {run_id} is missing required jobs: {', '.join(missing_names)}"
            )
            continue
        unsuccessful = sorted(
            f"{name}={observations[0][0]}/{observations[0][1] or 'no-conclusion'}"
            for name, observations in required_observations.items()
            if observations[0] != ("completed", "success")
        )
        if unsuccessful:
            failures.append(
                f"run {run_id} has unsuccessful required jobs: {', '.join(unsuccessful)}"
            )
            continue
        return run_id

    if matching_run_count == 0:
        reason = "no run matched the exact candidate SHA"
    elif failures:
        reason = "; ".join(failures[:8])
    else:
        reason = "no completed successful run contained every exact required job"
    raise VerificationError(
        f"workflow {identity.path} has no acceptable single run for {exact_sha}: {reason}"
    )


def verify_repository_origin(
    origin_url: str, server_url: str, repository: str
) -> None:
    """Require the hosted checkout to point at the exact workflow repository."""

    repo_parts = repository.split("/")
    if len(repo_parts) != 2 or any(not part for part in repo_parts):
        raise VerificationError("repository must use owner/name form")
    parsed_server = urllib.parse.urlsplit(server_url)
    if (
        parsed_server.scheme != "https"
        or not parsed_server.netloc
        or parsed_server.username is not None
        or parsed_server.password is not None
        or parsed_server.path not in ("", "/")
        or parsed_server.query
        or parsed_server.fragment
    ):
        raise VerificationError(
            "GitHub server URL must be a bare HTTPS origin"
        )
    normalized_server = server_url.rstrip("/")
    expected = f"{normalized_server}/{repository}"
    if origin_url not in (expected, f"{expected}.git"):
        raise VerificationError(
            "git origin does not match the exact GitHub workflow repository"
        )


def require_github_release_absent(api: GitHubApi, tag: str) -> None:
    """Require publication to start before any GitHub Release exists for the tag."""

    if (
        not tag
        or tag.startswith("refs/")
        or any(ord(character) < 32 for character in tag)
    ):
        raise VerificationError("tag must be a non-empty short tag name")
    encoded_tag = urllib.parse.quote(tag, safe="")
    response = api.get_optional_json(f"/releases/tags/{encoded_tag}")
    if not response.found:
        return
    release = _dict(response.payload, "GitHub Release")
    response_tag = _string(release.get("tag_name"), "GitHub Release tag_name")
    draft = _boolean(release.get("draft"), "GitHub Release draft")
    prerelease = _boolean(release.get("prerelease"), "GitHub Release prerelease")
    if response_tag != tag:
        raise VerificationError(
            f"GitHub Release tag mismatch: expected {tag}, received {response_tag}"
        )
    state = "draft" if draft else "prerelease" if prerelease else "published"
    raise VerificationError(
        f"GitHub Release {tag} already exists in {state} state; "
        "publication preflight requires it to be absent"
    )


def peel_annotated_tag(api: GitHubApi, tag: str) -> str:
    if not tag or tag.startswith("refs/") or any(ord(character) < 32 for character in tag):
        raise VerificationError("tag must be a non-empty short tag name")
    encoded_tag = urllib.parse.quote(tag, safe="")
    ref_payload = _dict(
        api.get_json(f"/git/ref/tags/{encoded_tag}"), "tag ref"
    )
    expected_ref = f"refs/tags/{tag}"
    if _string(ref_payload.get("ref"), "tag ref name") != expected_ref:
        raise VerificationError(f"tag ref mismatch for {expected_ref}")
    target = _dict(ref_payload.get("object"), "tag ref object")
    target_type = _string(target.get("type"), "tag ref object type")
    target_sha = normalize_sha(
        _string(target.get("sha"), "tag ref object SHA"), "tag object SHA"
    )
    if target_type != "tag":
        raise VerificationError(f"release tag {tag} must be annotated, not {target_type}")

    seen: set[str] = set()
    for _ in range(MAX_TAG_DEPTH):
        if target_sha in seen:
            raise VerificationError(f"annotated tag {tag} contains a tag-object cycle")
        seen.add(target_sha)
        tag_payload = _dict(
            api.get_json(f"/git/tags/{target_sha}"), "annotated tag object"
        )
        response_sha = normalize_sha(
            _string(tag_payload.get("sha"), "annotated tag SHA"),
            "annotated tag SHA",
        )
        if response_sha != target_sha:
            raise VerificationError(f"annotated tag object SHA mismatch for {tag}")
        target = _dict(tag_payload.get("object"), "annotated tag target")
        target_type = _string(target.get("type"), "annotated tag target type")
        target_sha = normalize_sha(
            _string(target.get("sha"), "annotated tag target SHA"),
            "annotated tag target SHA",
        )
        if target_type == "commit":
            return target_sha
        if target_type != "tag":
            raise VerificationError(
                f"annotated tag {tag} peels to unsupported object type {target_type}"
            )
    raise VerificationError(f"annotated tag {tag} exceeded the peel-depth safety limit")


def verify_release_evidence(
    api: GitHubApi,
    *,
    repository: str,
    origin_url: str,
    server_url: str,
    tag: str,
    candidate_sha: str,
    ci_workflow: str,
    aggregate_job: str,
    gpu_workflow: str,
    cuda_job: str,
    metal_job: str,
    ci_branch: str,
) -> tuple[int, int]:
    verify_repository_origin(origin_url, server_url, repository)
    require_github_release_absent(api, tag)
    expected_sha = normalize_sha(candidate_sha, "candidate SHA")
    peeled_sha = peel_annotated_tag(api, tag)
    if peeled_sha != expected_sha:
        raise VerificationError(
            f"release tag {tag} peels to {peeled_sha}, not candidate {expected_sha}"
        )
    return verify_candidate_evidence(
        api,
        candidate_sha=expected_sha,
        ci_workflow=ci_workflow,
        aggregate_job=aggregate_job,
        gpu_workflow=gpu_workflow,
        cuda_job=cuda_job,
        metal_job=metal_job,
        ci_branch=ci_branch,
    )


def verify_candidate_evidence(
    api: GitHubApi,
    *,
    candidate_sha: str,
    ci_workflow: str,
    aggregate_job: str,
    gpu_workflow: str,
    cuda_job: str,
    metal_job: str,
    ci_branch: str,
) -> tuple[int, int]:
    """Verify all post-freeze release evidence for one exact commit SHA."""

    expected_sha = normalize_sha(candidate_sha, "candidate SHA")
    ci_run = verify_workflow_run(
        api,
        ci_workflow,
        expected_sha,
        [aggregate_job],
        required_event="push",
        required_head_branch=ci_branch,
    )
    gpu_run = verify_workflow_run(
        api,
        gpu_workflow,
        expected_sha,
        [cuda_job, metal_job],
        required_event="workflow_dispatch",
    )
    return ci_run, gpu_run


def _add_api_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument(
        "--api-url", default=os.environ.get("GITHUB_API_URL", "https://api.github.com")
    )
    parser.add_argument("--token-env", default="GH_TOKEN")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    workflow_parser = subparsers.add_parser(
        "verify-workflow", help="verify exact jobs in one exact-SHA workflow run"
    )
    _add_api_arguments(workflow_parser)
    workflow_parser.add_argument("--workflow", required=True)
    workflow_parser.add_argument("--sha", required=True)
    workflow_parser.add_argument("--required-job", action="append", required=True)
    workflow_parser.add_argument("--event")
    workflow_parser.add_argument("--head-branch")

    pr_parser = subparsers.add_parser(
        "pr-gpu-policy", help="classify PR GPU paths and verify required hardware jobs"
    )
    _add_api_arguments(pr_parser)
    pr_parser.add_argument("--pr-number", required=True, type=int)
    pr_parser.add_argument("--head-sha", required=True)
    pr_parser.add_argument("--workflow", default="gpu-validation.yml")

    release_parser = subparsers.add_parser(
        "verify-release", help="verify annotated tag and exact-SHA CI/GPU evidence"
    )
    _add_api_arguments(release_parser)
    release_parser.add_argument("--origin-url", required=True)
    release_parser.add_argument("--server-url", required=True)
    release_parser.add_argument("--tag", required=True)
    release_parser.add_argument("--candidate-sha", required=True)
    release_parser.add_argument("--ci-workflow", default="ci.yml")
    release_parser.add_argument("--aggregate-job", default=RELEASE_CANDIDATE_JOB)
    release_parser.add_argument("--gpu-workflow", default="gpu-validation.yml")
    release_parser.add_argument("--cuda-job", default=CUDA_JOB)
    release_parser.add_argument("--metal-job", default=METAL_JOB)
    release_parser.add_argument("--ci-branch", default="main")

    candidate_parser = subparsers.add_parser(
        "verify-candidate",
        help="verify exact-SHA CI aggregate and GPU evidence without requiring a tag",
    )
    _add_api_arguments(candidate_parser)
    candidate_parser.add_argument("--candidate-sha", required=True)
    candidate_parser.add_argument("--ci-workflow", default="ci.yml")
    candidate_parser.add_argument("--aggregate-job", default=RELEASE_CANDIDATE_JOB)
    candidate_parser.add_argument("--gpu-workflow", default="gpu-validation.yml")
    candidate_parser.add_argument("--cuda-job", default=CUDA_JOB)
    candidate_parser.add_argument("--metal-job", default=METAL_JOB)
    candidate_parser.add_argument("--ci-branch", default="main")
    return parser


def _api_from_args(args: argparse.Namespace) -> GitHubApi:
    if not args.repository:
        raise VerificationError("repository is required (or set GITHUB_REPOSITORY)")
    token = os.environ.get(args.token_env, "")
    return GitHubApi(args.api_url, args.repository, token)


def run_command(args: argparse.Namespace) -> None:
    api = _api_from_args(args)
    if args.command == "verify-workflow":
        run_id = verify_workflow_run(
            api,
            args.workflow,
            args.sha,
            args.required_job,
            required_event=args.event,
            required_head_branch=args.head_branch,
        )
        print(f"verified workflow run {run_id} for exact SHA {args.sha.lower()}")
        return
    if args.command == "pr-gpu-policy":
        paths = fetch_pull_request_paths(api, args.pr_number)
        decision = classify_gpu_paths(paths)
        if not decision.required_jobs:
            print("No GPU path changes detected.")
            return
        run_id = verify_workflow_run(
            api,
            args.workflow,
            args.head_sha,
            decision.required_jobs,
            required_event="workflow_dispatch",
        )
        print("GPU path changes:")
        for path in decision.changed_gpu_paths:
            print(f"  - {path}")
        print("Required GPU jobs verified in one exact-SHA run:")
        for job in decision.required_jobs:
            print(f"  - {job}")
        print(f"Verified workflow run: {run_id}")
        return
    if args.command == "verify-release":
        ci_run, gpu_run = verify_release_evidence(
            api,
            repository=args.repository,
            origin_url=args.origin_url,
            server_url=args.server_url,
            tag=args.tag,
            candidate_sha=args.candidate_sha,
            ci_workflow=args.ci_workflow,
            aggregate_job=args.aggregate_job,
            gpu_workflow=args.gpu_workflow,
            cuda_job=args.cuda_job,
            metal_job=args.metal_job,
            ci_branch=args.ci_branch,
        )
        print(
            f"verified origin, absent GitHub Release, annotated tag {args.tag}, "
            "and exact-SHA evidence "
            f"(CI run {ci_run}, GPU run {gpu_run})"
        )
        return
    if args.command == "verify-candidate":
        ci_run, gpu_run = verify_candidate_evidence(
            api,
            candidate_sha=args.candidate_sha,
            ci_workflow=args.ci_workflow,
            aggregate_job=args.aggregate_job,
            gpu_workflow=args.gpu_workflow,
            cuda_job=args.cuda_job,
            metal_job=args.metal_job,
            ci_branch=args.ci_branch,
        )
        print(
            f"verified exact-SHA release candidate {args.candidate_sha.lower()} "
            f"(CI run {ci_run}, GPU run {gpu_run})"
        )
        return
    raise VerificationError(f"unsupported command {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_command(args)
    except VerificationError as error:
        print(f"verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
