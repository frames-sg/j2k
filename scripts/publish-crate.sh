#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: publish-crate.sh <crate>|--preflight-all" >&2
  exit 2
fi
requested="$1"
dry_run="${DRY_RUN_ONLY:-false}"

publishable_crates=(
 j2k-core
 j2k-profile
 j2k-types
 j2k-codec-math
 j2k-cuda-runtime
 j2k-metal-support
 j2k-native
 j2k-jpeg
 j2k-tilecodec
 j2k
 j2k-transcode
 j2k-transcode-cuda
 j2k-jpeg-metal
 j2k-metal
 j2k-transcode-metal
 j2k-jpeg-cuda
 j2k-cuda
 j2k-cli
)

registry_independent_crates=(
 j2k-core
 j2k-profile
 j2k-types
 j2k-codec-math
)

workspace_version() {
  awk '
    /^\[workspace.package\]/ { in_workspace_package = 1; next }
    /^\[/ && in_workspace_package { exit }
    in_workspace_package && $1 == "version" {
      gsub(/"/, "", $3)
      print $3
      exit
    }
  ' Cargo.toml
}

workspace_repository() {
  awk '
    /^\[workspace.package\]/ { in_workspace_package = 1; next }
    /^\[/ && in_workspace_package { exit }
    in_workspace_package && $1 == "repository" {
      gsub(/"/, "", $3)
      print $3
      exit
    }
  ' Cargo.toml
}

normalize_repository_identity() {
  local url="$1"
  local remainder
  local authority
  local host
  local path

  case "$url" in
    https://*)
      remainder="${url#https://}"
      authority="${remainder%%/*}"
      if [[ "$authority" == "$remainder" || "$authority" == *"@"* ]]; then
        return 1
      fi
      host="$authority"
      path="${remainder#*/}"
      if [[ "$host" == *":443" ]]; then
        host="${host%:443}"
      elif [[ "$host" == *":"* ]]; then
        return 1
      fi
      ;;
    git@*:*)
      remainder="${url#git@}"
      host="${remainder%%:*}"
      path="${remainder#*:}"
      if [[ "$host" == "$remainder" ]]; then
        return 1
      fi
      ;;
    ssh://*)
      remainder="${url#ssh://}"
      authority="${remainder%%/*}"
      if [[ "$authority" == "$remainder" ]]; then
        return 1
      fi
      path="${remainder#*/}"
      if [[ "$authority" == git@* ]]; then
        authority="${authority#git@}"
      elif [[ "$authority" == *"@"* ]]; then
        return 1
      fi
      if [[ "$authority" =~ ^([^:]+):[0-9]+$ ]]; then
        host="${BASH_REMATCH[1]}"
      elif [[ "$authority" == *":"* ]]; then
        return 1
      else
        host="$authority"
      fi
      ;;
    *) return 1 ;;
  esac

  path="${path#/}"
  path="${path%/}"
  path="${path%.git}"
  if [[ ! "$host" =~ ^[A-Za-z0-9.-]+$ \
     || ! "$path" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
    return 1
  fi
  printf '%s/%s\n' "$host" "$path" | tr '[:upper:]' '[:lower:]'
}

crate_version() {
  cargo pkgid -p "$1" | sed 's/.*#//'
}

is_publishable_crate() {
  local requested="$1"
  local publishable
  for publishable in "${publishable_crates[@]}"; do
    if [[ "$publishable" == "$requested" ]]; then
      return 0
    fi
  done
  return 1
}

require_publishable_crate() {
  if ! is_publishable_crate "$1"; then
    echo "${1}: not in the publishable release set; run cargo xtask release-integrity" >&2
    exit 1
  fi
}

require_canonical_origin_and_remote_tag() {
  local expected_tag="$1"
  local local_tag_object="$2"
  local expected_commit="$3"
  local canonical_repository
  local canonical_identity
  local origin_url
  local origin_identity
  local resolved_origin_url
  local resolved_origin_identity
  local remote_refs
  local remote_tag_object=""
  local remote_tag_commit=""
  local remote_sha
  local remote_ref

  canonical_repository="$(workspace_repository)"
  if [[ -z "$canonical_repository" ]] \
    || ! canonical_identity="$(normalize_repository_identity "$canonical_repository")"; then
    echo "failed to read a canonical workspace repository identity" >&2
    exit 1
  fi
  origin_url="$(git config --get-all remote.origin.url 2>/dev/null || true)"
  if [[ -z "$origin_url" ]] \
    || ! origin_identity="$(normalize_repository_identity "$origin_url")"; then
    echo "release origin is missing or is not a supported secure repository URL" >&2
    exit 1
  fi
  if [[ "$origin_identity" != "$canonical_identity" ]]; then
    echo "release origin does not match the canonical workspace repository" >&2
    exit 1
  fi
  resolved_origin_url="$(git remote get-url --all origin 2>/dev/null || true)"
  if [[ -z "$resolved_origin_url" ]] \
    || ! resolved_origin_identity="$(normalize_repository_identity "$resolved_origin_url")"; then
    echo "release origin resolves outside the supported canonical repository identity" >&2
    exit 1
  fi
  if [[ "$resolved_origin_identity" != "$canonical_identity" ]]; then
    echo "release origin resolves outside the canonical workspace repository" >&2
    exit 1
  fi

  if ! remote_refs="$(git ls-remote --tags origin \
    "refs/tags/${expected_tag}" "refs/tags/${expected_tag}^{}" 2>/dev/null)"; then
    echo "failed to verify the release tag on canonical origin" >&2
    exit 1
  fi
  while IFS=$'\t' read -r remote_sha remote_ref; do
    [[ -z "$remote_sha" && -z "$remote_ref" ]] && continue
    case "$remote_ref" in
      "refs/tags/${expected_tag}")
        if [[ -n "$remote_tag_object" ]]; then
          echo "canonical origin returned duplicate release tag objects" >&2
          exit 1
        fi
        remote_tag_object="$remote_sha"
        ;;
      "refs/tags/${expected_tag}^{}")
        if [[ -n "$remote_tag_commit" ]]; then
          echo "canonical origin returned duplicate peeled release tags" >&2
          exit 1
        fi
        remote_tag_commit="$remote_sha"
        ;;
      *)
        echo "canonical origin returned unexpected release tag evidence" >&2
        exit 1
        ;;
    esac
  done <<< "$remote_refs"

  if [[ -z "$remote_tag_object" ]]; then
    echo "release tag ${expected_tag} is missing from canonical origin" >&2
    exit 1
  fi
  if [[ -z "$remote_tag_commit" ]]; then
    echo "release tag ${expected_tag} on canonical origin must be annotated" >&2
    exit 1
  fi
  if [[ "$remote_tag_object" != "$local_tag_object" ]]; then
    echo "release tag ${expected_tag} object differs between canonical origin and checkout" >&2
    exit 1
  fi
  if [[ "$remote_tag_commit" != "$expected_commit" ]]; then
    echo "release tag ${expected_tag} on canonical origin does not peel to verified HEAD" >&2
    exit 1
  fi
}

require_release_preflight() {
  local expected_version="$1"
  local subject="$2"
  local actual_workspace_version
  local expected_tag
  local tag_object_type
  local tag_object_sha
  local tag_commit
  local head_commit
  local workflow_tag
  local worktree_state
  local publishable
  local publishable_version

  actual_workspace_version="$(workspace_version)"
  if [[ -z "$actual_workspace_version" ]]; then
    echo "failed to read workspace.package.version from Cargo.toml" >&2
    exit 1
  fi
  if [[ "$expected_version" != "$actual_workspace_version" ]]; then
    echo "${subject}: package version ${expected_version} does not match workspace version ${actual_workspace_version}" >&2
    exit 1
  fi

  expected_tag="v${actual_workspace_version}"
  if ! git show-ref --verify --quiet "refs/tags/${expected_tag}"; then
    echo "real publish requires annotated tag ${expected_tag}; that tag does not exist in this checkout" >&2
    exit 1
  fi
  tag_object_type="$(git cat-file -t "refs/tags/${expected_tag}" 2>/dev/null || true)"
  if [[ "$tag_object_type" != "tag" ]]; then
    echo "real publish requires ${expected_tag} to be an annotated tag" >&2
    exit 1
  fi
  tag_object_sha="$(git rev-parse --verify "refs/tags/${expected_tag}" 2>/dev/null || true)"
  tag_commit="$(git rev-parse --verify "refs/tags/${expected_tag}^{commit}" 2>/dev/null || true)"
  head_commit="$(git rev-parse --verify "HEAD^{commit}" 2>/dev/null || true)"
  if [[ -z "$tag_commit" || -z "$head_commit" || "$tag_commit" != "$head_commit" ]]; then
    echo "real publish requires annotated tag ${expected_tag} to peel exactly to HEAD" >&2
    exit 1
  fi
  workflow_tag="${GITHUB_REF_NAME:-}"
  if [[ -n "$workflow_tag" && "$workflow_tag" != "$expected_tag" ]]; then
    echo "GITHUB_REF_NAME ${workflow_tag} does not match verified tag ${expected_tag}" >&2
    exit 1
  fi
  if ! worktree_state="$(git status --porcelain=v1 --untracked-files=all)"; then
    echo "failed to inspect the release worktree" >&2
    exit 1
  fi
  if [[ -n "$worktree_state" ]]; then
    echo "real publish requires a clean worktree with no tracked or untracked changes" >&2
    exit 1
  fi
  require_canonical_origin_and_remote_tag "$expected_tag" "$tag_object_sha" "$head_commit"

  cargo xtask release-integrity --publish

  for publishable in "${publishable_crates[@]}"; do
    publishable_version="$(crate_version "$publishable")"
    if [[ "$publishable_version" != "$actual_workspace_version" ]]; then
      echo "${publishable}: version ${publishable_version} does not match workspace version ${actual_workspace_version}" >&2
      exit 1
    fi
  done
}

is_registry_independent_crate() {
  local requested_crate="$1"
  local independent
  for independent in "${registry_independent_crates[@]}"; do
    if [[ "$independent" == "$requested_crate" ]]; then
      return 0
    fi
  done
  return 1
}

published_rerun_enabled() {
  case "${CRATES_IO_ALLOW_PUBLISHED_RERUN:-false}" in
    true | 1)
      return 0
      ;;
    *) return 1 ;;
  esac
}

require_positive_decimal() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive decimal integer" >&2
    exit 1
  fi
}

require_nonnegative_decimal() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "${name} must be a nonnegative decimal integer" >&2
    exit 1
  fi
}

case "${CRATES_IO_ALLOW_PUBLISHED_RERUN:-false}" in
  true | false | 1 | 0) ;;
  *)
    echo "CRATES_IO_ALLOW_PUBLISHED_RERUN must be true, false, 1, or 0" >&2
    exit 1
    ;;
esac

case "$dry_run" in
  true | false) ;;
  *)
    echo "DRY_RUN_ONLY must be true or false" >&2
    exit 1
    ;;
esac

max_attempts="${CRATES_IO_PUBLISH_ATTEMPTS:-3}"
retry_seconds="${CRATES_IO_RATE_LIMIT_RETRY_SECONDS:-330}"
settle_seconds="${CRATES_IO_INDEX_SETTLE_SECONDS:-30}"
require_positive_decimal "CRATES_IO_PUBLISH_ATTEMPTS" "$max_attempts"
require_nonnegative_decimal "CRATES_IO_RATE_LIMIT_RETRY_SECONDS" "$retry_seconds"
require_nonnegative_decimal "CRATES_IO_INDEX_SETTLE_SECONDS" "$settle_seconds"

if [[ "$requested" == "--preflight-all" ]]; then
  version="$(workspace_version)"
  require_release_preflight "$version" "release set"
  version_args=()
  for crate in "${publishable_crates[@]}"; do
    version_args+=(--crate "$crate")
  done
  if published_rerun_enabled; then
    version_args+=(--allow-published-rerun)
  fi
  python3 scripts/crates_io_version.py verify-set \
    --version "$version" \
    "${version_args[@]}"
  exit 0
fi

crate="$requested"
require_publishable_crate "$crate"

if [[ "$dry_run" == "true" ]]; then
  if ! is_registry_independent_crate "$crate"; then
    echo "${crate}: constructing package without registry verification because its workspace dependencies are staged for publication"
    cargo package -p "$crate" --no-verify
    exit 0
  fi

  cargo publish -p "$crate" --dry-run
  exit 0
fi

version="$(workspace_version)"
require_release_preflight "$version" "$crate"
: "${CRATES_IO_API_TOKEN:?CRATES_IO_API_TOKEN is required for a real publish}"

version_state="$(python3 scripts/crates_io_version.py state --crate "$crate" --version "$version")"
if [[ "$version_state" == "published" ]]; then
  if published_rerun_enabled; then
    echo "${crate} ${version} is already published; idempotent rerun allowed"
    exit 0
  fi
  echo "${crate} ${version} is already published; set CRATES_IO_ALLOW_PUBLISHED_RERUN=true for an idempotent rerun" >&2
  exit 1
fi
if [[ "$version_state" != "available" ]]; then
  echo "${crate} ${version} returned unknown crates.io state ${version_state}" >&2
  exit 1
fi

export CARGO_REGISTRY_TOKEN="$CRATES_IO_API_TOKEN"
attempt=1

while true; do
  set +e
  output="$(cargo publish -p "$crate" 2>&1)"
  status=$?
  set -e
  printf '%s\n' "$output"

  if [[ "$status" -eq 0 ]]; then
    break
  fi

  if [[ "$output" != *"Too Many Requests"* || "$attempt" -ge "$max_attempts" ]]; then
    exit "$status"
  fi

  attempt=$((attempt + 1))
  echo "crates.io rate limited ${crate}; sleeping ${retry_seconds}s before retry ${attempt}/${max_attempts}"
  sleep "$retry_seconds"
done

sleep "$settle_seconds"
