#!/usr/bin/env bash
set -euo pipefail

crate="${1:?usage: publish-crate.sh <crate>}"
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

require_release_preflight() {
  local expected_version="$1"
  local actual_workspace_version
  local expected_tag
  local actual_tag
  local publishable
  local publishable_version

  actual_workspace_version="$(workspace_version)"
  if [[ -z "$actual_workspace_version" ]]; then
    echo "failed to read workspace.package.version from Cargo.toml" >&2
    exit 1
  fi
  if [[ "$expected_version" != "$actual_workspace_version" ]]; then
    echo "${crate}: package version ${expected_version} does not match workspace version ${actual_workspace_version}" >&2
    exit 1
  fi

  expected_tag="v${actual_workspace_version}"
  actual_tag="${GITHUB_REF_NAME:-}"
  if [[ -z "$actual_tag" ]]; then
    actual_tag="$(git describe --tags --exact-match 2>/dev/null || true)"
  fi
  if [[ "$actual_tag" != "$expected_tag" ]]; then
    echo "real publish requires tag ${expected_tag}; current tag is ${actual_tag:-<none>}" >&2
    exit 1
  fi

  for publishable in "${publishable_crates[@]}"; do
    publishable_version="$(crate_version "$publishable")"
    if [[ "$publishable_version" != "$actual_workspace_version" ]]; then
      echo "${publishable}: version ${publishable_version} does not match workspace version ${actual_workspace_version}" >&2
      exit 1
    fi
  done
}

require_publishable_crate "$crate"
version="$(crate_version "$crate")"

has_unpublished_workspace_dependency() {
  case "$1" in
   j2k-cuda-runtime | \
     j2k-metal-support | \
     j2k-native | \
     j2k-jpeg | \
     j2k-tilecodec | \
     j2k | \
     j2k-transcode | \
     j2k-transcode-cuda | \
     j2k-jpeg-metal | \
     j2k-metal | \
     j2k-transcode-metal | \
     j2k-jpeg-cuda | \
     j2k-cuda | \
     j2k-cli)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

if [[ "$dry_run" == "true" ]]; then
  if has_unpublished_workspace_dependency "$crate"; then
    echo "${crate}: dry-run package list only; unpublished workspace dependencies make cargo publish --dry-run invalid before staged publication"
    cargo package -p "$crate" --list
    exit 0
  fi

  cargo publish -p "$crate" --dry-run
  exit 0
fi

require_release_preflight "$version"
: "${CRATES_IO_API_TOKEN:?CRATES_IO_API_TOKEN is required for a real publish}"

if cargo info "${crate}@${version}" --registry crates-io >/dev/null 2>&1; then
  case "${CRATES_IO_ALLOW_PUBLISHED_RERUN:-false}" in
    true | 1)
      echo "${crate} ${version} is already published; idempotent rerun allowed"
      exit 0
      ;;
    *)
      echo "${crate} ${version} is already published; set CRATES_IO_ALLOW_PUBLISHED_RERUN=true for an idempotent rerun" >&2
      exit 1
      ;;
  esac
fi

export CARGO_REGISTRY_TOKEN="$CRATES_IO_API_TOKEN"
attempt=1
max_attempts="${CRATES_IO_PUBLISH_ATTEMPTS:-3}"
retry_seconds="${CRATES_IO_RATE_LIMIT_RETRY_SECONDS:-330}"

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

sleep "${CRATES_IO_INDEX_SETTLE_SECONDS:-30}"
