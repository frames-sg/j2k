#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0

# Narrow shared mechanics for pinned external reference builds. Callers retain
# ownership of source pins, CMake flags, artifact candidates, and environment names.

reference_prepare_checkout() {
  local label="$1"
  local source_dir="$2"
  local source_url="$3"
  local source_tag="$4"
  local source_commit="$5"

  if [[ -e "${source_dir}" && ! -d "${source_dir}/.git" ]]; then
    echo "${label} target exists but is not a Git checkout: ${source_dir}" >&2
    return 1
  fi
  if [[ ! -d "${source_dir}/.git" ]]; then
    mkdir -p "$(dirname "${source_dir}")"
    git clone \
      --branch "${source_tag}" \
      --depth 1 \
      --filter=blob:none \
      --single-branch \
      "${source_url}" \
      "${source_dir}"
  fi

  local actual_commit
  local actual_tag
  local actual_remote
  local tracked_changes
  actual_commit="$(git -C "${source_dir}" rev-parse HEAD)"
  actual_tag="$(git -C "${source_dir}" describe --tags --exact-match HEAD)"
  actual_remote="$(git -C "${source_dir}" remote get-url origin)"
  tracked_changes="$(git -C "${source_dir}" status --porcelain --untracked-files=no)"
  if [[ "${actual_commit}" != "${source_commit}" \
    || "${actual_tag}" != "${source_tag}" \
    || "${actual_remote}" != "${source_url}" \
    || -n "${tracked_changes}" ]]; then
    echo "${label} checkout does not match the clean pinned official source" >&2
    return 1
  fi
}

reference_find_artifact() {
  local missing_message="$1"
  shift
  local candidate
  for candidate in "$@"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  echo "${missing_message}" >&2
  return 1
}

reference_canonical_dir() {
  local path
  path="$(cd "$1" && pwd -P)"
  reference_platform_path "${path}"
}

reference_canonical_file() {
  local path
  path="$(cd "$(dirname "$1")" && pwd -P)/$(basename "$1")"
  reference_platform_path "${path}"
}

reference_platform_path() {
  local path="$1"
  if command -v cygpath >/dev/null 2>&1 \
    && [[ "${RUNNER_OS:-}" == "Windows" || "${OSTYPE:-}" == msys* ]]; then
    cygpath -w "${path}"
  else
    printf '%s\n' "${path}"
  fi
}

reference_emit_env() {
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    printf '%s\n' "$@" >> "${GITHUB_ENV}"
  else
    printf '%s\n' "$@"
  fi
}
