#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0

set -euo pipefail

version="0.19.0"
source_url="https://github.com/osamu620/OpenHTJ2K.git"
source_commit="e0f7ae853220d1e359c438b0bb6ad6cb2b3899db"
source_dir="${1:-target/t803/openhtj2k-v${version}}"
build_dir="${source_dir}/build-reference"

if [[ -e "${source_dir}" && ! -d "${source_dir}/.git" ]]; then
  echo "OpenHTJ2K target exists but is not a Git checkout: ${source_dir}" >&2
  exit 1
fi
if [[ ! -d "${source_dir}/.git" ]]; then
  mkdir -p "$(dirname "${source_dir}")"
  git clone \
    --branch "v${version}" \
    --depth 1 \
    --filter=blob:none \
    --single-branch \
    "${source_url}" \
    "${source_dir}"
fi

actual_commit="$(git -C "${source_dir}" rev-parse HEAD)"
actual_tag="$(git -C "${source_dir}" describe --tags --exact-match HEAD)"
actual_remote="$(git -C "${source_dir}" remote get-url origin)"
tracked_changes="$(git -C "${source_dir}" status --porcelain --untracked-files=no)"
if [[ "${actual_commit}" != "${source_commit}" \
  || "${actual_tag}" != "v${version}" \
  || "${actual_remote}" != "${source_url}" \
  || -n "${tracked_changes}" ]]; then
  echo "OpenHTJ2K checkout does not match the clean pinned official source" >&2
  exit 1
fi

cmake \
  -S "${source_dir}" \
  -B "${build_dir}" \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DOPENHTJ2K_QUIC=OFF
cmake \
  --build "${build_dir}" \
  --config Release \
  --target open_htj2k_dec \
  --parallel 2

decoder=""
for candidate in \
  "${build_dir}/bin/open_htj2k_dec" \
  "${build_dir}/bin/open_htj2k_dec.exe" \
  "${build_dir}/bin/Release/open_htj2k_dec.exe"; do
  if [[ -f "${candidate}" ]]; then
    decoder="${candidate}"
    break
  fi
done
if [[ -z "${decoder}" ]]; then
  echo "OpenHTJ2K build did not produce open_htj2k_dec" >&2
  exit 1
fi

source_dir="$(cd "${source_dir}" && pwd -P)"
decoder="$(cd "$(dirname "${decoder}")" && pwd -P)/$(basename "${decoder}")"
if command -v cygpath >/dev/null 2>&1 \
  && [[ "${RUNNER_OS:-}" == "Windows" || "${OSTYPE:-}" == msys* ]]; then
  source_dir="$(cygpath -w "${source_dir}")"
  decoder="$(cygpath -w "${decoder}")"
fi

if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    echo "J2K_OPENHTJ2K_DEC_BIN=${decoder}"
    echo "J2K_OPENHTJ2K_SOURCE_DIR=${source_dir}"
  } >> "${GITHUB_ENV}"
else
  echo "J2K_OPENHTJ2K_DEC_BIN=${decoder}"
  echo "J2K_OPENHTJ2K_SOURCE_DIR=${source_dir}"
fi
