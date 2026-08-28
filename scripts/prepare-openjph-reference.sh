#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
reference_common="reference-build-common.sh"
source "${script_dir}/${reference_common}"

version="0.31.0"
source_url="https://github.com/aous72/OpenJPH.git"
source_commit="c68064d0e4cad8e96bab9a068f6cc4e7799744fc"
source_dir="${1:-target/reference/openjph-${version}}"
build_dir="${source_dir}/build-reference"

reference_prepare_checkout \
  "OpenJPH" \
  "${source_dir}" \
  "${source_url}" \
  "${version}" \
  "${source_commit}"

cmake \
  -S "${source_dir}" \
  -B "${build_dir}" \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DOJPH_BUILD_EXECUTABLES=ON \
  -DOJPH_ENABLE_TIFF_SUPPORT=OFF
cmake \
  --build "${build_dir}" \
  --config Release \
  --target ojph_expand ojph_compress \
  --parallel 2

expand="$(reference_find_artifact \
  "OpenJPH build did not produce ojph_expand" \
  "${build_dir}/src/apps/ojph_expand/ojph_expand" \
  "${build_dir}/src/apps/ojph_expand/ojph_expand.exe" \
  "${build_dir}/src/apps/ojph_expand/Release/ojph_expand.exe")"

compress="$(reference_find_artifact \
  "OpenJPH build did not produce ojph_compress" \
  "${build_dir}/src/apps/ojph_compress/ojph_compress" \
  "${build_dir}/src/apps/ojph_compress/ojph_compress.exe" \
  "${build_dir}/src/apps/ojph_compress/Release/ojph_compress.exe")"

library="$(reference_find_artifact \
  "OpenJPH build did not produce the static reference library" \
  "${build_dir}/src/core/libopenjph.a" \
  "${build_dir}/src/core/openjph.lib" \
  "${build_dir}/src/core/Release/openjph.lib")"

source_dir="$(reference_canonical_dir "${source_dir}")"
expand="$(reference_canonical_file "${expand}")"
compress="$(reference_canonical_file "${compress}")"
lib_dir="$(reference_canonical_dir "$(dirname "${library}")")"

reference_emit_env \
  "J2K_OPENJPH_EXPAND_BIN=${expand}" \
  "J2K_OPENJPH_COMPRESS_BIN=${compress}" \
  "J2K_OPENJPH_SOURCE_DIR=${source_dir}" \
  "J2K_OPENJPH_LIB_DIR=${lib_dir}"
