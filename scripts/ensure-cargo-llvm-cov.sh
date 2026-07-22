#!/usr/bin/env bash
set -euo pipefail

readonly expected="cargo-llvm-cov 0.8.7"

if current="$(cargo llvm-cov --version 2>/dev/null)" && [[ "${current}" == "${expected}" ]]; then
  exit 0
fi

RUSTFLAGS= cargo install cargo-llvm-cov --version 0.8.7 --locked --force

actual="$(cargo llvm-cov --version)"
if [[ "${actual}" != "${expected}" ]]; then
  echo "expected ${expected}, found ${actual}" >&2
  exit 1
fi
