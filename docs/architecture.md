# Architecture

This document records current workspace boundaries. It is not a roadmap.

The public crate release centers on `j2k`. Runtime backend selection defaults to `Auto`: CPU remains the portable baseline, and explicit CUDA or Metal requests are strict. The workspace README records the public support and codec API policy.

The codec support boundary is JPEG 2000 Part 1 codestreams, JP2 still-image
files, HTJ2K Part 15 codestreams, and JPH still-image files. JPX / JPEG 2000
Part 2 extensions are out of scope unless required for standard JP2/JPH
still-image correctness. Keep row-level status synchronized with
[`docs/public-support.md`](public-support.md).

## Crate classes

| Crate | Class | Role |
| --- | --- | --- |
| `j2k` | public codec | Primary user-facing JPEG 2000 / HTJ2K API. |
| `j2k-core` | core | Shared traits, errors, geometry, pixel formats, backend requests, and device-surface contracts. |
| `j2k-jpeg`, `j2k-tilecodec` | codec | CPU/native codec implementations and stable codec APIs. |
| `j2k-native` | engine | Native JPEG 2000 / HTJ2K engine used by J2K APIs and adapter validation. |
| `j2k-profile`, `j2k-metal-support` | support | Runtime/profile helpers used by adapters and codec crates. |
| `j2k-cuda-runtime` | CUDA engine | CUDA Driver API integration, J2K-owned kernel modules, launch orchestration, and CUDA memory helpers shared by CUDA adapters. |
| `j2k-jpeg-cuda`, `j2k-cuda`, `j2k-transcode-cuda` | CUDA adapter | Codec-facing CUDA APIs, route policy, and CUDA device memory integration for supported paths. |
| `j2k-jpeg-metal`, `j2k-metal`, `j2k-transcode-metal` | Metal adapter | macOS Metal runtime integration for supported paths. |
| `j2k-transcode` | transcode | JPEG to J2K/HTJ2K transcode algorithms. |
| `j2k-cli` | CLI | Command-line inspection and JPEG-to-HTJ2K smoke transcode entry point. |
| `j2k-test-support` | dev helper | Shared fixture and benchmark input helpers for tests, benches, and examples. |
| `j2k-compare` | tooling | Comparator tooling. |
| `xtask` | workspace tool | Repository automation under `xtask/`. |

## Dependency rules

- The public `j2k` crate owns the JPEG 2000 / HTJ2K API surface.
- Codec crates may depend on `j2k-core` and support crates.
- Adapter crates may depend inward on codec/core/support crates.
- Support crates must not depend on adapters.
- Test support and comparator crates must not become runtime dependencies of
  stable public crates.
- CUDA paths must use J2K-owned CUDA kernels for codec stages they claim to
  support.

## Crate dependency graph

```text
j2k -> j2k-core, j2k-native, j2k-types
j2k-native -> j2k-types, j2k-profile
j2k-test-support -> j2k-native
j2k-cuda -> j2k-core, j2k-cuda-runtime, j2k, j2k-native, j2k-profile
j2k-metal -> j2k-core, j2k, j2k-native, j2k-metal-support, j2k-profile
j2k-jpeg -> j2k-core, j2k-profile
j2k-jpeg-cuda -> j2k-core, j2k-cuda-runtime, j2k-jpeg, j2k-profile
j2k-jpeg-metal -> j2k-core, j2k-jpeg, j2k-metal-support, j2k-profile
j2k-tilecodec -> j2k-core
j2k-compare -> j2k-core, j2k, j2k-native, j2k-test-support
j2k-transcode -> j2k-core, j2k, j2k-native, j2k-jpeg
j2k-metal-support -> j2k-core
j2k-cuda-runtime -> j2k-core
j2k-transcode-metal -> j2k-core, j2k-metal, j2k-metal-support, j2k-transcode
j2k-transcode-cuda -> j2k-cuda-runtime, j2k-native, j2k-transcode
j2k-cli -> j2k, j2k-jpeg, j2k-transcode
xtask -> j2k, j2k-compare, j2k-native, j2k-test-support
```

## Backend policy

CPU is the correctness baseline. Device adapters can add resident outputs and
stage acceleration, but they must preserve explicit unsupported errors for
unsupported requests.

CUDA adapters use `j2k-cuda-runtime`, which owns the shared CUDA Driver API
runtime, kernel modules, and host launch orchestration for supported CUDA codec
stages. `cuda-runtime` support is an implementation dependency, not proof of
NVIDIA performance.

Metal adapters use `j2k-metal-support` for device, queue, shader-library,
pipeline loading, and route-label helpers. Codec-specific kernels stay in codec
adapter crates.
