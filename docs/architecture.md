# Architecture

This document records current workspace boundaries. It is not a roadmap.

The facade release centers on `signinum`. Runtime backend selection defaults to `Auto`: CPU remains the portable baseline, and explicit CUDA or Metal requests are strict. The workspace README records the public support and WSI decode policy.

## Crate classes

| Crate | Class | Role |
| --- | --- | --- |
| `signinum` | facade | Primary user-facing API. |
| `signinum-core` | core | Shared traits, errors, geometry, pixel formats, backend requests, and device-surface contracts. |
| `signinum-jpeg`, `signinum-j2k`, `signinum-tilecodec` | codec | CPU/native codec implementations and stable codec APIs. |
| `signinum-j2k-native` | engine | Native JPEG 2000 / HTJ2K engine used by J2K APIs and adapter validation. |
| `signinum-profile`, `signinum-metal-support` | support | Runtime/profile helpers used by adapters and codec crates. |
| `signinum-cuda-runtime` | CUDA engine | CUDA Driver API integration, Signinum-owned kernel modules, launch orchestration, and CUDA memory helpers shared by CUDA adapters. |
| `signinum-jpeg-cuda`, `signinum-j2k-cuda`, `signinum-transcode-cuda` | CUDA adapter | Codec-facing CUDA APIs, route policy, and CUDA device memory integration for supported paths. |
| `signinum-jpeg-metal`, `signinum-j2k-metal`, `signinum-transcode-metal` | Metal adapter | macOS Metal runtime integration for supported paths. |
| `signinum-transcode` | transcode | JPEG to HTJ2K transcode algorithms. |
| `signinum-cli` | CLI | Command-line inspection entry point. |
| `signinum-test-support` | dev helper | Shared fixture and benchmark input helpers for tests, benches, and examples. |
| `signinum-j2k-compare` | tooling | Comparator tooling. |
| `xtask` | workspace tool | Repository automation under `xtask/`. |

## Dependency rules

- The facade may depend on stable codec crates and adapter crates behind
  features.
- Codec crates may depend on `signinum-core` and support crates.
- Adapter crates may depend inward on codec/core/support crates.
- Support crates must not depend on adapters.
- Test support and comparator crates must not become runtime dependencies of
  stable public crates.
- CUDA paths must use Signinum-owned CUDA kernels for codec stages they claim to
  support.

## Crate dependency graph

```text
signinum -> signinum-core, signinum-j2k, signinum-j2k-cuda, signinum-j2k-metal, signinum-jpeg, signinum-jpeg-cuda, signinum-jpeg-metal, signinum-tilecodec
signinum-j2k -> signinum-core, signinum-j2k-native, signinum-j2k-types
signinum-j2k-native -> signinum-j2k-types, signinum-profile
signinum-test-support -> signinum-j2k-native
signinum-j2k-cuda -> signinum-core, signinum-cuda-runtime, signinum-j2k, signinum-j2k-native, signinum-profile
signinum-j2k-metal -> signinum-core, signinum-j2k, signinum-j2k-native, signinum-metal-support, signinum-profile
signinum-jpeg -> signinum-core, signinum-profile
signinum-jpeg-cuda -> signinum-core, signinum-cuda-runtime, signinum-jpeg, signinum-profile
signinum-jpeg-metal -> signinum-core, signinum-jpeg, signinum-metal-support, signinum-profile
signinum-tilecodec -> signinum-core
signinum-j2k-compare -> signinum-core, signinum-j2k, signinum-test-support
signinum-transcode -> signinum-j2k, signinum-j2k-native, signinum-jpeg
signinum-metal-support -> signinum-core
signinum-cuda-runtime -> signinum-core
signinum-transcode-metal -> signinum-core, signinum-metal-support, signinum-transcode
signinum-transcode-cuda -> signinum-cuda-runtime, signinum-j2k-native, signinum-transcode
signinum-cli -> signinum-j2k, signinum-jpeg
```

## Backend policy

CPU is the correctness baseline. Device adapters can add resident outputs and
stage acceleration, but they must preserve explicit unsupported errors for
unsupported requests.

CUDA adapters use `signinum-cuda-runtime`, which owns the shared CUDA Driver API
runtime, kernel modules, and host launch orchestration for supported CUDA codec
stages. `cuda-runtime` support is an implementation dependency, not proof of
NVIDIA performance.

Metal adapters use `signinum-metal-support` for device, queue, shader-library,
pipeline loading, and route-label helpers. Codec-specific kernels stay in codec
adapter crates.
