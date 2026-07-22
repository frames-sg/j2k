# Patch provenance

Source: `wgpu` 29.0.4 from crates.io.

Pinned SHA-256 digests:

- crates.io archive: `76e8840e1ba2881d4cbb18d2147627a56af426ff064c0401eb0c8410c6325d07`
- upstream `src/api/buffer.rs`: `c7d2c13d8031a9416d5562ec970a46470da331544d3a5e15bc8fb7dae2a9de32`
- patched local `src/api/buffer.rs`: `19ba97b38b63fd491e480cfcc773fa2055f691a97ee727d7541a7255bbf82ba4`
- upstream `src/backend/wgpu_core.rs`: `8c0e1fab257ef93a86f6d6c93f1970f9756dececcaf3cebaa398a90a9d58efa9`
- patched local `src/backend/wgpu_core.rs`: `efdb2d9a3351718d29ca0a4a843b1e61cc778dcb5fb46a0c365e9dd79507101c`
- patched tree inventory, excluding this provenance file and generated root lockfile: `c958c8a0308bc8c1b3753f32f5696cc5d15ef50f3dec56b64eb1936d9a22a1f5`

Local change: add a hidden, unsafe, range-scoped
`Buffer::mark_external_write_initialized` hook and its private-construction
error type. The hook validates the requested buffer range and delegates only
to the patched `wgpu-core` lazy-initialization tracker; it does not submit,
synchronize, transition, or expose a backend handle. `j2k-ml` calls it only
while a fresh CubeCL allocation is uniquely owned, after validating that the
four-byte-rounded tracker range lies inside that exact suballocation and after
registering the Metal producer dependency on Burn's consumer queue.

This patch exists only because wgpu otherwise clears externally written bytes
on the first tracked use. It must be removed when the targeted wgpu/Burn/CubeCL
versions provide an upstream safe external-write handoff, or an external
allocation API that registers initialization and queue dependencies without a
local tracker hook.

## Release approval

- Reviewer identity: `greg`
- Approval date: `2026-07-22`

The reviewer approved the pinned source and documented external-write tracking
delta for the 0.7.5 release.
