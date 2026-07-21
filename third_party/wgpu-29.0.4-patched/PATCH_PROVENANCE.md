# Patch provenance

Source: `wgpu` 29.0.4 from crates.io, checksum
`76e8840e1ba2881d4cbb18d2147627a56af426ff064c0401eb0c8410c6325d07`.

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
