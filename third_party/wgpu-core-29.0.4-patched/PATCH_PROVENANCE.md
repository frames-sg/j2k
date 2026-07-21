# Patch provenance

Source: `wgpu-core` 29.0.4 from crates.io, checksum
`2f519832254e56965a9940c4af57dcb75f702b6f6fa4a0b172f685395843a4d7`.

Local change: add the hidden core half of the audited external-write handoff.
It validates that the requested byte range belongs to a live buffer and drains
only that range from wgpu's lazy buffer-initialization tracker. It does not
submit GPU work, change resource state, or establish synchronization; those
obligations remain in the unsafe public wrapper and the sole `j2k-ml` caller.

This patch exists only to prevent wgpu from clearing a uniquely owned Burn
allocation after J2K's Metal producer has initialized it. It must be removed
when the targeted wgpu/Burn/CubeCL versions provide an upstream safe
external-write initialization and dependency-registration mechanism.
