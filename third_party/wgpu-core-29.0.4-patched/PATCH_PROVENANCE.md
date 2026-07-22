# Patch provenance

Source: `wgpu-core` 29.0.4 from crates.io.

Pinned SHA-256 digests:

- crates.io archive: `2f519832254e56965a9940c4af57dcb75f702b6f6fa4a0b172f685395843a4d7`
- upstream `src/device/global.rs`: `8d8554b2400b46b595af68b0d2fab472da06b8c11fa1823299728b5bc2cf2caa`
- patched local `src/device/global.rs`: `ceab63ff6129a22e1bea8e9e1e691edb745080541d9c65b3ec043981d0a21978`
- patched tree inventory, excluding this provenance file and generated root lockfile: `13631d9cf9230ea52ae5458c440d60ccf23988839b9d89ff48cb026e7c324411`

Local change: add the hidden core half of the audited external-write handoff.
It validates that the requested byte range belongs to a live buffer and drains
only that range from wgpu's lazy buffer-initialization tracker. It does not
submit GPU work, change resource state, or establish synchronization; those
obligations remain in the unsafe public wrapper and the sole `j2k-ml` caller.

This patch exists only to prevent wgpu from clearing a uniquely owned Burn
allocation after J2K's Metal producer has initialized it. It must be removed
when the targeted wgpu/Burn/CubeCL versions provide an upstream safe
external-write initialization and dependency-registration mechanism.

## Release approval

- Reviewer identity: `greg`
- Approval date: `2026-07-22`

The reviewer approved the pinned source and documented external-write
initialization delta for the 0.7.5 release.
