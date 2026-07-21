# Patch provenance

Source: `wgpu-hal` 29.0.4 from crates.io, checksum
`97ace1c17727311c22a46e4e3faf56ea6de81af99dcc839bdfb54857b94d448d`.

Local change: add three Metal-only retained raw-handle accessors, for the
selected `MTLDevice`, its `MTLCommandQueue`, and an `MTLBuffer`. Each accessor
transfers a single Objective-C +1 retain as an opaque pointer. `j2k-ml`
immediately adopts that retain into the existing `metal` crate owner while the
corresponding wgpu resource guard is live; raw handles do not enter its public
API. The queue accessor lets `j2k-ml` pass Burn's exact native queue to the
codec before producer commit. Exact-queue submissions rely on queue order and
allocate no event bridge. A same-device, different-queue codec caller uses a
session-owned `MTLEvent` timeline; the legacy API that chooses a consumer queue
after submission retains its compatibility `MTLSharedEvent` bridge. The Burn
adapter uses the exact-queue route.

This patch exists only for Burn/wgpu-to-J2K external destination interop. It
must be removed when the targeted wgpu release exposes an equivalent audited
Metal buffer/device ownership bridge.
