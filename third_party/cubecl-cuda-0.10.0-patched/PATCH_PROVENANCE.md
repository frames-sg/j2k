# Patch provenance

Source: `cubecl-cuda` 0.10.0 from crates.io, checksum
`b6b0a69ff45688d322ad8e92c8bf645167b9ca490fa8fa087fc6adac8c5e46be`.

Local change: implement CubeCL runtime's hidden external-write hook by
resolving the allocation binding on its owning CUDA stream and returning that
stream as an opaque numeric token. `j2k-ml` immediately passes the token to
`j2k-cuda-runtime`, which orders the codec and CubeCL streams with CUDA events.

Remove this patch when CubeCL exposes an equivalent external-write/event
ordering contract.
