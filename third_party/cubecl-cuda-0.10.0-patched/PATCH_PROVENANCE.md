# Patch provenance

Source: `cubecl-cuda` 0.10.0 from crates.io.

Pinned SHA-256 digests:

- crates.io archive: `b6b0a69ff45688d322ad8e92c8bf645167b9ca490fa8fa087fc6adac8c5e46be`
- upstream `src/compute/command.rs`: `ea7697c7fb33fd28a598ff05bb2d7ef6f07b21c1ecd68eff57a1680dcd68b797`
- patched local `src/compute/command.rs`: `d71de50b6035d5b98895e21b0fd6994a389b25240f48812e8a1231a0b6cdbe5c`
- upstream `src/compute/server.rs`: `7411535c6ae5c72efac9a85ec42d451a8f7ece4e890c1d4ecf35ff4ff4cd2a1d`
- patched local `src/compute/server.rs`: `ad81a690415f228c5a43f9972f06a3ae1bc7ba8673e3a545d79e15a6655d5a3c`
- patched tree inventory, excluding this provenance file and generated root lockfile: `bc1d7a26f591d748244ac3b18d4d641969d03506dff9d0f1c3ce73e10086166d`

Local change: implement CubeCL runtime's hidden external-write hook by
resolving the allocation binding on its owning CUDA stream and returning that
stream as an opaque numeric token. `j2k-ml` immediately passes the token to
`j2k-cuda-runtime`, which orders the codec and CubeCL streams with CUDA events.

Remove this patch when CubeCL exposes an equivalent external-write/event
ordering contract.

## Release approval

- Reviewer identity: `greg`
- Approval date: `2026-07-22`

The reviewer approved the pinned source and documented external-write/event
ordering delta for the 0.7.5 release.
