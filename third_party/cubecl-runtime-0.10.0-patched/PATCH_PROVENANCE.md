# Patch provenance

Source: `cubecl-runtime` 0.10.0 from crates.io, checksum
`b68491bf5b3e997ae36bdc4e63b4ccd6d2f0e86b3b596a5d7a48d2b9e92622a0`.

Local change: add one hidden `ExternalWriteServer` hook and one unsafe
`ComputeClient::external_write_stream` method. The hook resolves the normal
CubeCL binding dependencies before returning a backend-native stream token.
The token is used only inside `j2k-ml`'s lifetime-guarded CUDA event bridge;
it is never part of a public J2K or Burn adapter API.

Remove this patch when CubeCL exposes an equivalent external-write/event
ordering contract.
