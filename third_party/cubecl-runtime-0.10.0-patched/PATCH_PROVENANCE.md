# Patch provenance

Source: `cubecl-runtime` 0.10.0 from crates.io.

Pinned SHA-256 digests:

- crates.io archive: `b68491bf5b3e997ae36bdc4e63b4ccd6d2f0e86b3b596a5d7a48d2b9e92622a0`
- upstream `src/client.rs`: `dd25c67906e3db68d05d916d45234e4bdf20cd5b65ac1ad2b00a2ddcb520e85a`
- patched local `src/client.rs`: `2ba96df7c895b975d9e171f3c2452fc61e887091af565384061bbd816d471e1b`
- upstream `src/server/base.rs`: `c6292c16d5a73d7a1570776ffd25ec10afc339b942dbc5a693d8c791feb8ef5b`
- patched local `src/server/base.rs`: `93ad450ed4aba9c39478be65afd428110118f6014dd080f79ced701612baafe2`
- patched tree inventory, excluding this provenance file and generated root lockfile: `0ae2be14bb10fc27c633c99b0a862dce8b237fc32768fd63eff0b6ad53ef0047`

Local change: add one hidden `ExternalWriteServer` hook and one unsafe
`ComputeClient::external_write_stream` method. The hook resolves the normal
CubeCL binding dependencies before returning a backend-native stream token.
The token is used only inside `j2k-ml`'s lifetime-guarded CUDA event bridge;
it is never part of a public J2K or Burn adapter API.

Remove this patch when CubeCL exposes an equivalent external-write/event
ordering contract.

## Release approval

- Reviewer identity: `greg`
- Approval date: `2026-07-22`

The reviewer approved the pinned source and documented external-write/event
ordering delta for the 0.7.5 release.
