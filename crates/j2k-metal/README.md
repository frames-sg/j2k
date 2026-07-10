# j2k-metal

Metal adapter for JPEG 2000 / HTJ2K decode and encode-stage paths on macOS.

The crate provides resident Metal decode and encode-stage integration for
supported workloads. It uses `j2k-metal-support` for runtime setup while
keeping codec-specific kernels local.

Encode support is stage-oriented unless a documented resident path accepts the
shape. `Auto` keeps single-frame HTJ2K host-output encode on CPU because the
measured resident path only clearly wins after batching amortizes setup cost.
Full resident host-output packetization/assembly is a batch path: batched Gray8
can use it at the 512 x 512 stage gate, while batched RGB8 requires 1,024 x
1,024 or larger resident input. Explicit Metal requests are strict: supported
shapes dispatch, and unsupported direct Metal requests return
`UnsupportedMetalRequest` instead of silently changing backend.

Metal routing is deliberately selective. `Auto` may decline small tiles,
irregular packet shapes, or stages where host/device transfer and dispatch
overhead dominate. Stage-by-stage host-output `Auto` currently limits Metal to
deinterleave, forward RCT/ICT, forward 5/3 and 9/7 DWT, and subband
quantization. Classic Tier-1, HT code-block encode, packetization, and
codestream assembly stay CPU for that route unless a documented resident path
supports the shape with parity and benchmark evidence.

## Full Resident Encode Path

Use `submit_lossless_batch_to_metal` when the output should remain a
Metal-backed codestream buffer. This is the full resident contract: coefficient
prep, packetization, and codestream assembly must all report `true` in
`MetalLosslessEncodeResidency` before the row is described as a full resident
encode path.

The resident path expects `MetalLosslessEncodeTile` inputs with
`MetalEncodeInputStaging::AlreadyPaddedContiguous` for no-copy Metal-buffer
workflows. For supported host-visible outputs, call
`submit_lossless_batch(...).wait()` to resolve the submission to
`Vec<EncodedJ2k>`. The hidden `encode_lossless_batch_with_report` helper is for
internal benchmarking and diagnostics, not the normal application contract.
When collecting benchmark diagnostics, report host readback separately from
resident buffer timing.

Keep benchmark claims scoped: compare `resident_host_ms` against CPU only when
`packetization_used=true`, `codestream_assembly_used=true`, and `batch_size > 1`.
Treat `resident_buffer_ms` as device-pipeline context unless the consumer can
keep the codestream buffer resident.

Run the decode route-report example to inspect Auto CPU fallback and strict
Metal behavior:

```bash
cargo run -p j2k-metal --example decode_route_report
```

Run the Auto HTJ2K encode report example to inspect final backend selection and
per-stage Metal dispatch counts:

```bash
cargo run -p j2k-metal --example htj2k_encode_auto_report
```

Run the resident encode example on macOS to produce a Metal-backed HTJ2K
codestream buffer and validate it through the CPU decoder:

```bash
cargo run -p j2k-metal --example resident_encode_buffer
```

## Links

- API docs: <https://docs.rs/j2k-metal>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
