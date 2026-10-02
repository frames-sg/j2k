# j2k-metal

Metal adapter for JPEG 2000 / HTJ2K decode and encode stages on macOS.

The crate decodes into Metal buffers and runs encode stages on Metal for the
supported workloads. Runtime setup comes from `j2k-metal-support`; the codec
kernels live in this crate.

Since version 0.9, the expert constructors and raw buffer handoffs take
`objc2-metal` protocol objects. Pass retained protocol objects for devices,
queues, command buffers, and buffers that J2K should own, and borrowed
references otherwise. The older `metal-rs` types are no longer accepted.

Encoding runs stage by stage on Metal unless one of the resident paths below
accepts the shape. For lossless HTJ2K encode to host memory, `Auto` runs
coefficient preparation and HT Tier-1 on Metal and packetizes on the CPU for
RGB8 at 1,024 x 1,024 and Gray8/RGB8 at 2,048 x 2,048. The 512 x 512 cells and
Gray8 at 1,024 x 1,024 stay on the CPU. These thresholds were measured on an
Apple M4 Pro.

Fully resident packetization and codestream assembly is a separate batch
path: batched Gray8 can use it from 512 x 512, and batched RGB8 from
1,024 x 1,024. Explicit Metal requests run on Metal when the shape is
supported and otherwise return `UnsupportedMetalRequest`; they never switch to
another backend.

`Auto` skips Metal for small tiles, irregular packets, and stages where
transfer and dispatch cost more than they save. In the host-output encode
cells listed above, `Auto` can run deinterleave, forward RCT/ICT, forward 5/3
and 9/7 DWT, subband quantization, and HT Tier-1 on Metal. Classic Tier-1,
packetization, and codestream assembly stay on the CPU for that path.

## Fully resident encode

Use `submit_lossless_batch_to_metal` when the encoded codestream should stay in
a Metal buffer. An encode counts as fully resident only when
`MetalLosslessEncodeResidency` reports `true` for coefficient prep,
packetization, and codestream assembly.

For no-copy input from Metal buffers, pass `MetalLosslessEncodeTile` inputs
with `MetalEncodeInputStaging::AlreadyPaddedContiguous`. To get host output,
call `submit_lossless_batch(...).wait()`, which returns `Vec<EncodedJ2k>`. The
hidden `encode_lossless_batch_with_report` helper is for internal benchmarks
and diagnostics. When benchmarking, time host readback separately from the
resident buffer work.

Compare `resident_host_ms` with the CPU only when `packetization_used=true`,
`codestream_assembly_used=true`, and `batch_size > 1`. `resident_buffer_ms` is
only meaningful if the consumer keeps the codestream in GPU memory.

See which backend `Auto` picks for decoding, and how explicit Metal requests
behave:

```bash
cargo run -p j2k-metal --example decode_route_report
```

See the final backend and per-stage Metal dispatch counts for `Auto` HTJ2K
encoding:

```bash
cargo run -p j2k-metal --example htj2k_encode_auto_report
```

Encode an HTJ2K codestream into a Metal buffer and check it with the CPU
decoder:

```bash
cargo run -p j2k-metal --example resident_encode_buffer
```

## Links

- API docs: <https://docs.rs/j2k-metal>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
