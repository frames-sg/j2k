# j2k-transcode-metal

Metal acceleration adapter for J2K JPEG-to-J2K/HTJ2K transcode stages on
macOS.

This crate accelerates supported transform stages and delegates runtime setup to
`j2k-metal-support`.

Auto routing is conservative by default. Single-job reversible 5/3 and 9/7
Metal transcode thresholds are disabled with `usize::MAX`, so single-tile
requests stay on the CPU unless callers explicitly lower
`with_auto_reversible_min_samples` or `with_auto_dwt97_min_samples`. Same-shape
batches use the shared 32-job / `224 * 224 * 32` floors. Auto also avoids the
staged 9/7 batch path when either tile axis exceeds 1024 samples; strict Metal
requests and caller-lowered thresholds remain explicit policy decisions. These
defaults are routing policy, not a speedup promise.

High-level route-report example:

```bash
cargo run -p j2k-transcode-metal --example jpeg_to_htj2k_route_report
```

The example prints the requested backend, selected transform backend, final
codestream output backend, structured Auto fallback reason, transfer bytes, and
the transcode pipeline residency map.

On macOS, `resident_codestream_buffer_from_metal_encoded_j2k` converts
buffer-backed `j2k-metal` encode output into the shared
`ResidentCodestreamBuffer` handoff descriptor with allocation and capacity
validation.

## Links

- API docs: <https://docs.rs/j2k-transcode-metal>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
