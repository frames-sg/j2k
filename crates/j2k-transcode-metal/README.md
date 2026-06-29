# j2k-transcode-metal

Metal acceleration adapter for J2K JPEG-to-J2K/HTJ2K transcode stages on
macOS.

This crate accelerates supported transform stages and delegates runtime setup to
`j2k-metal-support`.

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
