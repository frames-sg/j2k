# j2k-transcode-metal

Metal acceleration for J2K's JPEG-to-HTJ2K coefficient-domain transcode on
macOS.

This crate runs the supported transform stages on Metal and uses
`j2k-metal-support` for runtime setup.

Since version 0.9, device constructors take a retained `objc2-metal`
`ProtocolObject<dyn MTLDevice>`. The `metal-rs::Device` constructors were
removed.

By default `Auto` keeps single tiles on the CPU: the single-job reversible
5/3 and 9/7 thresholds are `usize::MAX` unless you lower them with
`with_auto_reversible_min_samples` or `with_auto_dwt97_min_samples`. Same-shape
batches go to Metal from 32 jobs and `224 * 224 * 32` samples. `Auto` also
skips the staged 9/7 batch path when either tile side is over 1024 samples.
Explicit Metal requests and lowered thresholds are used as given. Meeting a
threshold does not guarantee a speedup.

High-level route-report example:

```bash
cargo run -p j2k-transcode-metal --example jpeg_to_htj2k_route_report
```

The example prints the requested backend, selected transform backend, final
codestream output backend, structured Auto fallback reason, transfer bytes, and
where each transcode stage ran.

On macOS, `resident_codestream_buffer_from_metal_encoded_j2k` converts
buffer-backed `j2k-metal` encode output into the shared
`ResidentCodestreamBuffer` descriptor, checking the allocation and capacity.

## Links

- API docs: <https://docs.rs/j2k-transcode-metal>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
