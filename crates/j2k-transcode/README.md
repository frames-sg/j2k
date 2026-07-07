# j2k-transcode

JPEG to J2K/HTJ2K transcode crate for J2K.

The crate owns CPU transcode algorithms and shared accelerator hooks. CUDA and
Metal acceleration live in adapter crates. Unsupported source classes and
unsupported transcode modes return explicit errors.

Resident handoff descriptors such as `ResidentJpegDctGrid`,
`ResidentDwtSubband`, and `ResidentCodestreamBuffer` provide validated metadata
contracts for backend adapters that keep transcode stages in device memory.

## Links

- API docs: <https://docs.rs/j2k-transcode>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
