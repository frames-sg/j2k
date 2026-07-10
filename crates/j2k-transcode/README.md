# j2k-transcode

JPEG-to-HTJ2K coefficient-domain transcode crate for J2K.

The crate owns the CPU transcode pipeline and shared accelerator hooks. CUDA
and Metal adapters can accelerate supported stages, but the public transcode
entry points assemble HTJ2K codestreams. Unsupported source classes and modes
return explicit errors.

Resident handoff descriptors such as `ResidentJpegDctGrid`,
`ResidentDwtSubband`, and `ResidentCodestreamBuffer` provide validated metadata
contracts for backend adapters that keep transcode stages in device memory.

## Links

- API docs: <https://docs.rs/j2k-transcode>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
