# j2k-transcode

JPEG-to-HTJ2K coefficient-domain transcode crate for J2K.

The crate contains the CPU transcode pipeline and the hooks the CUDA and Metal
adapters use to accelerate stages; the HTJ2K codestream is always assembled
here. Unsupported JPEG types and modes return an error.

The default CPU `transcode_batch` processes tiles in parallel waves bounded by
the current Rayon pool and the existing host-memory cap. Outputs and per-tile
errors retain input order. Explicit accelerator calls keep their component
batching. Parallel CPU stage reports sum per-tile intervals, so they can exceed
elapsed batch latency.

`ResidentJpegDctGrid`, `ResidentDwtSubband`, and `ResidentCodestreamBuffer`
describe GPU buffers for adapters that keep transcode stages in device memory;
their constructors check the metadata.

## Links

- API docs: <https://docs.rs/j2k-transcode>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
