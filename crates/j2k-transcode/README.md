# j2k-transcode

JPEG-to-HTJ2K coefficient-domain transcode crate for J2K.

The crate contains the CPU transcode pipeline and the hooks the CUDA and Metal
adapters use to accelerate stages; the HTJ2K codestream is always assembled
here. Unsupported JPEG types and modes return an error.

`ResidentJpegDctGrid`, `ResidentDwtSubband`, and `ResidentCodestreamBuffer`
describe GPU buffers for adapters that keep transcode stages in device memory;
their constructors check the metadata.

## Links

- API docs: <https://docs.rs/j2k-transcode>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
