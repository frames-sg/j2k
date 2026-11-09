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

## 9/7 precision

The irreversible 9/7 path computes in IEEE binary32 by default
(`TranscodePrecision::F32`). The IDCT, lifting and quantizer follow one
operation sequence, defined in `j2k_codec_math::transcode97`, that the CPU,
Metal and CUDA backends all evaluate. The same JPEG and options therefore give
byte-identical codestreams on every backend. The quantizer multiplies by the
binary32 reciprocal of each step size.

`TranscodePrecision::F64` keeps the binary64 path that was the default before
this release. It reproduces that release's CPU output byte for byte. It runs
only on the CPU, and its output differs from GPU output. An explicit Metal or
CUDA accelerator rejects it with `TranscodeStageError::Unsupported`; `Auto`
runs it on the CPU.

`ResidentJpegDctGrid`, `ResidentDwtSubband`, and `ResidentCodestreamBuffer`
describe GPU buffers for adapters that keep transcode stages in device memory;
their constructors check the metadata.

## Links

- Project website: <https://frames-sg.com/jpeg-to-htj2k-transcoding/>
- API docs: <https://docs.rs/j2k-transcode>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
