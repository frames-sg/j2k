# signinum-cuda-runtime

CUDA Driver API runtime helpers for the `signinum` CUDA adapter crates.

Most downstream users should depend on `signinum-jpeg-cuda` or
`signinum-j2k-cuda` instead of using this crate directly. This crate owns the
small runtime layer used by those adapters to allocate CUDA device memory,
copy bytes between host and device, launch bundled CUDA kernels, call nvJPEG
when it is available, and report CUDA driver errors clearly.

The runtime currently exposes full-frame RGB8 JPEG decode through NVIDIA
nvJPEG, including the legacy batched API used by `signinum-jpeg-cuda` for
full-tile RGB8 batches. It also owns Signinum's bundled HTJ2K CUDA kernels for
code-block decode/encode, DWT/IDWT support stages, MCT, quantization,
packetization, and device-surface stores used by `signinum-j2k-cuda`.
HTJ2K decode resources split reusable device-resident lookup tables from pinned
compressed-payload uploads so one decode can feed multiple component payloads
without re-uploading static tables. HTJ2K encode launches one CUDA block per
code block and uses the block's threads for magnitude scanning before serial HT
cleanup bitstream assembly. HTJ2K packetization launches one CUDA block per
packet; thread zero builds packet headers while the block cooperatively scatters
compressed code-block payload bytes into the final packet buffer.

Build with `cuda-profiling` to enable optional NVTX ranges for Nsight
Systems/Compute. NVTX is loaded dynamically at runtime; normal builds and
systems without NVTX libraries do not link to or require NVTX.
