# j2k-transcode-cuda

NVIDIA CUDA GPU acceleration adapter for Rust JPEG-to-J2K/HTJ2K transcode
stages.

This crate accelerates supported transform and code-block preparation stages.
It does not replace the transcode API in `j2k-transcode`.
