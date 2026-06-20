# j2k-metal-support

Shared Metal runtime setup helpers for J2K Metal adapters.

The crate centralizes system device lookup, checked command-queue creation,
shader-library compilation, named pipeline loading, and stable route labels.
Codec-specific kernels stay in the codec adapter crates.
