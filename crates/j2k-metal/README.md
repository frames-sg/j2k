# j2k-metal

Metal adapter for J2K J2K / HTJ2K paths on macOS.

The crate provides resident Metal decode and encode-stage integration for
supported workloads. It uses `j2k-metal-support` for runtime setup while
keeping codec-specific kernels local.
