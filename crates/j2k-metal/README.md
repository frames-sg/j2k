# j2k-metal

Metal adapter for JPEG 2000 / HTJ2K decode and encode-stage paths on macOS.

The crate provides resident Metal decode and encode-stage integration for
supported workloads. It uses `j2k-metal-support` for runtime setup while
keeping codec-specific kernels local.

Encode support is stage-oriented unless a documented resident path accepts the
shape. `Auto` host-output encode may dispatch benchmark-gated coefficient-prep
stages for 512 x 512 and larger stage inputs, and the resident HTJ2K RGB8
lossless shortcut is gated to 1,024 x 1,024 and larger tiles. Explicit Metal
requests are strict: supported shapes dispatch, and unsupported direct Metal
requests return `UnsupportedMetalRequest` instead of silently changing backend.
