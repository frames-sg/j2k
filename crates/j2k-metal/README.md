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

Metal routing is deliberately selective. `Auto` may decline small tiles,
irregular packet shapes, or stages where host/device transfer and dispatch
overhead dominate. Stage-by-stage host-output `Auto` currently limits Metal to
deinterleave, forward RCT/ICT, forward 5/3 and 9/7 DWT, and subband
quantization. Classic Tier-1, HT code-block encode, packetization, and
codestream assembly stay CPU for that route unless a documented resident path
supports the shape with parity and benchmark evidence.
