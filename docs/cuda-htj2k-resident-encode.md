# CUDA HTJ2K Resident Encode Status

This note records the measured state of the experimental CUDA HTJ2K Tier-1
encode path after the quality/RD sweep, CUDA resident HT integration, batched
payload readback, compatible 9/7 component grouping, and async HT launch cleanup.

## Acceptance Corpus

- Runner: self-hosted Linux CUDA runner, RTX 4070.
- Corpus: 109 committed pancreas JPEG tiles.
- Corpus hash: `c1060319d3236928`.
- Image area: 7.143424 MP.
- Selected RD index: `2`.
- Signinum scale: `1.85`.
- Byte matching: enabled against the NVIDIA reused-session serial baseline with
  2% tolerance.

## Latest Measurement

Run: `26706505889`, commit `a2b63a3`.

| Row | Bytes | Wall ms | GPU ms | Wall MP/s | GPU MP/s | PSNR | Dispatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Signinum CUDA transform + CPU HT | 6,258,565 | 84.859 | 26.388 | 84.18 | 270.71 | 47.737 dB | `2/0/0` |
| Signinum CUDA transform + CUDA HT block + CPU packet | 6,258,565 | 85.747 | 42.449 | 83.31 | 168.28 | 47.737 dB | `2/8/8` |
| NVIDIA reused-session serial | 6,233,094 | 135.733 | 130.908 | 52.63 | 54.57 | n/a | n/a |

Interpretation:

- Signinum production and experimental paths are both faster than the NVIDIA
  reused-session serial baseline at matched bytes.
- The experimental CUDA HT path is not a production win yet: it trails the
  production CPU HT path by 0.888 ms wall-clock on this corpus.
- The CUDA HT path is still useful evidence: it removed coefficient readback and
  now runs the HT block encoder on CUDA with strict nonzero dispatch gates, but
  the present shape does not beat the CPU Tier-1 implementation.

Scope note:

This note covers the JPEG DCT-grid to HTJ2K 9/7 resident-HT experiment on the
pancreas transcode corpus. It is not the same workload as the public facade's
lossless host-pixel RGB/RGBA8 HTJ2K Auto gate. The facade gate uses separate
lossless host-output measurements and may select CUDA DWT plus CUDA HT block
coding for large 8-bit RGB/RGBA tiles while keeping MCT/RCT and packetization on
CPU. Do not use the 9/7 transcode row in this note to disable that lossless
facade route without re-running the matching facade benchmark.

## What Improved

Before these changes, the CUDA HT experimental row was about 202 ms wall-clock
with `3/12/12` dispatches. After batched payload readback and compatible 9/7
component grouping, it is about 86 ms with `2/8/8` dispatches.

The component grouping also improved the production CUDA-transform row, because
both production and experimental rows share the same 9/7 transform batching.

## No-Win Blocker

The current experimental CUDA HT design still returns encoded codeblock payloads
to the CPU and uses CPU packetization/codestream assembly. It also launches HT
encode separately for each subband group:

- two transform groups on the pancreas corpus: luma and grouped chroma;
- four HT subband encodes per group;
- eight HT encode dispatches total before CPU packetization.

At this tile size, those launches and the CPU packetization boundary leave the
CUDA HT row nearly tied with, but still slower than, the existing CPU HT row.
Removing the redundant host launch synchronize did not change the measured
outcome enough to win.

## Required Next Architecture

To make CUDA HT a real production win, the next design should copy the Metal
resident structure more closely:

- encode all resident subbands for a tile/component batch through one flattened
  CUDA descriptor stream instead of one launch per subband;
- keep the encoded payload arena and codeblock metadata resident after Tier-1;
- move packet header generation and codestream assembly to CUDA or to a single
  final compact DtoH payload;
- reuse CUDA arenas across tiles/batches so the benchmark path does not allocate
  per subband group.

Until that resident packetization/codestream stage exists, the supported
production claim for this JPEG DCT-grid 9/7 transcode corpus should stay
conservative: Signinum CUDA transform + CPU HT is the fastest Signinum row on
the measured corpus, and both Signinum rows beat the NVIDIA reused-session
serial baseline at matched bytes.
