# Metal Encode-Stage Coverage

## Goal

Fill the missing Metal encode-stage acceleration gaps for JPEG 2000 and HTJ2K
encoding on macOS while keeping CPU fallback behavior explicit and testable.

## Current State

`crates/j2k-metal/src/encode/stage_accelerator.rs` currently reports Metal
dispatches for:

- deinterleave for public 1-4 component, 1-16 bit host encode sample layouts
- forward RCT
- forward ICT
- forward 5/3 DWT
- forward 9/7 DWT
- subband quantization
- classic Tier-1 code-block encode
- HT code-block encode
- packetization

The automatic host-output path is conservative and does not try every available
Metal stage. `MetalEncodeStageAccelerator::for_auto_host_output()` currently
allows benchmark-gated coefficient-prep stages only:

- deinterleave, forward RCT, forward ICT, forward 5/3 DWT, forward 9/7 DWT,
  and subband quantization may dispatch when the individual stage input has at
  least `512 * 512` samples or coefficients.
- Classic Tier-1, HT code-block, and packetization stay disabled for
  stage-by-stage Auto host-output routing.
- The resident HTJ2K RGB8 lossless shortcut remains separately gated to
  1,024 x 1,024 and larger tiles, with CPU packetization.
- Shapes below these gates fall back to CPU explicitly; no silent strict-device
  fallback is used.

## Benchmark Evidence

Command:

```bash
cargo test -p j2k-metal --test encode_auto_routing_benchmark -- --ignored --nocapture
```

Host: macOS 26.5 (25F71), Apple M4 Pro CPU, Apple M4 Pro 16-core GPU,
Metal 4. Timings are median wall-clock milliseconds from the ignored benchmark
harness and are intended as routing evidence, not a cross-machine performance
claim.

Stage microbenchmarks:

| Stage | 128 CPU / Metal | 512 CPU / Metal | 1024 CPU / Metal | Auto gate |
| --- | ---: | ---: | ---: | --- |
| deinterleave | 1.160 / 0.232 | 8.170 / 0.512 | 32.398 / 1.776 | >= 512 x 512 |
| forward RCT | 0.192 / 0.217 | 2.973 / 0.313 | 11.986 / 0.508 | >= 512 x 512 |
| forward ICT | 0.145 / 0.184 | 2.407 / 0.244 | 9.806 / 0.607 | >= 512 x 512 |
| forward 5/3 DWT | 0.946 / 0.199 | 15.392 / 0.434 | 62.272 / 0.873 | >= 512 x 512 |
| forward 9/7 DWT | 1.353 / 0.263 | 22.041 / 0.629 | 87.317 / 1.403 | >= 512 x 512 |
| subband quantization | 0.124 / 0.181 | 2.082 / 0.546 | 8.217 / 1.092 | >= 512 x 512 coefficients |

Selected Auto-route results:

| Route | 128 dispatch | 512 CPU / Auto | 512 dispatch | 1024 CPU / Auto | 1024 dispatch |
| --- | --- | ---: | --- | ---: | --- |
| lossless classic Gray8 | none | 42.776 / 21.572 | deinterleave=1, dwt53=1 | 169.933 / 61.478 | deinterleave=1, dwt53=1, quantize=4 |
| lossless classic RGB8 | none | 115.098 / 65.294 | deinterleave=1, rct=1, dwt53=3 | 429.063 / 187.509 | deinterleave=1, rct=1, dwt53=3, quantize=12 |
| lossless HTJ2K RGB8 | none below resident gate | 73.426 / 71.201 | none | 284.845 / 10.012 | rct=1, dwt53=3, ht=1 |
| lossy HTJ2K RGB8 | none | 186.633 / 39.586 | deinterleave=2, ict=2, dwt97=6 | 752.962 / 99.089 | deinterleave=2, ict=2, dwt97=6, quantize=24 |

## Remaining Gap

Subband quantization has landed for the existing contiguous encode-stage job
shape, with dispatch accounting, CPU parity tests, and explicit Metal errors for
unsupported quantization parameters. Auto routing now uses benchmark-gated
coefficient-prep stages and the existing resident HTJ2K RGB8 shortcut, but this
is not full end-to-end Metal encode coverage for every public encode route.
Tier-1, HT code-block, packetization, and codestream assembly remain CPU for
stage-by-stage Auto host-output routes unless a resident encode path explicitly
supports the shape.

That boundary is intentional. Do not widen Auto routing simply to increase the
number of Metal stages. New automatic dispatch should land only when the stage
is parity-covered, benchmark-backed, and does not lose the benefit through
extra host/device transfer or dispatch overhead.

## Acceptance Criteria

- Explicit Metal encode requests either dispatch supported stages or return a
  structured unsupported request error.
- Auto mode remains conservative unless benchmarks justify widening dispatch.
- CPU parity tests cover each new Metal stage.
- GPU validation records stage-level dispatch counts and performance artifacts.
- Public docs describe the supported Metal encode-stage surface without
  implying full end-to-end Metal coverage for unsupported shapes.
