# signinum-j2k

JPEG 2000 / HTJ2K inspect, CPU decode, and lossless encode for whole-slide
imaging workloads.

Install:

```sh
cargo add signinum-j2k
```

The stable `0.5.x` surface covers borrowed inspect/parse, compressed
passthrough candidates, full-frame decode, ROI decode, reduced-resolution
decode, combined ROI+reduced-resolution decode, row-bounded decode, tile-batch
decode, and lossless JPEG 2000 / HTJ2K encode through the shared
`signinum-core` traits.

GPU adapter crates are versioned separately and are not part of the CPU-first
stable surface.
