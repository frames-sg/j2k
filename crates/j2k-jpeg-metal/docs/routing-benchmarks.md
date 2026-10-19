# JPEG Metal Routing and Benchmarks

The current batch execution design, ownership rules, measured comparisons, and
reproduction commands are in [Decode architecture](decode-architecture.md).

Metal JPEG decode covers every JPEG the CPU decoder supports, through two
routes: the fast-packet kernels (GPU entropy decode for baseline 4:2:0, 4:2:2,
and 4:4:4) and hybrid decode (CPU entropy decode, then Metal IDCT or lossless
reconstruction, upsampling, color, and packing) for everything else.

## Routing rules

Explicit `BackendRequest::Metal` requests:

- Fast-packet kernels: baseline/extended 8-bit 4:2:0 or 4:2:2 more than 4 px
  wide, and 4:4:4, to `Gray8`, `Rgb8`, or `Rgba8`, when the `j2k-jpeg` fast
  packet builders produce a packet.
- Hybrid decode: every other request CPU decode accepts, including 16-bit
  output for 12-bit and lossless JPEG.
- Rejections: requests CPU decode rejects (with a Metal-worded reason for
  output-format mismatches), unsupported output formats, unsupported backend,
  or an unavailable Metal runtime. These return an error before any kernel is
  launched.

`BackendRequest::Auto` is deliberately narrower:

- Single-image full, region, scaled, and region-scaled requests stay on CPU even
  when a fast packet exists.
- Full RGB batches can use Metal for at least 16 compatible non-restart 4:2:0
  or 4:2:2 tiles at least 256×256 pixels, including distinct inputs. Dimensions,
  tables, and checkpoint counts must match across the batch.
- Smaller, mixed-table, restart-coded, and scaled tile batches stay CPU.
- JPEG classes outside the fast-packet kernels stay CPU: `jpeg_metal_hybrid_decode`
  measures hybrid decode at about CPU speed for single images.
- Sparse viewport workloads stay CPU for scheduled surface output; contiguous
  restart-coded fast-packet viewports may decode through Metal on macOS.
  Reusable resident viewport outputs may use direct contiguous decode or
  resident composition. Explicit Metal viewports of JPEG classes outside the
  fast-packet kernels entropy-decode once and compose every tile through
  hybrid decode.

## Benchmark Map

Run:

```sh
cargo bench -p j2k-jpeg-metal --bench compare
```

The harness also accepts extra JPEG corpus roots through `J2K_BENCH_INPUTS`.
Inputs are grouped by dimensions, restart/checkpoint shape, and fast packet
family so batch results make coalescing visible.

Use these groups to decide where Metal makes sense:

- `decode_rgb`: cold single full-frame CPU vs explicit Metal. This is the loss
  check for CPU-preferred single decode.
- `wsi_tile_batch_rgb` and `wsi_tile_batch_scaled_rgb_q4`: repeated tile batches
  comparing CPU, explicit Metal, and Auto.
- `jpeg_distinct_batch_routing`: completed submissions for distinct 4:2:0 and
  4:2:2 inputs, including plan construction, at 64×64 and 256×256 dimensions.
- `jpeg_metal_mixed_tables`: interleaved table groups targeting reusable RGB
  buffers and textures, to measure the cost of submitting multiple groups.
- `wsi_tile_batch_region_scaled_coalesced_rgb_q4`: coalesced region+scaled batch
  candidate where Metal can amortize setup.
- `wsi_tile_batch_region_scaled_distinct_rgb_q4`: low-coalescing control case.
  Only route these to Metal if this benchmark shows a win.
- `wsi_tile_batch_rgba_textures`: resident texture batches that avoid host
  downloads.
- `jpeg_metal_hybrid_decode` (`--bench hybrid`): CPU decode against explicit
  Metal hybrid decode for progressive, grayscale, 4:4:0, CMYK, and any
  `J2K_BENCH_INPUTS` file such as 12-bit or lossless JPEG. Promote a hybrid
  class to `Auto` only if this shows a win.
- `viewer_region_scaled_composite_rgb` and
  `viewer_region_scaled_composite_rgb_device`: CPU/hybrid viewport comparisons
  with host and device output.
- `viewer_region_scaled_composite_rgb_warm` and
  `viewer_region_scaled_composite_rgb_device_warm`: repeated viewer work with
  setup separated from warm routing and execution.
- `viewer_contiguous_region_scaled_rgb` and
  `viewer_contiguous_region_scaled_rgb_device`: contiguous viewport controls
  with host and device output.
- `viewer_best_region_scaled_rgb_device` and
  `viewer_best_region_scaled_composite_rgb_device`: best-supported resident
  viewport and composite paths.
- `jpeg_metal_fast_packet_planning`: route-discovery overhead for accepted and
  rejected fast packet families.

The benchmark surface intentionally includes both likely wins and likely losses.
Do not broaden Auto routing unless the relevant group shows repeatable wins for
the workload class being changed.
