# j2k-jpeg-metal

Metal adapter for J2K JPEG decode and baseline encode paths on macOS.

Supported paths decode into Metal memory or run selected stages on Metal.
Explicit Metal requests return an error for unsupported shapes.

Since version 0.9, the session, buffer, and texture APIs take retained or
borrowed `objc2-metal` protocol objects. The older `metal-rs` types are not
accepted.

## JPEG Decode Scope

Metal JPEG decode covers a subset of JPEG: baseline images that fit the fast
packet path, batches of whole-slide-image tiles, and viewport or texture
workloads where the output stays on the GPU.

Explicit `BackendRequest::Metal` decode accepts only JPEG inputs that can build a
fast 4:2:0, 4:2:2, or 4:4:4 baseline packet and only `Gray8`, `Rgb8`, or
`Rgba8` output. Unsupported sampling families, unsupported color spaces, and
unsupported output formats return `UnsupportedMetalRequest`; there is no
fallback. 4:2:0 and 4:2:2 images at most four pixels wide are also
excluded from the fast shapes: explicit Metal requests reject them, while
`Auto` uses the CPU.

Supported scaled and region-scaled Metal decodes use the CPU decoder's
component-specific IDCT sizes and chroma interpolation rules. Cropped regions
retain the neighboring chroma samples required for smoothing.

`BackendRequest::Auto` keeps single-image decode on the CPU. Full RGB batches
can use Metal for at least 16 compatible non-restart 4:2:0 or 4:2:2 tiles of at
least 256×256 pixels, including distinct inputs. The tiles must share dimensions,
tables, and checkpoint count. Smaller, mixed-table, restart-coded, and scaled
batches stay on the CPU. The threshold was measured on an M4 Pro; widen it
only with results from the benchmark groups in
[`docs/routing-benchmarks.md`](docs/routing-benchmarks.md).

## Links

- API docs: <https://docs.rs/j2k-jpeg-metal>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
