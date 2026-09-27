# j2k-jpeg-metal

Metal adapter for J2K JPEG decode and baseline encode paths on macOS.

Supported paths return resident Metal outputs or use Metal kernels for selected
adapter stages. Explicit Metal requests are strict and fail for unsupported
shapes.

Version 0.9 uses retained and borrowed `objc2-metal` protocol objects in its
expert session, buffer, and texture APIs. It does not provide aliases or
adapters for the former `metal-rs` types.

## JPEG Decode Scope

JPEG Metal decode is selective acceleration, not full JPEG feature coverage. It
is intended for fast baseline/checkpointed packet paths, batched WSI-style tile
decode, and resident-output viewport or texture workflows where the output can
stay on Metal.

Explicit `BackendRequest::Metal` decode accepts only JPEG inputs that can build a
fast 4:2:0, 4:2:2, or 4:4:4 baseline packet and only `Gray8`, `Rgb8`, or
`Rgba8` output. Unsupported sampling families, unsupported color spaces, and
unsupported output formats return `UnsupportedMetalRequest` instead of silently
falling back.

`BackendRequest::Auto` keeps single-image decode on the CPU. Full RGB batches
can use Metal for at least 16 compatible non-restart 4:2:0 or 4:2:2 tiles of at
least 256×256 pixels, including distinct inputs. The tiles must share dimensions,
tables, and checkpoint count. Smaller, mixed-table, restart-coded, and scaled
batches remain on the CPU. This threshold follows completed
submission measurements on an M4 Pro; further widening should be backed by
the benchmark groups documented in
[`docs/routing-benchmarks.md`](docs/routing-benchmarks.md).

## Links

- API docs: <https://docs.rs/j2k-jpeg-metal>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
