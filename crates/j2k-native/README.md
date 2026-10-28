# j2k-native

Native JPEG 2000 Part 1 / HTJ2K Part 15 engine used by J2K APIs and adapters.

This crate owns codestream parsing, native encode/decode helpers, packetization
support, JP2/JPH still-image wrapper handling, HTJ2K cleanup/refinement table
helpers, and header inspection helpers used by higher-level crates.

It supports the same inputs as `j2k`: raw J2K/J2C codestreams, JP2 files, raw
HTJ2K codestreams, and JPH files. JPX (JPEG 2000 Part 2) extensions are not
supported.

The default `profile` feature compiles the stage profiling enabled by
`J2K_PROFILE_STAGES`. Without it, profiling compiles to no-ops; builds that
disable default features and need the rows must enable `profile` explicitly.

Most application code should use `j2k` instead.

## Repeated region decoding

With `std`, existing region decode calls automatically retain a bounded packet
boundary directory after the second eligible request through the same `Image`.
This includes the component-plane and prepared-region APIs. Independent contexts
share the immutable directory; no new decoder API, setting, or feature is needed.
The first request avoids a full preparation scan. Unsupported layouts and failed
optional scans keep the ordinary parser. Preparation currently covers one RPCL
tile with one tile-part, inline headers, no POC, and no PLT, with at most 16 MiB
retained per image. Allocation pressure can evict an unshared directory before
retrying. `no_std` builds keep the existing parser.

The native benchmark uses the existing integer-output component method. Its
control constructs a fresh `Image` per request outside the decode timer to
prevent reuse. Image setup is measured separately. To run 1024² regions on a local workload with twelve shared
workers:

```
J2K_PACKET_DIRECTORY_SOURCE=/path/to/large.jp2 \
J2K_PACKET_DIRECTORY_REGION_SIDE=1024 J2K_PACKET_DIRECTORY_THREADS=12 \
cargo bench -p j2k-native --bench direct_cpu -- packet_directory
```

Keep `J2K_PROFILE_STAGES` unset for timing. Results include bit-for-bit plane
checks, interleaved A/A and A/B latency, fresh/reused contexts, retained capacity,
and eight concurrent requests on the same worker pool. Benchmark names include
the input name, dimensions, region size and worker count.

## Links

- API docs: <https://docs.rs/j2k-native>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
