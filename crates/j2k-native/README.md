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

With `std`, the second eligible region decode through one `Image` builds a
packet boundary directory that later region decodes reuse, including the
component-plane and prepared-region APIs. It needs no setting, holds at most
16 MiB per image, and is shared between contexts.

The directory covers images with one RPCL tile in one tile-part, inline packet
headers, no POC, and no PLT. Other layouts, and `no_std` builds, parse packets
on every request. Under allocation pressure an unshared directory can be
evicted.

To benchmark 1024x1024 regions of a local image with twelve workers:

```
J2K_PACKET_DIRECTORY_SOURCE=/path/to/large.jp2 \
J2K_PACKET_DIRECTORY_REGION_SIDE=1024 J2K_PACKET_DIRECTORY_THREADS=12 \
cargo bench -p j2k-native --bench direct_cpu -- packet_directory
```

Keep `J2K_PROFILE_STAGES` unset for timing. The bench checks planes bit for
bit and reports latency with fresh and reused contexts and with eight
concurrent requests. Its control builds a fresh `Image` per request outside
the timer.

## Links

- API docs: <https://docs.rs/j2k-native>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
