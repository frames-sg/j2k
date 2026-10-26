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

## Links

- API docs: <https://docs.rs/j2k-native>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
