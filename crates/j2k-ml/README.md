# j2k-ml

Thin Burn 0.21 adapter for the `j2k` owned batch codec.

`j2k-ml` is an independent integration maintained by the `j2k` project. It is
not an official Tracel or Burn crate. It is included in the `j2k` 0.7.5 release
and follows the workspace's reviewed semver policy.

The codec crates own parsing, preparation, grouping, decoding, and device
execution. This crate only allocates ordinary rank-4 Burn integer tensors and
connects their storage to persistent CPU, CUDA, or Metal codec sessions. It
does not own normalization, augmentation, dataset policy, or another decoding
pipeline.

See the single [Burn integration guide](../../docs/j2k-ml.md) for the API,
support matrix, safety boundary, validation commands, and benchmark status.
The published API is on [docs.rs](https://docs.rs/j2k-ml), source is maintained
in the [J2K repository](https://github.com/frames-sg/j2k), and the codec support
boundary is recorded in [docs/public-support.md](../../docs/public-support.md).
