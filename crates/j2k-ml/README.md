# j2k-ml

Experimental thin Burn 0.21 adapter for the `j2k` owned batch codec.

`j2k-ml` is an independent integration maintained by the `j2k` project. It is
not an official Tracel or Burn crate. It remains unpublished during the 0.7
release cycle.

The codec crates own parsing, preparation, grouping, decoding, and device
execution. This crate only allocates ordinary rank-4 Burn integer tensors and
connects their storage to persistent CPU, CUDA, or Metal codec sessions. It
does not own normalization, augmentation, dataset policy, or another decoding
pipeline.

See the single [Burn integration guide](../../docs/j2k-ml.md) for the API,
support matrix, safety boundary, validation commands, and benchmark status.
