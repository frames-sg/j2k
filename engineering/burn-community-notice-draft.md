# Burn community notice draft

Do not send this notice yet. It is prepared for the maintainer to review and
send personally after every checkbox below is complete. This repository does
not authorize an agent or automation to post, message, comment, or otherwise
contact Burn or Tracel.

## Send gate

- [ ] `j2k-ml 0.7.6` is published on crates.io.
- [ ] Fresh external consumers pinned to `j2k-ml = "=0.7.6"` compile with
      `cpu`, `cuda`, `cpu,cuda`, `metal`, and `cpu,metal` on their applicable
      hosts without third-party path patches.
- [ ] The exact release SHA passes hosted CI, the full CUDA release gate, and
      the full Metal release gate.
- [ ] The linked batch results use content-distinct inputs at batches
      1/8/32/64 and clearly identify codec decode, readback, and Burn upload.
- [ ] Every link below resolves to the published version and final evidence.

## Draft message

**Showcase: batch JPEG 2000 / HTJ2K decoding for Burn**

I maintain [`j2k-ml`](https://crates.io/crates/j2k-ml), an independent adapter
from the J2K codec project to ordinary Burn tensors. This is a community
integration, not a proposal to add the code to Burn or have the Burn team own
it.

```bash
cargo add j2k-ml@0.7.6 --features cpu
# Or select `cuda` / `metal` for staged accelerator decode and upload.
```

The ownership boundary is:

- the Burn application owns its `Dataset`, sample selection, labels,
  `DataLoader`, training `Batcher`, prefetching, transforms, and training loop;
- J2K parses each encoded image, groups compatible images, and batch-decodes
  JPEG 2000 or HTJ2K;
- `j2k-ml` returns the decoded groups as Burn tensors and preserves
  `source_indices` so the application can realign labels.

The runnable
[`training_batcher`](https://github.com/frames-sg/j2k/blob/v0.7.6/crates/j2k-ml/examples/training_batcher.rs)
example demonstrates that complete flow with a persistent CPU decoder. The
[`CUDA upload`](https://github.com/frames-sg/j2k/blob/v0.7.6/crates/j2k-ml/examples/cuda_upload.rs)
and
[`Metal upload`](https://github.com/frames-sg/j2k/blob/v0.7.6/crates/j2k-ml/examples/metal_upload.rs)
examples execute codec decoding on the named accelerator, read the completed
decoded pixels back to host staging, and use Burn's public API for the ordinary
tensor upload. They are intentionally described as staged adapters, not
direct-destination, asynchronous cross-runtime, or zero-copy integrations.

Install and support details are in the
[`j2k-ml` guide](https://github.com/frames-sg/j2k/blob/v0.7.6/docs/j2k-ml.md).
Corrected content-distinct batch measurements and their transfer accounting
are here: **[replace with the final exact-version benchmark link]**.

If this is useful to Burn users, I would appreciate a community share. I would
also welcome API feedback on the application `Batcher` example and the returned
group/error shape; repository ownership is not being requested.
