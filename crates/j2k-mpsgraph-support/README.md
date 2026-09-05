# j2k-mpsgraph-support

Shared asynchronous MPSGraph ownership for the J2K and JPEG XR adapters.
Available on Apple Silicon macOS. This package owns the submitted graph,
placeholder, target/feed dictionaries, execution descriptor, callback and
completion signal. A concrete caller-owned input guard stays alive until
completion and is released after the graph resources.

The unsafe submission boundary requires a validated single-placeholder graph,
matching tensor data/device, ordered input writes, and an input owner that keeps
all unretained storage alive. `wait` reports an owned Foundation error. Drop waits
for completion without interpreting codec metadata or allocating output vectors.
The guard is neither Send nor Sync; adapters retain their codec submission and
assemble their own results and diagnostics.

This package must be published before the JPEG XR consumer can use a registry-only
build. Local Cargo source patches validate development changes only.
