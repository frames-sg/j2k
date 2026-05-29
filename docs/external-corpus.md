# External Corpus

Committed fixtures are intentionally small. Real-world adoption requires
repeatable validation against external WSI-derived corpora without committing
large or proprietary image data.

Every external corpus manifest should record:

- source: where the bytes came from and how they were extracted
- license: redistribution and internal-use constraints
- hash: SHA-256 for each input or archive
- manifest: machine-readable list of paths, codec, dimensions, sampling, and
  expected support status
- unsupported: unsupported modes and the expected structured error
- reproducer: exact `cargo test`, `cargo bench`, or CLI command

Suggested local layout:

```text
SIGNINUM_WSI_ROOT/
  corpus.json
  jpeg/
  j2k/
  htj2k/
```

The manifest should keep clinical metadata out of the path names and should not
include patient identifiers. When a corpus exposes a bug, minimize the input if
possible and link the minimized reproducer to a private security report or a
public regression test after triage.

