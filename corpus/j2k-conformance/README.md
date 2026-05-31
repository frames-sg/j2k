# JPEG 2000 / HTJ2K ISO Conformance Scope

`manifest.tsv` defines the release-signoff scope for external ISO/IEC 15444-4
and Part 15 conformance vectors.

Set `SIGNINUM_J2K_ISO_CONFORMANCE_DIR` to a local checkout or unpacked copy of
the conformance vectors before running the harness:

```sh
SIGNINUM_J2K_ISO_CONFORMANCE_DIR=/path/to/vectors cargo test -p signinum-j2k --test iso_conformance
```

With `SIGNINUM_J2K_ISO_CONFORMANCE_DIR` unset, the harness only validates the
manifest shape so developers without the licensed corpus can still run the test
target. With the variable set, every `blocking` row must resolve to an available
vector and must decode successfully; missing or failing blocking vectors fail
release signoff. A release candidate must populate the listed blocking paths
from the local ISO vector checkout or adjust the manifest to the equivalent
official vector filenames before signoff.

`known-limitation` rows document deferred features and are reported rather than
decoded as release blockers. `investigate` rows are not allowed while the
env-gated harness is enabled.
