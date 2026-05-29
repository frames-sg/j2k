# Fuzz Corpus

This directory documents the seed corpus policy for fuzzing. The repository may
carry tiny synthetic seeds when their source, license, and hash are clear.
Generated seeds should be reproducible from checked-in scripts or committed
fixture bytes.

Do not commit large WSI tiles, proprietary clinical data, or minimized crash
files before triage. CI uploads fuzz artifacts so maintainers can retrieve the
raw artifact, run `cargo fuzz tmin`, and turn the minimized crash into a
private reproducer or a public regression fixture as appropriate.

Seed corpus entries should record:

- source
- license
- hash
- target fuzz harness
- reproducer command

