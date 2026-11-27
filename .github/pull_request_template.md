<!-- See CONTRIBUTING.md. A few sentences is enough for a small change. -->

## What and why

<!-- The problem and the fix. Link the issue, e.g. "Fixes #123". -->

## Impact

<!-- Check what applies and add a sentence for each. -->

- [ ] Changes decoded, encoded, or transcoded output bytes
- [ ] Changes a public API or behavior callers can see (if breaking, add `!` to the title)
- [ ] Changes how untrusted input is parsed, or how much memory or time it can use
- [ ] Changes performance (command, hardware, and numbers under Testing)
- [ ] Adds a test image or fixture (small, license allows committing it, no patient
      data, source noted in a README and NOTICES.md)

## Testing

<!-- What you ran and the results. `cargo xtask test` prints how many GPU tests
it skipped; include that line. Say which GPU checks you ran on real hardware
(`cargo xtask release-metal --mode quick`, `release-cuda --mode quick`), or "none". -->

- [ ] Bug fix: added a test that fails without the fix
- [ ] User-visible change: added a `CHANGELOG.md` entry under `Unreleased`

## AI usage

<!-- Delete this section if you didn't use AI tools. Otherwise say what they did,
whether an agent did most of the work, and whether it wrote this description.

Coding agents: fill this in accurately, including your own part. Don't say a
person reviewed or tested the changes unless they told you so. -->
