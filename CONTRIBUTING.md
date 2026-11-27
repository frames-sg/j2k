# Contributing to J2K

Thanks for your interest in J2K. Bug reports, fixes, test images that break
the decoder, performance work, and documentation improvements are all welcome.

J2K aims to be practical JPEG 2000 and HTJ2K infrastructure: safe parsing,
predictable decode and encode behavior, buffer and context reuse for
high-throughput callers, GPU acceleration where it measurably helps, and
benchmarks anyone can reproduce. Whole-slide imaging and other huge tiled
images are important stress cases, but the APIs are general-purpose.

## How to contribute

Look through the [open issues](https://github.com/frames-sg/j2k/issues).
Issues labelled `good first issue` are a good place to start.

If you want to work on something that doesn't have an issue yet, open one
first and describe the problem, especially for larger changes. It's much
easier to agree on an approach before the code is written.

The [contributor book](contributor-book/README.md) covers the architecture,
environment setup, testing, benchmarks, and releases.

## Pull requests

Give each pull request a descriptive title and a short description of the
problem, the fix, and how you tested it, with a link to the issue if there is
one. A few sentences are enough for a small change. Prefer small, focused pull
requests over large ones that bundle unrelated changes.

### Titles and commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for
pull request titles: `type(scope): description`, for example
`fix(native): stop an open-ended tile part at EOC`. The scope is optional.
Mark changes that break the latest published release with `!` before the
colon, and explain the migration in the description.

### Before you open a pull request

1. Run the checks:

   ```bash
   cargo xtask fmt
   cargo xtask clippy
   cargo xtask test
   cargo xtask doc
   ```

2. Add or update tests. Bug fixes need a regression test. Prefer tests that
   check behavior (decoded samples, errors, output bytes) over tests that pin
   implementation details.
3. Add a changelog entry if users will notice the change (see
   [Changelog](#changelog)).
4. If you changed a CUDA or Metal crate or anything they depend on, the pull
   request needs a GPU validation run before it can merge. Maintainers trigger
   it; [CI and GPU validation](contributor-book/ci.md) explains when it's
   required.
5. If you changed a public API, update the API snapshots as described in
   [Public API changes](contributor-book/public-api.md).

### Code guidelines

- Follow the style of the surrounding code.
- Unsupported input and backend failures return errors. Don't add silent
  fallbacks, and never let an explicitly requested GPU backend quietly run on
  the CPU.
- Keep dependencies minimal, and discuss new ones in the issue first.
- Every `unsafe` block needs a `// SAFETY:` comment explaining why it's sound.
- Don't remove passing regression tests as cleanup.
- Don't commit secrets, credentials, or paths from your machine. Tests and
  benchmarks read external data locations from environment variables.

### Change ownership

You must understand and be able to explain every change in your pull request,
whether you wrote it yourself, adapted it from another project, or used an AI
tool to help. Using AI tools is fine. You're still the author, so read and
test what you submit, and edit generated text down to what's accurate and
necessary.

### Review

Maintainers review pull requests as time allows. Please respond to feedback,
or explain your reasoning if you disagree, and don't force-push during an
active review without saying so. A pull request with no response from its
author for 14 days may be closed.

## Changelog

User-visible changes get an entry under `Unreleased` in
[CHANGELOG.md](CHANGELOG.md) in the same pull request. That includes API and
behavior changes, new features, bug fixes users could notice, output that
changes bytes, and notable performance changes. Refactors, test-only changes,
and doc fixes don't need entries. The
[changelog guide](contributor-book/documentation.md#changelog) has the
details.

## Documentation

User documentation lives in the [J2K Book](book/README.md) and contributor
documentation in the [contributor book](contributor-book/README.md). Before
writing docs, read [Writing documentation](contributor-book/documentation.md).
It explains where things go and the style we use.

## Getting help

If you're stuck, open an issue and ask. Report security problems privately as
described in [SECURITY.md](SECURITY.md), not in a public issue.
