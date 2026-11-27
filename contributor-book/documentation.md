# Writing documentation

J2K has four kinds of documentation. Each has one job, and a fact should live
in exactly one of them:

| Where | Audience | What goes there |
| --- | --- | --- |
| `README.md` | Someone deciding whether to use J2K | What it is, a taste of the API, where to go next |
| `book/` | Someone using J2K | How to do things: decode, encode, GPUs, bindings, limits |
| `contributor-book/` | Someone changing J2K | How the code is laid out, how to test, benchmark, and release |
| `crates/*/README.md` | Someone who found one crate on crates.io | What this crate is for and its feature flags |

API details belong in rustdoc. The book explains how the pieces fit together
and links to docs.rs for signatures; it doesn't restate them.

## Voice

Write the way you'd explain J2K to a colleague who knows Rust and has a rough
idea what JPEG 2000 is.

- **Lead with what the reader can do.** "Decode a 1024 x 1024 region at
  quarter resolution:" followed by code beats a paragraph about region
  planning.
- **Say it once.** If the format support matrix lives in
  `book/format-support.md`, other pages link to it instead of summarizing it.
- **Keep sentences short and concrete.** Name the type or function. Prefer
  "returns an error" to "fails closed".
- **Give the common case first and the exceptions after.** Rare edge cases go
  in a short list at the end of the section, or in rustdoc.
- **Use "we" for the project and "you" for the reader.**
- **Round numbers in prose.** "About twice as fast" in the book; exact figures
  belong in the tables in `book/performance.md`.

Avoid these:

- Audit language: "evidence", "promoted", "signoff", "fail-closed",
  "publication-eligible", "reviewed", "canonical", "contract" (except for the
  actual API stability policy).
- Process details in user docs. CI job names, workflow dispatch rules, run
  IDs, and release gates go in the contributor book.
- Test function names in user docs. Link to the test file in the contributor
  book if it matters.
- Version numbers that go stale. Don't write "as of 0.14.0" or "in this
  release" outside the changelog and `SECURITY.md`.
- Changelog material in guides. "Since 0.12, ..." or "this used to ..." belongs
  in the release notes.
- Hedging stacks such as "may potentially", "in some cases can", or "generally
  tends to".
- Bold run-in labels on every bullet. Use them only when a list is long enough
  to need scanning.

Names are exempt: a file, job, or command such as `xtask/release-evidence/` or
`cargo xtask j2k-bench-signoff` keeps its real name.

## Structure

- Use sentence case for headings: "Region decoding", not "Region Decoding".
- Keep pages under about 300 lines. Split a page that grows past that.
- Wrap prose at 80 columns. Code blocks can be wider.
- Code samples must compile against the current API. Prefer short samples
  with `?` over long ones with setup; mention what was left out.
- Put long or optional material in `<details>` blocks in the README, the way
  most large Rust projects do.
- Link with paths relative to the file, so links work on GitHub and in
  mdBook.

Both books follow the [mdBook](https://rust-lang.github.io/mdBook/) layout.
`SUMMARY.md` is the table of contents. To preview one locally:

```bash
cargo install mdbook
mdbook serve book
```

## Crate READMEs

Every crate README uses the same shape, so crates.io pages read as one
project:

````markdown
# J2K CUDA

> CUDA decode and encode for [J2K](https://github.com/frames-sg/j2k)

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-cuda.svg)](https://crates.io/crates/j2k-cuda)
[![Documentation](https://docs.rs/j2k-cuda/badge.svg)](https://docs.rs/j2k-cuda)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

One or two paragraphs: what the crate does and when to use it instead of a
neighbouring crate.

## Usage

A short example, if the crate has an API people call directly.

## Feature flags

- `cuda-runtime`: what it turns on.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
````

Crates that exist only to share code between other J2K crates say so in their
first sentence and point to the crate people should use instead. They don't
need a usage section.

Crates that aren't published (`publish = false`) drop the crates.io and docs.rs
badges.

## Changelog

User-visible changes go under `Unreleased` in [`CHANGELOG.md`](../CHANGELOG.md)
in the same pull request. That covers public API and behavior changes, new
features, bug fixes users could notice, output that changes bytes, and
performance changes large enough to mention. Internal refactors, test-only
changes, and doc fixes don't need entries.

`Unreleased` is organized by subject (`###` groups such as "Decoding",
"Encoding", "GPU", "Python"). Add an entry to the group it belongs to. Within
a group:

- Describe the effect on users, not the implementation. One to three
  sentences is usually enough.
- Mark breaking changes with **Breaking:** and say what to change.
- Mark changes to output bytes with **Output change:**.
- Link the issue or pull request at the end:
  `([#304](https://github.com/frames-sg/j2k/issues/304))`.
- Leave a blank line between entries that run longer than one line.

If a later pull request changes something already listed under `Unreleased`,
edit that entry so it describes the end result.

Published releases live on [GitHub
Releases](https://github.com/frames-sg/j2k/releases). At release time the
`Unreleased` entries become the release notes (see
[Releasing](releasing.md)). The pull request that starts the next version
empties the section again.
