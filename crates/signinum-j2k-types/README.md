# signinum-j2k-types

Shared JPEG 2000 encode-stage contract types for the signinum workspace.

This crate is the neutral public contract between the `signinum-j2k` adapter
surface and the `signinum-j2k-native` codec engine: encode-stage job, output,
and report types are defined once here so neither crate mirrors the other's
definitions. It contains plain data types only — codec behavior lives in
`signinum-j2k-native`, and the encode-stage accelerator traits stay in their
owning crates.
