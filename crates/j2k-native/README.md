# j2k-native

Native JPEG 2000 Part 1 / HTJ2K Part 15 engine used by J2K APIs and adapters.

This crate owns codestream parsing, native encode/decode helpers, packetization
support, JP2/JPH still-image wrapper handling, HTJ2K cleanup/refinement table
helpers, and header inspection helpers used by higher-level crates.

The support boundary follows the public facade: raw J2K/J2C codestreams, JP2
still-image files, raw HTJ2K codestreams, and JPH still-image files. JPX /
JPEG 2000 Part 2 extension support is outside this engine's current claim.

Most application code should use `j2k` instead.
