// SPDX-License-Identifier: MIT OR Apache-2.0

//! Borrowed-input byte descriptors shared by prepared decode plans.

/// Byte range inside a complete encoded J2K, JP2, or JPH input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct J2kCodestreamRange {
    /// Zero-based offset from the beginning of the encoded input.
    pub offset: usize,
    /// Number of bytes in the range.
    pub length: usize,
}

impl J2kCodestreamRange {
    /// End offset, or `None` when the range overflows `usize`.
    #[must_use]
    pub const fn end(self) -> Option<usize> {
        self.offset.checked_add(self.length)
    }
}

/// One retained cleanup/refinement record for an HTJ2K code block.
///
/// The first record for a code block contains its cleanup range and may
/// contain the first refinement fragment. When refinement bytes are split
/// across packets, immediately following continuation records have an empty
/// cleanup range and a non-empty refinement range. Consumers use the code
/// block's declared refinement length to determine where the next block starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HtCodeBlockPayloadRanges {
    /// Cleanup-pass payload range.
    pub cleanup: J2kCodestreamRange,
    /// SigProp/MagRef payload fragment, when refinement is present.
    pub refinement: Option<J2kCodestreamRange>,
}

/// Ordered encoded-input fragment span for one classic JPEG 2000 code block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct J2kClassicCodeBlockPayload {
    /// Index of the first fragment in the owning plan's range array.
    pub first_range: usize,
    /// Number of ordered fragments contributing to this code block.
    pub range_count: usize,
    /// Total bytes after concatenating every fragment.
    pub combined_length: usize,
}

impl J2kClassicCodeBlockPayload {
    /// Exclusive fragment index, or `None` when the descriptor overflows.
    #[must_use]
    pub const fn end_range(self) -> Option<usize> {
        self.first_range.checked_add(self.range_count)
    }
}
