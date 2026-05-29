// SPDX-License-Identifier: Apache-2.0

use crate::sample::Sample;

/// Destination for row-streaming decode.
pub trait RowSink<S: Sample> {
    /// Error type returned by the caller-provided sink.
    type Error: core::error::Error + Send + Sync + 'static;

    /// Write one decoded row at source/output y coordinate `y`.
    fn write_row(&mut self, y: u32, row: &[S]) -> Result<(), Self::Error>;
}
