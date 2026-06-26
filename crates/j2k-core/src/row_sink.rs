// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::sample::Sample;

/// Destination for row-streaming decode output.
pub trait RowSink<S: Sample> {
    /// Error returned by the sink when it cannot accept a row.
    type Error: core::error::Error + Send + Sync + 'static;

    /// Write one decoded row at source/output row index `y`.
    fn write_row(&mut self, y: u32, row: &[S]) -> Result<(), Self::Error>;
}
