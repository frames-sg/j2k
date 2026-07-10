// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_jpeg::{JpegError, RowSink};

#[derive(Default)]
pub(crate) struct NullSink;

impl RowSink<u8> for NullSink {
    type Error = JpegError;

    fn write_row(&mut self, _y: u32, _row: &[u8]) -> Result<(), JpegError> {
        Ok(())
    }
}
