// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::backend::Backend;
use crate::error::JpegError;
use crate::internal::scratch::SinkRows;
use crate::output::{InterleavedRgbWriter, OutputWriter};
use j2k_core::RowSink;

pub(crate) struct SinkWriter<'a, S> {
    sink: &'a mut S,
    rows: SinkRows,
    backend: Backend,
}

impl<'a, S> SinkWriter<'a, S> {
    pub(crate) fn new(sink: &'a mut S, rows: SinkRows, backend: Backend) -> Self {
        debug_assert_eq!(rows.top_row.len(), rows.bottom_row.len());
        Self {
            sink,
            rows,
            backend,
        }
    }

    pub(crate) fn into_rows(self) -> SinkRows {
        self.rows
    }
}

impl<S> InterleavedRgbWriter for SinkWriter<'_, S>
where
    S: RowSink<u8, Error = JpegError>,
{
    fn with_rgb_rows<R, F>(&mut self, y: u32, row_count: usize, fill: F) -> Result<R, JpegError>
    where
        F: FnOnce(&mut [u8], Option<&mut [u8]>) -> Result<R, JpegError>,
    {
        let result = match row_count {
            1 => fill(&mut self.rows.top_row, None),
            2 => fill(&mut self.rows.top_row, Some(&mut self.rows.bottom_row)),
            _ => unreachable!("SinkWriter only supports one or two rows"),
        }?;
        self.sink.write_row(y, &self.rows.top_row)?;
        if row_count == 2 {
            self.sink.write_row(y + 1, &self.rows.bottom_row)?;
        }
        Ok(result)
    }
}

impl<S> OutputWriter for SinkWriter<'_, S>
where
    S: RowSink<u8, Error = JpegError>,
{
    fn write_rgb_row(
        &mut self,
        y: u32,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
    ) -> Result<(), JpegError> {
        self.backend
            .fill_rgb_row_from_rgb(r_row, g_row, b_row, &mut self.rows.top_row);
        self.sink.write_row(y, &self.rows.top_row)
    }

    fn write_ycbcr_row(
        &mut self,
        y: u32,
        y_row: &[u8],
        cb_row: &[u8],
        cr_row: &[u8],
    ) -> Result<(), JpegError> {
        self.backend
            .fill_rgb_row_from_ycbcr(y_row, cb_row, cr_row, &mut self.rows.top_row);
        self.sink.write_row(y, &self.rows.top_row)
    }

    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) -> Result<(), JpegError> {
        self.backend
            .fill_rgb_row_from_gray(gray_row, &mut self.rows.top_row);
        self.sink.write_row(y, &self.rows.top_row)
    }
}
