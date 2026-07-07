// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use j2k_jpeg::{Decoder, JpegError, RowSink};
use libfuzzer_sys::fuzz_target;

const MAX_ROW_STREAM_BYTES: usize = 1 << 20;

fuzz_target!(|data: &[u8]| {
    let Ok(decoder) = Decoder::new(data) else {
        return;
    };

    let (width, height) = decoder.info().dimensions;
    let Some(max_rows_bytes) = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
    else {
        return;
    };
    if max_rows_bytes == 0 || max_rows_bytes > MAX_ROW_STREAM_BYTES {
        return;
    }

    let mut sink = CountingSink { bytes_seen: 0 };
    let _ = decoder.decode_rows(&mut sink);
});

struct CountingSink {
    bytes_seen: usize,
}

impl RowSink<u8> for CountingSink {
    type Error = JpegError;

    fn write_row(&mut self, _y: u32, row: &[u8]) -> Result<(), JpegError> {
        self.bytes_seen = self.bytes_seen.saturating_add(row.len());
        Ok(())
    }
}
