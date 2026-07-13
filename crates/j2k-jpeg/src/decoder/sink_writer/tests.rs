// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;
use alloc::vec::Vec;

use j2k_core::RowSink;

use super::SinkWriter;
use crate::backend::Backend;
use crate::error::JpegError;
use crate::internal::scratch::SinkRows;
use crate::output::OutputWriter;

#[derive(Default)]
struct RecordingSink {
    rows: Vec<(u32, Vec<u8>)>,
}

impl RowSink<u8> for RecordingSink {
    type Error = JpegError;

    fn write_row(&mut self, y: u32, row: &[u8]) -> Result<(), Self::Error> {
        self.rows.push((y, row.to_vec()));
        Ok(())
    }
}

fn writer_for(sink: &mut RecordingSink, width: usize) -> SinkWriter<'_, RecordingSink> {
    SinkWriter::new(
        sink,
        SinkRows {
            top_row: vec![0xA5; width * 3],
            bottom_row: vec![0x5A; width * 3],
        },
        Backend::detect(),
    )
}

#[test]
fn rgb_row_interleaves_channels_across_backend_tail_boundary() {
    let mut sink = RecordingSink::default();
    let mut writer = writer_for(&mut sink, 9);

    writer
        .write_rgb_row(
            7,
            &[0, 1, 2, 3, 4, 5, 6, 7, 8],
            &[10, 11, 12, 13, 14, 15, 16, 17, 18],
            &[20, 21, 22, 23, 24, 25, 26, 27, 28],
        )
        .expect("recording sink accepts RGB row");

    assert_eq!(
        sink.rows,
        vec![(
            7,
            vec![
                0, 10, 20, 1, 11, 21, 2, 12, 22, 3, 13, 23, 4, 14, 24, 5, 15, 25, 6, 16, 26, 7, 17,
                27, 8, 18, 28,
            ],
        )]
    );
}

#[test]
fn ycbcr_row_converts_and_clamps_across_backend_tail_boundary() {
    let mut sink = RecordingSink::default();
    let mut writer = writer_for(&mut sink, 9);

    writer
        .write_ycbcr_row(
            23,
            &[0, 32, 64, 76, 100, 160, 192, 224, 255],
            &[0, 64, 128, 85, 150, 192, 255, 128, 255],
            &[0, 192, 128, 255, 200, 64, 255, 0, 128],
        )
        .expect("recording sink accepts YCbCr row");

    assert_eq!(
        sink.rows,
        vec![(
            23,
            vec![
                0, 135, 0, 122, 8, 0, 64, 64, 64, 254, 0, 0, 201, 41, 139, 70, 184, 255, 255, 58,
                255, 45, 255, 224, 255, 211, 255,
            ],
        )]
    );
}

#[test]
fn gray_row_expands_channels_and_preserves_maximum_row_index() {
    let mut sink = RecordingSink::default();
    let mut writer = writer_for(&mut sink, 9);

    writer
        .write_gray_row(u32::MAX, &[0, 1, 2, 3, 4, 5, 6, 7, 255])
        .expect("recording sink accepts grayscale row");

    assert_eq!(
        sink.rows,
        vec![(
            u32::MAX,
            vec![
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 255, 255,
                255,
            ],
        )]
    );
}
