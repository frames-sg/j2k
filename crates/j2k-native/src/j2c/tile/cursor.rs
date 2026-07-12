// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-free decode cursors over immutable retained tile-part metadata.

use super::{PacketLengthMetadata, TilePart};
use crate::reader::BitReader;

pub(crate) enum TilePartCursor<'part, 'data> {
    Merged {
        data: BitReader<'data>,
        packet_lengths: PacketLengthCursor<'part>,
    },
    Separated {
        retained_headers: &'part [BitReader<'data>],
        active_header_reader: usize,
        header: BitReader<'data>,
        body: BitReader<'data>,
        packet_lengths: PacketLengthCursor<'part>,
    },
}

pub(crate) struct PacketLengthCursor<'part> {
    present: bool,
    lengths: &'part [u32],
    next: usize,
}

impl<'part> PacketLengthCursor<'part> {
    fn new(metadata: &'part PacketLengthMetadata) -> Self {
        Self {
            present: metadata.present,
            lengths: &metadata.lengths,
            next: 0,
        }
    }

    fn next(&mut self) -> Option<PacketLengthExpectation> {
        if !self.present {
            return Some(PacketLengthExpectation::NotTracked);
        }
        let packet_length = self.lengths.get(self.next).copied()?;
        self.next += 1;
        Some(PacketLengthExpectation::Length(packet_length))
    }

    fn fully_consumed(&self) -> bool {
        !self.present || self.next == self.lengths.len()
    }
}

enum PacketLengthExpectation {
    NotTracked,
    Length(u32),
}

impl<'data> TilePart<'data> {
    pub(crate) fn cursor<'part>(&'part self) -> Option<TilePartCursor<'part, 'data>> {
        match self {
            Self::Merged(part) => Some(TilePartCursor::Merged {
                data: part.data.clone(),
                packet_lengths: PacketLengthCursor::new(&part.packet_lengths),
            }),
            Self::Separated(part) => Some(TilePartCursor::Separated {
                retained_headers: &part.headers,
                active_header_reader: 0,
                header: part.headers.first()?.clone(),
                body: part.body.clone(),
                packet_lengths: PacketLengthCursor::new(&part.packet_lengths),
            }),
        }
    }
}

impl<'data> TilePartCursor<'_, 'data> {
    pub(crate) fn header(&mut self) -> &mut BitReader<'data> {
        match self {
            Self::Merged { data, .. } => data,
            Self::Separated {
                retained_headers,
                active_header_reader,
                header,
                ..
            } => {
                while header.at_end() && *active_header_reader + 1 < retained_headers.len() {
                    *active_header_reader += 1;
                    *header = retained_headers[*active_header_reader].clone();
                }
                header
            }
        }
    }

    pub(crate) fn body(&mut self) -> &mut BitReader<'data> {
        match self {
            Self::Merged { data, .. } => data,
            Self::Separated { body, .. } => body,
        }
    }

    pub(crate) fn packet_start_offset(&self) -> Option<usize> {
        match self {
            Self::Merged {
                data,
                packet_lengths,
            } if packet_lengths.present => Some(data.offset()),
            Self::Merged { .. } | Self::Separated { .. } => None,
        }
    }

    pub(crate) fn validate_packet_length(&mut self, packet_start: Option<usize>) -> Option<()> {
        let expected = match self {
            Self::Merged { packet_lengths, .. } | Self::Separated { packet_lengths, .. } => {
                packet_lengths.next()?
            }
        };
        let PacketLengthExpectation::Length(expected) = expected else {
            return Some(());
        };
        let packet_start = packet_start?;
        let actual = match self {
            Self::Merged { data, .. } => data.offset().checked_sub(packet_start)?,
            Self::Separated { .. } => return Some(()),
        };
        (actual == expected as usize).then_some(())
    }

    pub(crate) fn validate_all_packet_lengths_consumed(&self) -> Option<()> {
        let consumed = match self {
            Self::Merged { packet_lengths, .. } | Self::Separated { packet_lengths, .. } => {
                packet_lengths.fully_consumed()
            }
        };
        consumed.then_some(())
    }
}

#[cfg(test)]
mod tests;
