// SPDX-License-Identifier: MIT OR Apache-2.0

//! Byte-level edits of JPEG 2000 codestreams around their packets, for tests
//! of what a decode reads: data after a tile's last packet, PLT lengths past
//! it, damaged packets and packed packet headers, and tile-parts of several
//! tiles interleaved. Codestreams are raw (`SOC` first), each tile-part a
//! `SOT` marker segment with a nonzero `Psot`, and `EOC` last.

use std::ops::Range;

const SOT: [u8; 2] = [0xff, 0x90];
const SOD: [u8; 2] = [0xff, 0x93];
const PLT: [u8; 2] = [0xff, 0x58];
const PPM: [u8; 2] = [0xff, 0x60];
const PPT: [u8; 2] = [0xff, 0x61];
const EOC: [u8; 2] = [0xff, 0xd9];

fn marker_segment_end(bytes: &[u8], at: usize) -> usize {
    at + 2 + usize::from(u16::from_be_bytes([bytes[at + 2], bytes[at + 3]]))
}

fn first_tile_part(codestream: &[u8]) -> usize {
    let mut at = 2;
    while codestream[at..at + 2] != SOT {
        at = marker_segment_end(codestream, at);
    }
    at
}

fn tile_part_end(codestream: &[u8], sot: usize) -> usize {
    let psot = u32::from_be_bytes(
        codestream[sot + 6..sot + 10]
            .try_into()
            .expect("Psot is four bytes"),
    );
    assert_ne!(psot, 0, "every tile-part signals its length");
    sot + usize::try_from(psot).expect("Psot fits usize")
}

/// The tile-parts of `codestream`, each from its `SOT` marker to its end.
fn tile_parts(codestream: &[u8]) -> Vec<Range<usize>> {
    let mut parts = Vec::new();
    let mut at = first_tile_part(codestream);
    while codestream[at..at + 2] == SOT {
        let end = tile_part_end(codestream, at);
        parts.push(at..end);
        at = end;
    }
    assert_eq!(codestream[at..], EOC, "EOC follows the last tile-part");
    parts
}

/// `codestream` with `edit` applied to each tile-part, from its `SOT` marker
/// to its end, and each `Psot` set to the edited length. `edit` is called
/// once per tile-part, in codestream order.
///
/// # Panics
///
/// When `codestream` is not as the module describes, or an edited tile-part
/// is too long for its `Psot`.
#[must_use]
pub fn edit_tile_parts(codestream: &[u8], mut edit: impl FnMut(&mut Vec<u8>)) -> Vec<u8> {
    let parts = tile_parts(codestream);
    let mut edited = codestream[..parts[0].start].to_vec();
    for part in &parts {
        let mut bytes = codestream[part.clone()].to_vec();
        edit(&mut bytes);
        let psot = u32::try_from(bytes.len()).expect("edited Psot fits");
        bytes[6..10].copy_from_slice(&psot.to_be_bytes());
        edited.extend_from_slice(&bytes);
    }
    edited.extend_from_slice(&EOC);
    edited
}

/// `codestream` with a zero byte after the last packet of each tile-part,
/// inside it.
#[must_use]
pub fn with_byte_after_last_packets(codestream: &[u8]) -> Vec<u8> {
    edit_tile_parts(codestream, |part| part.push(0))
}

/// Add one packet length (5) at the end of a tile-part's first PLT marker.
/// `part` runs from the tile-part's `SOT` marker; its `Psot` is left to the
/// caller (see [`edit_tile_parts`]).
///
/// # Panics
///
/// When the tile-part header has no PLT marker.
pub fn add_packet_length(part: &mut Vec<u8>) {
    let mut at = 12;
    while part[at..at + 2] != PLT {
        assert_ne!(part[at..at + 2], SOD, "tile-part has a PLT marker");
        at = marker_segment_end(part, at);
    }
    let length = u16::from_be_bytes([part[at + 2], part[at + 3]]);
    part.insert(at + 2 + usize::from(length), 5);
    part[at + 2..at + 4].copy_from_slice(&(length + 1).to_be_bytes());
}

/// `codestream` with one more packet length in each tile-part's PLT marker
/// than the tile-part has packets.
#[must_use]
pub fn with_surplus_packet_lengths(codestream: &[u8]) -> Vec<u8> {
    edit_tile_parts(codestream, add_packet_length)
}

/// The byte range of every packet of `codestream`, header and body, in
/// codestream order (tile-part by tile-part, each in its progression order),
/// from the PLT lengths each tile-part signals for its in-band packets.
///
/// # Panics
///
/// When a tile-part has no PLT marker or its lengths do not end at its end.
#[must_use]
pub fn packet_ranges(codestream: &[u8]) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    for part in tile_parts(codestream) {
        let mut at = part.start + 12;
        let mut lengths = Vec::new();
        while codestream[at..at + 2] != SOD {
            let end = marker_segment_end(codestream, at);
            if codestream[at..at + 2] == PLT {
                let mut length = 0_usize;
                for &byte in &codestream[at + 5..end] {
                    length = (length << 7) | usize::from(byte & 0x7f);
                    if byte & 0x80 == 0 {
                        lengths.push(length);
                        length = 0;
                    }
                }
                assert_eq!(length, 0, "PLT lengths terminate");
            }
            at = end;
        }
        assert!(!lengths.is_empty(), "tile-part has PLT lengths");
        let mut start = at + 2;
        for length in lengths {
            ranges.push(start..start + length);
            start += length;
        }
        assert_eq!(start, part.end, "PLT lengths end at the tile-part's end");
    }
    ranges
}

/// `codestream` with every byte of the packets at `ranges` set to `byte`,
/// lengths unchanged.
#[must_use]
pub fn with_packets_filled(codestream: &[u8], ranges: &[Range<usize>], byte: u8) -> Vec<u8> {
    let mut edited = codestream.to_vec();
    for range in ranges {
        edited[range.clone()].fill(byte);
    }
    edited
}

/// `codestream` with each tile-part's PLT lengths passed through `edit` and
/// written back as one PLT marker, packets unchanged.
///
/// # Panics
///
/// When a tile-part has no PLT marker or its new lengths do not fit one.
#[must_use]
pub fn with_packet_lengths_edited(
    codestream: &[u8],
    mut edit: impl FnMut(&mut Vec<usize>),
) -> Vec<u8> {
    edit_tile_parts(codestream, |part| {
        let mut header = part[..12].to_vec();
        let mut lengths = Vec::new();
        let mut at = 12;
        while part[at..at + 2] != SOD {
            let end = marker_segment_end(part, at);
            if part[at..at + 2] == PLT {
                let mut length = 0_usize;
                for &byte in &part[at + 5..end] {
                    length = (length << 7) | usize::from(byte & 0x7f);
                    if byte & 0x80 == 0 {
                        lengths.push(length);
                        length = 0;
                    }
                }
            } else {
                header.extend_from_slice(&part[at..end]);
            }
            at = end;
        }
        assert!(!lengths.is_empty(), "tile-part has PLT lengths");
        edit(&mut lengths);
        let mut entries = Vec::new();
        for &length in &lengths {
            let mut groups = vec![u8::try_from(length & 0x7f).expect("seven bits")];
            let mut rest = length >> 7;
            while rest != 0 {
                groups.push(u8::try_from(rest & 0x7f).expect("seven bits") | 0x80);
                rest >>= 7;
            }
            entries.extend(groups.iter().rev());
        }
        let marker_length = u16::try_from(entries.len() + 3).expect("lengths fit one PLT marker");
        header.extend_from_slice(&PLT);
        header.extend_from_slice(&marker_length.to_be_bytes());
        header.push(0);
        header.extend_from_slice(&entries);
        header.extend_from_slice(&part[at..]);
        *part = header;
    })
}

/// `codestream` with its tile-parts reordered so that tiles take turns: every
/// tile's first tile-part in tile order, then every second one, and so on.
/// Each tile's tile-parts keep their order, as the standard requires.
///
/// # Panics
///
/// When `codestream` is not as the module describes.
#[must_use]
pub fn with_tile_parts_interleaved(codestream: &[u8]) -> Vec<u8> {
    let parts = tile_parts(codestream);
    let mut tiles: Vec<(u16, Vec<Range<usize>>)> = Vec::new();
    for part in parts.iter().cloned() {
        let tile = u16::from_be_bytes([codestream[part.start + 4], codestream[part.start + 5]]);
        match tiles.iter_mut().find(|(index, _)| *index == tile) {
            Some((_, tile_parts)) => tile_parts.push(part),
            None => tiles.push((tile, vec![part])),
        }
    }
    let mut interleaved = codestream[..parts[0].start].to_vec();
    let rounds = tiles
        .iter()
        .map(|(_, parts)| parts.len())
        .max()
        .unwrap_or(0);
    for round in 0..rounds {
        for (_, tile_parts) in &tiles {
            if let Some(part) = tile_parts.get(round) {
                interleaved.extend_from_slice(&codestream[part.clone()]);
            }
        }
    }
    interleaved.extend_from_slice(&EOC);
    interleaved
}

/// The bytes of the packed packet headers of a codestream of one tile-part
/// with one PPM segment in its main header or one PPT segment in its
/// tile-part header: every packet's header, in progression order.
///
/// # Panics
///
/// When `codestream` has no such single segment.
#[must_use]
pub fn packed_packet_headers(codestream: &[u8]) -> Range<usize> {
    let first_part = first_tile_part(codestream);
    let mut at = 2;
    while at < first_part {
        if codestream[at..at + 2] == PPM {
            // Zppm, then the first tile-part's Nppm.
            let end = marker_segment_end(codestream, at);
            return at + 9..end;
        }
        at = marker_segment_end(codestream, at);
    }
    let mut at = first_part + 12;
    while codestream[at..at + 2] != SOD {
        if codestream[at..at + 2] == PPT {
            return at + 5..marker_segment_end(codestream, at);
        }
        at = marker_segment_end(codestream, at);
    }
    panic!("no packed packet headers");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SOC, a SIZ-like dummy segment, one tile-part with PLT lengths 2 and 3
    /// for five packet bytes, EOC.
    fn codestream() -> Vec<u8> {
        let mut bytes = vec![0xff, 0x4f, 0xff, 0x51, 0, 3, 0];
        let header = [0xff, 0x58, 0, 5, 0, 2, 3, 0xff, 0x93];
        let psot = u32::try_from(12 + header.len() + 5).expect("small");
        bytes.extend_from_slice(&[0xff, 0x90, 0, 10, 0, 0]);
        bytes.extend_from_slice(&psot.to_be_bytes());
        bytes.extend_from_slice(&[0, 1]);
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&[1, 2, 3, 4, 5]);
        bytes.extend_from_slice(&EOC);
        bytes
    }

    #[test]
    fn packet_ranges_follow_plt_lengths() {
        let bytes = codestream();
        let ranges = packet_ranges(&bytes);
        assert_eq!(ranges.len(), 2);
        assert_eq!(bytes[ranges[0].clone()], [1, 2]);
        assert_eq!(bytes[ranges[1].clone()], [3, 4, 5]);
    }

    #[test]
    fn edits_keep_psot_consistent() {
        let bytes = codestream();
        let after = with_byte_after_last_packets(&bytes);
        assert_eq!(after.len(), bytes.len() + 1);
        assert_eq!(tile_parts(&after)[0].len(), tile_parts(&bytes)[0].len() + 1);
        let surplus = with_surplus_packet_lengths(&bytes);
        assert_eq!(surplus.len(), bytes.len() + 1);
        assert_eq!(
            tile_parts(&surplus)[0].len(),
            tile_parts(&bytes)[0].len() + 1
        );
        let filled = with_packets_filled(&bytes, &packet_ranges(&bytes)[1..], 0xff);
        assert_eq!(filled[filled.len() - 5..filled.len() - 2], [0xff; 3]);
        assert_eq!(with_packet_lengths_edited(&bytes, |_| {}), bytes);
        let moved = with_packet_lengths_edited(&bytes, |lengths| {
            lengths[0] += 1;
            lengths[1] -= 1;
        });
        let ranges = packet_ranges(&moved);
        assert_eq!(moved[ranges[0].clone()], [1, 2, 3]);
        assert_eq!(moved[ranges[1].clone()], [4, 5]);
    }

    /// A tile-part of `tile` whose header holds `header` and whose data is
    /// `data`.
    fn tile_part(tile: u16, header: &[u8], data: &[u8]) -> Vec<u8> {
        let psot = u32::try_from(12 + header.len() + 2 + data.len()).expect("small");
        let mut part = vec![0xff, 0x90, 0, 10];
        part.extend_from_slice(&tile.to_be_bytes());
        part.extend_from_slice(&psot.to_be_bytes());
        part.extend_from_slice(&[0, 0]);
        part.extend_from_slice(header);
        part.extend_from_slice(&SOD);
        part.extend_from_slice(data);
        part
    }

    #[test]
    fn interleaving_lets_tiles_take_turns() {
        let main = [0xff, 0x4f, 0xff, 0x51, 0, 3, 0];
        let parts = [
            tile_part(0, &[], &[1]),
            tile_part(0, &[], &[2]),
            tile_part(1, &[], &[3]),
        ];
        let mut bytes = main.to_vec();
        for part in &parts {
            bytes.extend_from_slice(part);
        }
        bytes.extend_from_slice(&EOC);
        let mut expected = main.to_vec();
        for part in [&parts[0], &parts[2], &parts[1]] {
            expected.extend_from_slice(part);
        }
        expected.extend_from_slice(&EOC);
        assert_eq!(with_tile_parts_interleaved(&bytes), expected);
    }

    #[test]
    fn packed_headers_are_found_in_ppt_and_ppm() {
        let ppt = [0xff, 0x61, 0, 6, 0, 7, 8, 9];
        let mut bytes = vec![0xff, 0x4f, 0xff, 0x51, 0, 3, 0];
        bytes.extend_from_slice(&tile_part(0, &ppt, &[1]));
        bytes.extend_from_slice(&EOC);
        assert_eq!(bytes[packed_packet_headers(&bytes)], [7, 8, 9]);
        // Zppm, then Nppm = 3 for the one tile-part.
        let ppm = [0xff, 0x60, 0, 10, 0, 0, 0, 0, 3, 7, 8, 9];
        let mut bytes = vec![0xff, 0x4f, 0xff, 0x51, 0, 3, 0];
        bytes.extend_from_slice(&ppm);
        bytes.extend_from_slice(&tile_part(0, &[], &[1]));
        bytes.extend_from_slice(&EOC);
        assert_eq!(bytes[packed_packet_headers(&bytes)], [7, 8, 9]);
    }
}
