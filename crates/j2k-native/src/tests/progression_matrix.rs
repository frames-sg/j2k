// SPDX-License-Identifier: MIT OR Apache-2.0

//! Region decodes over codestreams the native encoder generates across
//! progression orders and packet layouts, which the T.803 codestreams do not
//! sweep. Every combination of values from any three axes appears in some
//! codestream. Each region must match the same area of a full decode, and
//! reversible codestreams must decode to their source exactly.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::iter::{once, repeat_n};

use crate::EncodeProgressionOrder::{Cprl, Lrcp, Pcrl, Rlcp, Rpcl};
use crate::{encode, DecodeSettings, DecoderContext, EncodeOptions, Image};

// Every edge tile of both tilings is at least 32 samples on each side, so all
// of them support five decomposition levels.
const WIDTH: u32 = 235;
const HEIGHT: u32 = 181;

/// Each axis's values as case names show them, in the order `Case::new` reads
/// them. An empty name leaves the option out of the case name.
const AXES: [&[&str]; 12] = [
    &["lrcp", "rlcp", "rpcl", "pcrl", "cprl"],
    &["classic", "ht"],
    &["53", "97"],
    &["1layer", "3layers"],
    &["cb64", "cb16", "cb32x16"],
    &["untiled", "tiles64", "tiles61x47"],
    &["", "tileparts"],
    &["plain", "sop-eph", "ppm", "ppt"],
    &["", "plt-tlm"],
    &["gray", "rgb"],
    &["8bit", "12bit"],
    &["3levels", "5levels"],
];

/// Three axes in order, with one value each.
type Triple = [(usize, usize); 3];

struct Case {
    name: String,
    components: u8,
    bit_depth: u8,
    options: EncodeOptions,
}

impl Case {
    fn new(row: [usize; AXES.len()]) -> Self {
        let levels = [3, 5][row[11]];
        let code_blocks = [None, Some((2, 2)), Some((3, 2))][row[4]];
        let headers = row[7];
        Self {
            name: row
                .iter()
                .zip(AXES)
                .map(|(&value, names)| names[value])
                .filter(|name| !name.is_empty())
                .collect::<Vec<_>>()
                .join("-"),
            components: [1, 3][row[9]],
            bit_depth: [8, 12][row[10]],
            options: EncodeOptions {
                progression_order: [Lrcp, Rlcp, Rpcl, Pcrl, Cprl][row[0]],
                use_ht_block_coding: row[1] == 1,
                reversible: row[2] == 0,
                num_layers: [1, 3][row[3]],
                code_block_width_exp: code_blocks.map_or(4, |(width, _)| width),
                code_block_height_exp: code_blocks.map_or(4, |(_, height)| height),
                // The smallest precincts those code-blocks allow.
                precinct_exponents: code_blocks.map_or_else(Vec::new, |(width, height)| {
                    once((width + 2, height + 2))
                        .chain(repeat_n((width + 3, height + 3), usize::from(levels)))
                        .collect()
                }),
                tile_size: [None, Some((64, 64)), Some((61, 47))][row[5]],
                tile_part_packet_limit: [None, Some(3)][row[6]],
                write_sop: headers == 1,
                write_eph: headers == 1,
                write_ppm: headers == 2,
                write_ppt: headers == 3,
                write_plt: row[8] == 1,
                write_tlm: row[8] == 1,
                num_decomposition_levels: levels,
                ..EncodeOptions::default()
            },
        }
    }

    /// Deterministic noise, so every sub-band carries significant coefficients,
    /// and its codestream.
    fn encode(&self) -> (Vec<u8>, Vec<u8>) {
        let mut state = 0x2545_f491_u32;
        let mut source = Vec::new();
        for _ in 0..WIDTH * HEIGHT * u32::from(self.components) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            if self.bit_depth > 8 {
                source.extend_from_slice(&((state >> 20) as u16).to_le_bytes());
            } else {
                source.push((state >> 24) as u8);
            }
        }
        let codestream = encode(
            &source,
            WIDTH,
            HEIGHT,
            u16::from(self.components),
            self.bit_depth,
            false,
            &self.options,
        )
        .unwrap_or_else(|err| panic!("encode {}: {err:?}", self.name));
        (source, codestream)
    }
}

fn triples() -> Vec<Triple> {
    let mut triples = Vec::new();
    for (a, a_values) in AXES.iter().enumerate() {
        for (b, b_values) in AXES.iter().enumerate().skip(a + 1) {
            for (c, c_values) in AXES.iter().enumerate().skip(b + 1) {
                for x in 0..a_values.len() {
                    for y in 0..b_values.len() {
                        for z in 0..c_values.len() {
                            triples.push([(a, x), (b, y), (c, z)]);
                        }
                    }
                }
            }
        }
    }
    triples
}

fn holds(row: &[Option<usize>], triple: &Triple) -> bool {
    triple.iter().all(|&(axis, value)| row[axis] == Some(value))
}

/// Rows that together hold every triple. Each row starts from the first
/// uncovered triple and gives each other axis the value that completes the
/// most uncovered triples, breaking ties in a rotating order so the filler
/// values vary.
fn rows() -> Vec<[usize; AXES.len()]> {
    let mut uncovered = triples();
    let mut rows = Vec::new();
    while let Some(first) = uncovered.first() {
        let mut row = [None; AXES.len()];
        for &(axis, value) in first {
            row[axis] = Some(value);
        }
        for axis in 0..AXES.len() {
            if row[axis].is_some() {
                continue;
            }
            let count = AXES[axis].len();
            let preferred = rows.len() % count;
            row[axis] = (0..count).max_by_key(|&value| {
                let mut candidate = row;
                candidate[axis] = Some(value);
                let completed = uncovered
                    .iter()
                    .filter(|triple| holds(&candidate, triple))
                    .count();
                (completed, Reverse((value + count - preferred) % count))
            });
        }
        uncovered.retain(|triple| !holds(&row, triple));
        rows.push(row.map(|value| value.expect("every axis has a value")));
    }
    rows
}

fn crop(samples: &[u8], width: u32, height: u32, roi: (u32, u32, u32, u32)) -> Vec<u8> {
    let pixel_bytes = samples.len() / (width * height) as usize;
    let (x, y, roi_width, roi_height) = roi;
    (y..y + roi_height)
        .flat_map(|row| {
            let start = (row * width + x) as usize * pixel_bytes;
            &samples[start..start + roi_width as usize * pixel_bytes]
        })
        .copied()
        .collect()
}

fn check_regions(case: &Case, source: &[u8], codestream: &[u8]) {
    let name = &case.name;
    for reduction in [0_u8, 1] {
        let image = Image::new_with_reduction(codestream, &DecodeSettings::default(), reduction)
            .unwrap_or_else(|err| panic!("parse {name}: {err:?}"));
        let (width, height) = (image.width(), image.height());
        let mut context = DecoderContext::default();
        let mut lines = DecoderContext::default();
        lines.set_line_region_idwt(true);
        let native = image
            .decode_native_with_context(&mut context)
            .expect(name)
            .data;
        let packed = image.decode_with_context(&mut context).expect(name).data;
        if case.options.reversible && reduction == 0 {
            assert!(
                native == source,
                "{name}: lossless decode differs from the source"
            );
        }
        // A tile corner, or the middle of an untiled image.
        let (corner_x, corner_y) = case
            .options
            .tile_size
            .map_or((width / 2, height / 2), |(w, h)| {
                (w.div_ceil(1 << reduction), h.div_ceil(1 << reduction))
            });
        for roi in [
            (0, 0, width, height),
            (width / 5, height / 6, width / 2, height / 2),
            (width / 3, height / 3, width / 7, height / 9),
            (corner_x - 5, corner_y - 3, 11, 7),
            (width - 13, height - 9, 13, 9),
            (width - 1, height - 1, 1, 1),
        ] {
            for (context, idwt) in [(&mut context, "whole-window"), (&mut lines, "streamed")] {
                let label = format!("{name} reduction={reduction} roi={roi:?} {idwt}");
                let region = image
                    .decode_native_region_with_context(roi, context)
                    .expect(&label);
                assert!(
                    region.data == crop(&native, width, height, roi),
                    "{label}: native"
                );
                let region = image
                    .decode_region_with_context(roi, context)
                    .expect(&label);
                assert!(
                    region.data == crop(&packed, width, height, roi),
                    "{label}: packed"
                );
            }
        }
    }
}

#[test]
fn rows_hold_every_combination_of_three_axis_values() {
    let rows = rows();
    for triple in triples() {
        assert!(
            rows.iter()
                .any(|row| triple.iter().all(|&(axis, value)| row[axis] == value)),
            "{triple:?}"
        );
    }
}

#[test]
fn regions_match_full_decodes_across_progressions_and_packet_layouts() {
    let cases = rows().into_iter().map(Case::new).collect::<Vec<_>>();
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    std::thread::scope(|scope| {
        for chunk in cases.chunks(cases.len().div_ceil(threads)) {
            scope.spawn(move || {
                for case in chunk {
                    let (source, codestream) = case.encode();
                    check_regions(case, &source, &codestream);
                }
            });
        }
    });
}

#[test]
#[ignore = "writes the matrix codestreams to J2K_PROGRESSION_MATRIX_DIR; run explicitly with --ignored"]
fn write_progression_matrix_codestreams() {
    let dir =
        std::path::PathBuf::from(std::env::var_os("J2K_PROGRESSION_MATRIX_DIR").expect(
            "set J2K_PROGRESSION_MATRIX_DIR to the directory that receives the codestreams",
        ));
    std::fs::create_dir_all(&dir).unwrap_or_else(|err| panic!("create {}: {err}", dir.display()));
    for (index, row) in rows().into_iter().enumerate() {
        let case = Case::new(row);
        let path = dir.join(format!("{index:02}-{}.j2c", case.name));
        std::fs::write(&path, case.encode().1)
            .unwrap_or_else(|err| panic!("write {}: {err}", path.display()));
    }
}
