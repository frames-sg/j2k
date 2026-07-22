// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashSet;

use j2k_native::{
    execute_referenced_htj2k_plan, prepare_referenced_htj2k_staged, DecodeSettings, DecoderContext,
    Image, J2kDirectCpuEntropyWorkspace, J2kDirectCpuScratch, J2kDirectGrayscaleStep, J2kRect,
};

const OPENJPH_GRAY_U12_53: &[u8] = include_bytes!("../fixtures/htj2k/gray_u12_53.j2c");
const OPENJPH_GRAY_U12_53_ORACLE: &[u8] =
    include_bytes!("../fixtures/htj2k/gray_u12_53.oracle.raw");
const OPENJPH_RGB_U12_53: &[u8] = include_bytes!("../fixtures/htj2k/rgb_u12_53.j2c");
const OPENJPH_RGB_U12_53_ORACLE: &[u8] = include_bytes!("../fixtures/htj2k/rgb_u12_53.oracle.raw");

#[test]
fn referenced_htj2k_grayscale_executes_all_tiles_bit_exactly() {
    let image = Image::new(OPENJPH_GRAY_U12_53, &DecodeSettings::strict())
        .expect("parse independent OpenJPH fixture");
    let mut context = DecoderContext::default();
    let plan = image
        .build_referenced_htj2k_plan_region_with_context(&mut context, (0, 0, 19, 13))
        .expect("build referenced multi-tile HTJ2K plan");
    let mut scratch = J2kDirectCpuScratch::new();
    let decoded = execute_referenced_htj2k_plan(&plan, OPENJPH_GRAY_U12_53, false, &mut scratch)
        .expect("execute every referenced tile");

    assert_eq!(decoded.dimensions(), (19, 13));
    assert_eq!(decoded.component_count(), 1);
    let actual = decoded.plane(0).expect("grayscale plane").samples();
    let expected = OPENJPH_GRAY_U12_53_ORACLE
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]));
    assert_eq!(actual.len(), 19 * 13);
    for (index, (actual, expected)) in actual.iter().copied().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            f32::from(expected).to_bits(),
            "sample {index}"
        );
    }
}

#[test]
fn referenced_htj2k_rgb_executes_all_tiles_bit_exactly() {
    let image = Image::new(OPENJPH_RGB_U12_53, &DecodeSettings::strict())
        .expect("parse independent OpenJPH fixture");
    let mut context = DecoderContext::default();
    let plan = image
        .build_referenced_htj2k_plan_region_with_context(&mut context, (0, 0, 19, 13))
        .expect("build referenced multi-tile RGB HTJ2K plan");
    let mut scratch = J2kDirectCpuScratch::new();
    let decoded = execute_referenced_htj2k_plan(&plan, OPENJPH_RGB_U12_53, false, &mut scratch)
        .expect("execute every referenced RGB tile");

    assert_eq!(decoded.dimensions(), (19, 13));
    assert_eq!(decoded.component_count(), 3);
    let planes = [
        decoded.plane(0).expect("red plane"),
        decoded.plane(1).expect("green plane"),
        decoded.plane(2).expect("blue plane"),
    ];
    let expected: Vec<_> = OPENJPH_RGB_U12_53_ORACLE
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect();
    for pixel in 0..19 * 13 {
        for component in 0..3 {
            assert_eq!(
                planes[component].samples()[pixel].to_bits(),
                f32::from(expected[pixel * 3 + component]).to_bits(),
                "pixel {pixel}, component {component}"
            );
        }
    }
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the structural regression validates tile order, payload spans, globally unique bands, stores, and bounded staged scratch together"
)]
fn referenced_htj2k_grayscale_plan_preserves_all_odd_edge_tiles() {
    let image = Image::new(OPENJPH_GRAY_U12_53, &DecodeSettings::strict())
        .expect("parse independent OpenJPH fixture");
    let mut context = DecoderContext::default();
    let plan = image
        .build_referenced_htj2k_plan_region_with_context(&mut context, (0, 0, 19, 13))
        .expect("build referenced multi-tile HTJ2K plan");

    assert_eq!(plan.full_dimensions(), (19, 13));
    assert_eq!(
        plan.output_rect(),
        J2kRect {
            x0: 0,
            y0: 0,
            x1: 19,
            y1: 13,
        }
    );
    assert!(
        plan.grayscale_geometry().is_none(),
        "the legacy single-tile accessor must fail closed for multi-tile plans"
    );

    let tiles = plan.tiles();
    assert_eq!(tiles.len(), 4);
    let expected_rects = [
        J2kRect {
            x0: 0,
            y0: 0,
            x1: 11,
            y1: 7,
        },
        J2kRect {
            x0: 11,
            y0: 0,
            x1: 19,
            y1: 7,
        },
        J2kRect {
            x0: 0,
            y0: 7,
            x1: 11,
            y1: 13,
        },
        J2kRect {
            x0: 11,
            y0: 7,
            x1: 19,
            y1: 13,
        },
    ];

    let mut produced_band_ids = HashSet::new();
    let mut next_payload_record = 0;
    let mut maximum_live_band_owners = 0usize;
    let mut all_tile_band_owners = 0usize;
    for (expected_index, (tile, expected_rect)) in tiles.iter().zip(expected_rects).enumerate() {
        assert_eq!(tile.tile_index(), expected_index);
        assert_eq!(tile.decoded_rect(), expected_rect);
        assert_eq!(tile.destination_rect(), expected_rect);

        let span = tile.payload_records();
        assert_eq!(span.first_record, next_payload_record);
        next_payload_record = span.end_record().expect("payload span does not overflow");

        let geometry = tile
            .grayscale_geometry()
            .expect("grayscale tile has grayscale geometry");
        assert_eq!(geometry.dimensions, (19, 13));
        let mut store_count = 0;
        let mut job_count = 0;
        let mut tile_band_owners = 0usize;
        for step in &geometry.steps {
            match step {
                J2kDirectGrayscaleStep::HtSubBand(sub_band) => {
                    tile_band_owners += 1;
                    assert!(
                        produced_band_ids.insert(sub_band.band_id),
                        "sub-band IDs must be globally unique across tiles"
                    );
                    job_count += sub_band.jobs.len();
                }
                J2kDirectGrayscaleStep::Idwt(idwt) => {
                    tile_band_owners += 1;
                    assert!(
                        produced_band_ids.insert(idwt.output_band_id),
                        "IDWT output IDs must be globally unique across tiles"
                    );
                }
                J2kDirectGrayscaleStep::Store(store) => {
                    store_count += 1;
                    assert_eq!((store.output_width, store.output_height), (19, 13));
                    assert_eq!(
                        (store.output_x, store.output_y),
                        (expected_rect.x0, expected_rect.y0)
                    );
                    assert_eq!(
                        (store.copy_width, store.copy_height),
                        (expected_rect.width(), expected_rect.height())
                    );
                }
                J2kDirectGrayscaleStep::ClassicSubBand(_) => {
                    panic!("HT plan contains classic entropy work")
                }
            }
        }
        assert_eq!(store_count, 1);
        assert_eq!(job_count, span.record_count);
        maximum_live_band_owners = maximum_live_band_owners.max(tile_band_owners);
        all_tile_band_owners += tile_band_owners;
    }

    assert_eq!(next_payload_record, plan.payloads().len());
    let mut scratch = J2kDirectCpuScratch::new();
    prepare_referenced_htj2k_staged(
        &plan,
        &mut scratch,
        &mut J2kDirectCpuEntropyWorkspace::default(),
    )
    .expect("prepare bounded staged multi-tile scratch");
    assert_eq!(
        scratch.retained_band_owner_count(),
        maximum_live_band_owners,
        "live coefficient owners must be bounded by one tile"
    );
    assert!(
        scratch.retained_band_owner_count() < all_tile_band_owners,
        "staged scratch must not retain one band set per tile"
    );
}

#[test]
fn referenced_htj2k_reduced_roi_stores_are_disjoint_and_cover_the_destination() {
    let settings = DecodeSettings {
        target_resolution: Some((10, 7)),
        ..DecodeSettings::strict()
    };
    let image = Image::new(OPENJPH_GRAY_U12_53, &settings)
        .expect("parse reduced independent OpenJPH fixture");
    assert_eq!((image.width(), image.height()), (10, 7));
    let mut context = DecoderContext::default();
    let plan = image
        .build_referenced_htj2k_plan_region_with_context(&mut context, (2, 1, 7, 5))
        .expect("build reduced ROI multi-tile plan");

    assert_eq!(plan.full_dimensions(), (10, 7));
    assert_eq!(
        plan.output_rect(),
        J2kRect {
            x0: 2,
            y0: 1,
            x1: 9,
            y1: 6,
        }
    );
    let mut coverage = [false; 7 * 5];
    let mut next_payload_record = 0;
    for tile in plan.tiles() {
        let span = tile.payload_records();
        assert_eq!(span.first_record, next_payload_record);
        next_payload_record = span.end_record().expect("payload span does not overflow");

        let destination = tile.destination_rect();
        let decoded = tile.decoded_rect();
        assert_eq!(decoded.x0, destination.x0 + 2);
        assert_eq!(decoded.y0, destination.y0 + 1);
        assert_eq!(decoded.x1, destination.x1 + 2);
        assert_eq!(decoded.y1, destination.y1 + 1);
        let geometry = tile.grayscale_geometry().expect("grayscale tile geometry");
        let store = geometry
            .steps
            .iter()
            .find_map(|step| match step {
                J2kDirectGrayscaleStep::Store(store) => Some(store),
                _ => None,
            })
            .expect("one tile store");
        assert_eq!((store.output_width, store.output_height), (7, 5));
        assert_eq!(
            (store.output_x, store.output_y),
            (destination.x0, destination.y0)
        );
        assert_eq!(
            (store.copy_width, store.copy_height),
            (destination.width(), destination.height())
        );

        for y in destination.y0..destination.y1 {
            for x in destination.x0..destination.x1 {
                let index = y as usize * 7 + x as usize;
                assert!(!coverage[index], "tile stores overlap at ({x}, {y})");
                coverage[index] = true;
            }
        }
    }
    assert_eq!(next_payload_record, plan.payloads().len());
    assert!(coverage.into_iter().all(|covered| covered));
}
