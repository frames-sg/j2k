// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::{
    component_plans, plan_area, window_referenced_classic_plan_to_stores,
    window_referenced_htj2k_plan_to_stores,
};
use crate::{
    encode, encode_htj2k, execute_referenced_classic_plan, execute_referenced_htj2k_plan,
    DecodeSettings, DecoderContext, EncodeOptions, Image, J2kDirectCpuScratch,
    J2kDirectGrayscalePlan, J2kDirectGrayscaleStep, J2kRect, J2kReferencedTilePlan,
};

fn gray_fixture(width: u32, height: u32, ht: bool, reversible: bool) -> Vec<u8> {
    let samples = (0..width * height)
        .map(|index| {
            let (x, y) = (index % width, index / width);
            u8::try_from((x * 5 + y * 3 + (x ^ y) % 37) & 0xff).expect("masked sample")
        })
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible,
        num_decomposition_levels: 4,
        ..EncodeOptions::default()
    };
    let codec = if ht { encode_htj2k } else { encode };
    codec(&samples, width, height, 1, 8, false, &options).expect("encode gray fixture")
}

fn contains(outer: J2kRect, inner: J2kRect) -> bool {
    outer.x0 <= inner.x0 && inner.x1 <= outer.x1 && outer.y0 <= inner.y0 && inner.y1 <= outer.y1
}

/// Every windowed band lies inside its whole-level band, every job inside
/// its windowed band, and every store copies the same samples from inside
/// its windowed input.
fn assert_windows_nest(whole: &J2kDirectGrayscalePlan, windowed: &J2kDirectGrayscalePlan) {
    assert_eq!(whole.steps.len(), windowed.steps.len());
    for (whole, windowed) in whole.steps.iter().zip(&windowed.steps) {
        match (whole, windowed) {
            (
                J2kDirectGrayscaleStep::ClassicSubBand(whole),
                J2kDirectGrayscaleStep::ClassicSubBand(windowed),
            ) => {
                assert!(contains(whole.rect, windowed.rect));
                assert_eq!(whole.jobs.len(), windowed.jobs.len());
                for job in &windowed.jobs {
                    assert!(job.output_x + job.width <= windowed.width);
                    assert!(job.output_y + job.height <= windowed.height);
                    assert_eq!(job.output_stride, windowed.width as usize);
                }
            }
            (
                J2kDirectGrayscaleStep::HtSubBand(whole),
                J2kDirectGrayscaleStep::HtSubBand(windowed),
            ) => {
                assert!(contains(whole.rect, windowed.rect));
                assert_eq!(whole.jobs.len(), windowed.jobs.len());
                for job in &windowed.jobs {
                    assert!(job.output_x + job.width <= windowed.width);
                    assert!(job.output_y + job.height <= windowed.height);
                    assert_eq!(job.output_stride, windowed.width as usize);
                }
            }
            (J2kDirectGrayscaleStep::Idwt(whole), J2kDirectGrayscaleStep::Idwt(windowed)) => {
                assert!(contains(whole.rect, windowed.rect));
            }
            (J2kDirectGrayscaleStep::Store(whole), J2kDirectGrayscaleStep::Store(windowed)) => {
                assert_eq!(
                    (
                        whole.copy_width,
                        whole.copy_height,
                        whole.output_x,
                        whole.output_y
                    ),
                    (
                        windowed.copy_width,
                        windowed.copy_height,
                        windowed.output_x,
                        windowed.output_y
                    )
                );
                assert_eq!(
                    (
                        whole.input_rect.x0 + whole.source_x,
                        whole.input_rect.y0 + whole.source_y
                    ),
                    (
                        windowed.input_rect.x0 + windowed.source_x,
                        windowed.input_rect.y0 + windowed.source_y
                    )
                );
                assert!(windowed.source_x + windowed.copy_width <= windowed.input_rect.width());
                assert!(windowed.source_y + windowed.copy_height <= windowed.input_rect.height());
            }
            (whole, windowed) => panic!("step kinds differ: {whole:?} vs {windowed:?}"),
        }
    }
}

/// Check nesting and return the whole-level and windowed coefficient areas.
fn checked_areas(
    whole: &[J2kReferencedTilePlan],
    windowed: &[J2kReferencedTilePlan],
) -> (u64, u64) {
    assert_eq!(whole.len(), windowed.len());
    let (mut whole_area, mut windowed_area) = (0, 0);
    for (whole, windowed) in whole.iter().zip(windowed) {
        for (whole, windowed) in component_plans(whole).zip(component_plans(windowed)) {
            assert_windows_nest(whole, windowed);
            whole_area += plan_area(whole);
            windowed_area += plan_area(windowed);
        }
    }
    (whole_area, windowed_area)
}

#[test]
fn small_regions_decode_identically_from_windowed_bands() {
    let region = (301, 207, 23, 17);
    for reversible in [true, false] {
        let case = format!("reversible={reversible}");
        let bytes = gray_fixture(512, 384, true, reversible);
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("HTJ2K image");
        let whole = image
            .build_referenced_htj2k_plan_region_with_context(&mut DecoderContext::default(), region)
            .expect("HTJ2K region plan");
        let windowed = window_referenced_htj2k_plan_to_stores(&whole)
            .expect("window HTJ2K plan")
            .expect("a 23 x 17 region windows a 512 x 384 plan");
        let (whole_area, windowed_area) = checked_areas(whole.tiles(), windowed.tiles());
        assert!(
            windowed_area * 2 <= whole_area,
            "{case}: {windowed_area} of {whole_area}"
        );

        let (mut whole_scratch, mut windowed_scratch) =
            (J2kDirectCpuScratch::new(), J2kDirectCpuScratch::new());
        let expected = execute_referenced_htj2k_plan(&whole, &bytes, false, &mut whole_scratch)
            .expect("execute whole-level plan");
        let actual = execute_referenced_htj2k_plan(&windowed, &bytes, false, &mut windowed_scratch)
            .expect("execute windowed plan");
        assert_eq!(expected.dimensions(), actual.dimensions(), "{case}");
        assert_eq!(
            expected.plane(0).expect("whole plane").samples(),
            actual.plane(0).expect("windowed plane").samples(),
            "{case}"
        );

        let bytes = gray_fixture(512, 384, false, reversible);
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("classic image");
        let whole = image
            .build_referenced_classic_plan_region_with_context(
                &mut DecoderContext::default(),
                region,
            )
            .expect("classic region plan");
        let windowed = window_referenced_classic_plan_to_stores(&whole)
            .expect("window classic plan")
            .expect("a 23 x 17 region windows a 512 x 384 plan");
        let (whole_area, windowed_area) = checked_areas(whole.tiles(), windowed.tiles());
        assert!(
            windowed_area * 2 <= whole_area,
            "{case}: {windowed_area} of {whole_area}"
        );

        let expected = execute_referenced_classic_plan(&whole, &bytes, false, &mut whole_scratch)
            .expect("execute whole-level classic plan");
        let actual =
            execute_referenced_classic_plan(&windowed, &bytes, false, &mut windowed_scratch)
                .expect("execute windowed classic plan");
        assert_eq!(expected.dimensions(), actual.dimensions(), "{case}");
        assert_eq!(
            expected.plane(0).expect("whole plane").samples(),
            actual.plane(0).expect("windowed plane").samples(),
            "{case}"
        );
    }
}

#[test]
fn full_plans_and_regions_covering_most_of_a_small_image_stay_whole() {
    let bytes = gray_fixture(240, 240, true, false);
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("HTJ2K image");
    for region in [(0, 0, 240, 240), (20, 20, 200, 200)] {
        let whole = image
            .build_referenced_htj2k_plan_region_with_context(&mut DecoderContext::default(), region)
            .expect("HTJ2K region plan");
        assert!(
            window_referenced_htj2k_plan_to_stores(&whole)
                .expect("window HTJ2K plan")
                .is_none(),
            "region {region:?} must keep the shared plan"
        );
    }
}
