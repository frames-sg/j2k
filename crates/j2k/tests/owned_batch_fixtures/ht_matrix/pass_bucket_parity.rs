// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, sync::Arc};

use j2k::{
    prepare_batch, BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples, EncodedImage,
    J2kDecoder, PixelFormat,
};
use j2k_native::{encode_htj2k, EncodeOptions, J2kDirectGrayscalePlan, J2kDirectGrayscaleStep};
use j2k_test_support::{
    openhtj2k_refinement_fixture, openhtj2k_refinement_odd_fixture,
    openhtj2k_refinement_odd_pixels, openhtj2k_refinement_pixels, openhtj2k_sigprop_fixture,
    openhtj2k_sigprop_overlap_fixture, openhtj2k_sigprop_overlap_pixels,
    openhtj2k_sigprop_pixels_le,
};

fn ht_pass_counts(prepared: &j2k::PreparedImage) -> Vec<u8> {
    let plan = prepared.htj2k_plan().expect("HTJ2K referenced plan");
    let plan = plan
        .adapter_view()
        .downcast_ref::<j2k_native::J2kReferencedHtj2kPlan>()
        .expect("j2k-native prepared-plan adapter");
    let mut passes = Vec::new();
    for tile in plan.tiles() {
        if let Some(geometry) = tile.grayscale_geometry() {
            extend_ht_pass_counts(geometry, &mut passes);
        } else if let Some(geometry) = tile.color_geometry() {
            for component in &geometry.component_plans {
                extend_ht_pass_counts(component, &mut passes);
            }
        } else if let Some(geometry) = tile.rgba_geometry() {
            for component in &geometry.component_plans {
                extend_ht_pass_counts(component, &mut passes);
            }
        }
    }
    passes
}

fn extend_ht_pass_counts(plan: &J2kDirectGrayscalePlan, passes: &mut Vec<u8>) {
    passes.extend(
        plan.steps
            .iter()
            .filter_map(|step| match step {
                J2kDirectGrayscaleStep::HtSubBand(sub_band) => Some(&sub_band.jobs),
                _ => None,
            })
            .flatten()
            .map(|job| job.number_of_coding_passes),
    );
}

fn layered_ht_fixture(num_layers: u8) -> (Arc<[u8]>, Vec<u8>) {
    const WIDTH: u32 = 8;
    const HEIGHT: u32 = 8;
    let pixels = (0..WIDTH * HEIGHT)
        .map(|index| ((index * 73 + index / 7) & 0xff) as u8)
        .collect::<Vec<_>>();
    let encoded = encode_htj2k(
        &pixels,
        WIDTH,
        HEIGHT,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: false,
            num_decomposition_levels: 0,
            guard_bits: 2,
            num_layers,
            use_mct: false,
            ..EncodeOptions::default()
        },
    )
    .expect("encode layered HT fixture");
    let mut scalar = J2kDecoder::new(&encoded).expect("scalar layered decoder");
    let mut oracle = vec![0_u8; pixels.len()];
    scalar
        .decode_into(&mut oracle, WIDTH as usize, PixelFormat::Gray8)
        .expect("scalar layered oracle");
    (Arc::from(encoded), oracle)
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one end-to-end matrix keeps pass classification and independent decoder oracles together"
)]
fn external_cleanup_magref_and_generated_sigprop_jobs_decode_in_batches() {
    let (cleanup, cleanup_pixels) = layered_ht_fixture(1);
    let (sigprop, sigprop_pixels) = layered_ht_fixture(2);
    let (magref, magref_pixels) = layered_ht_fixture(3);
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let prepared = prepare_batch(
        vec![
            EncodedImage::full(cleanup),
            EncodedImage::full(sigprop),
            EncodedImage::full(magref),
            EncodedImage::full(Arc::from(openhtj2k_refinement_fixture())),
            EncodedImage::full(Arc::from(openhtj2k_refinement_odd_fixture())),
            EncodedImage::full(Arc::from(openhtj2k_sigprop_fixture())),
            EncodedImage::full(Arc::from(openhtj2k_sigprop_overlap_fixture())),
        ],
        options,
    )
    .expect("prepare HT pass matrix");

    assert!(
        prepared.errors().is_empty(),
        "prepare errors: {:?}",
        prepared.errors()
    );
    let generated_group = prepared
        .groups()
        .iter()
        .find(|group| group.source_indices() == [0, 1, 2])
        .expect("generated pass group");
    let pass_sets = generated_group
        .images()
        .iter()
        .map(|image| ht_pass_counts(image).into_iter().collect::<BTreeSet<_>>())
        .collect::<Vec<_>>();
    assert_eq!(pass_sets[0], BTreeSet::from([1]));
    assert!(
        pass_sets[1].contains(&2),
        "two-layer fixture: {:?}",
        pass_sets[1]
    );
    assert!(
        pass_sets[2].contains(&3),
        "three-layer fixture: {:?}",
        pass_sets[2]
    );

    let independent_passes = prepared
        .groups()
        .iter()
        .filter(|group| group.source_indices().iter().any(|index| *index >= 3))
        .flat_map(j2k::PreparedBatchGroup::images)
        .flat_map(ht_pass_counts)
        .collect::<Vec<_>>();
    assert!(
        independent_passes.contains(&1),
        "independent OpenHT fixtures must exercise cleanup-only blocks"
    );
    assert!(
        independent_passes.contains(&2),
        "independent OpenHT fixtures must exercise exactly-two-pass SigProp blocks"
    );
    assert!(
        independent_passes.contains(&3),
        "independent OpenHT fixtures must exercise three-pass refinement"
    );

    let mut decoder = CpuBatchDecoder::new(options);
    let result = decoder
        .decode_prepared(&prepared)
        .expect("decode prepared HT pass matrix");
    assert!(
        result.errors().is_empty(),
        "decode errors: {:?}",
        result.errors()
    );
    let stats = decoder.workspace_stats();
    assert!(stats.flattened_group_plans() > 0);
    assert_eq!(
        stats.flattened_payload_jobs(),
        stats.flattened_cleanup_jobs()
            + stats.flattened_sigprop_jobs()
            + stats.flattened_magref_jobs()
    );
    assert!(stats.flattened_cleanup_jobs() > 0);
    assert!(stats.flattened_sigprop_jobs() > 0);
    assert!(stats.flattened_magref_jobs() > 0);
    assert_eq!(stats.flattened_classic_jobs(), 0);
    let generated = result
        .groups()
        .iter()
        .find(|group| group.source_indices() == [0, 1, 2])
        .expect("decoded generated pass group");
    let CpuBatchSamples::U8(samples) = generated.samples() else {
        panic!("generated HT pass group must retain u8 samples")
    };
    assert_eq!(
        samples,
        &[cleanup_pixels, sigprop_pixels, magref_pixels].concat()
    );

    for (source_index, expected) in [
        (3, openhtj2k_refinement_pixels()),
        (4, openhtj2k_refinement_odd_pixels()),
    ] {
        let group = result
            .groups()
            .iter()
            .find(|group| group.source_indices() == [source_index])
            .expect("independent OpenHT output group");
        let CpuBatchSamples::U8(samples) = group.samples() else {
            panic!("independent OpenHT fixture must retain u8 samples")
        };
        assert_eq!(samples, expected);
    }

    let sigprop_group = result
        .groups()
        .iter()
        .find(|group| group.source_indices() == [5])
        .expect("independent OpenHT SigProp output group");
    let CpuBatchSamples::U16(sigprop_samples) = sigprop_group.samples() else {
        panic!("independent OpenHT RGB12 fixture must retain u16 samples")
    };
    let expected_sigprop = openhtj2k_sigprop_pixels_le()
        .chunks_exact(2)
        .map(|sample| u16::from_le_bytes([sample[0], sample[1]]))
        .collect::<Vec<_>>();
    assert_eq!(sigprop_samples.len(), expected_sigprop.len());
    let mut scalar_decoder =
        J2kDecoder::new(openhtj2k_sigprop_fixture()).expect("independent OpenHT scalar decoder");
    let mut scalar_bytes = vec![0_u8; 128 * 128 * 3 * 2];
    scalar_decoder
        .decode_into(&mut scalar_bytes, 128 * 3 * 2, PixelFormat::Rgb16)
        .expect("independent OpenHT scalar RGB16 decode");
    let scalar_samples = scalar_bytes
        .chunks_exact(2)
        .map(|sample| u16::from_le_bytes([sample[0], sample[1]]))
        .collect::<Vec<_>>();
    let (scalar_difference_index, scalar_difference) = sigprop_samples
        .iter()
        .zip(&scalar_samples)
        .enumerate()
        .map(|(index, (actual, expected))| (index, actual.abs_diff(*expected)))
        .max_by_key(|(_, difference)| *difference)
        .unwrap_or_default();
    assert!(
        scalar_difference <= 1,
        "prepared decode differs from scalar by {scalar_difference} LSBs at sample {scalar_difference_index}: prepared={}, scalar={}",
        sigprop_samples[scalar_difference_index], scalar_samples[scalar_difference_index]
    );
    let (max_difference_index, max_difference) = sigprop_samples
        .iter()
        .zip(&expected_sigprop)
        .enumerate()
        .map(|(index, (actual, expected))| (index, actual.abs_diff(*expected)))
        .max_by_key(|(_, difference)| *difference)
        .unwrap_or_default();
    assert!(
        max_difference <= 1,
        "independent irreversible RGB12 reconstruction differs from OpenJPH by {max_difference} LSBs at sample {max_difference_index}: codec={}, OpenJPH={}",
        sigprop_samples[max_difference_index],
        expected_sigprop[max_difference_index]
    );

    let overlap_group = result
        .groups()
        .iter()
        .find(|group| group.source_indices() == [6])
        .expect("independent overlapping-refinement output group");
    let CpuBatchSamples::U8(overlap_samples) = overlap_group.samples() else {
        panic!("independent overlapping-refinement fixture must retain u8 samples")
    };
    let expected_overlap = openhtj2k_sigprop_overlap_pixels();
    assert_eq!(overlap_samples.len(), expected_overlap.len());
    let (overlap_difference_index, overlap_difference) = overlap_samples
        .iter()
        .zip(expected_overlap)
        .enumerate()
        .map(|(index, (actual, expected))| (index, actual.abs_diff(*expected)))
        .max_by_key(|(_, difference)| *difference)
        .unwrap_or_default();
    assert!(
        overlap_difference <= 1,
        "overlapping SigProp/MagRef reconstruction differs from OpenHTJ2K by {overlap_difference} LSBs at sample {overlap_difference_index}: codec={}, OpenHTJ2K={}",
        overlap_samples[overlap_difference_index],
        expected_overlap[overlap_difference_index]
    );
}
