// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn reversible_batch_matrix_preserves_native_gray_and_rgb_samples() {
    let mut cases = Vec::new();
    for route in [CodingRoute::Classic, CodingRoute::Htj2k] {
        for components in [1, 3] {
            for signed in [false, true] {
                for precision in [8, 12, 16] {
                    cases.push(encode_case(route, components, precision, signed));
                }
            }
        }
    }
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder = CpuBatchDecoder::new(options);
    let result = decoder
        .decode(
            cases
                .iter()
                .map(|case| EncodedImage::full(Arc::clone(&case.encoded)))
                .collect(),
        )
        .expect("decode native sample matrix");

    assert!(
        result.errors().is_empty(),
        "matrix errors: {:?}",
        result.errors()
    );
    assert_eq!(result.groups().len(), cases.len());
    for (source_index, case) in cases.iter().enumerate() {
        let group = result
            .groups()
            .iter()
            .find(|group| group.source_indices() == [source_index])
            .unwrap_or_else(|| panic!("{}: output group", case.name));
        assert_eq!(group.info().precision, case.precision, "{}", case.name);
        assert_eq!(group.info().signed, case.signed, "{}", case.name);
        assert_eq!(
            group.info().color.channels(),
            case.components,
            "{}",
            case.name
        );
        assert_eq!(group.info().route, case.route, "{}", case.name);
        assert_eq!(
            group.info().transform,
            BatchWaveletTransform::Reversible53,
            "{}",
            case.name
        );
        assert_eq!(
            group.info().sample_type,
            if case.signed {
                NativeSampleType::I16
            } else if case.precision <= 8 {
                NativeSampleType::U8
            } else {
                NativeSampleType::U16
            },
            "{}",
            case.name
        );
        case.oracle.assert_samples(group.samples(), &case.name);
    }
}

#[test]
fn independent_openhtj2k_raw_and_derived_jph_outputs_are_exact_and_indexed() {
    let raw = openhtj2k_refinement_fixture();
    let expected = openhtj2k_refinement_pixels();
    // The codestream and oracle are independent OpenHT/OpenJPH artifacts; only
    // the JPH container around those bytes is constructed by this repository.
    let jph = wrap_j2k_codestream(raw, J2kFileWrapOptions::jph()).expect("wrap OpenHT fixture");
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder = CpuBatchDecoder::new(options);
    let result = decoder
        .decode(vec![
            EncodedImage::full(Arc::from(raw)),
            EncodedImage::full(Arc::from([0_u8, 1, 2, 3])),
            EncodedImage::full(Arc::from(raw)),
            EncodedImage::full(Arc::from(jph)),
        ])
        .expect("decode raw Part 15 and JPH batch");

    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].index, 1);
    assert_eq!(result.groups().len(), 2);
    let raw_group = result
        .groups()
        .iter()
        .find(|group| group.info().payload_kind == CompressedPayloadKind::Jpeg2000Codestream)
        .expect("raw Part 15 group");
    assert_eq!(raw_group.source_indices(), [0, 2]);
    assert_eq!(raw_group.info().route, BatchCodecRoute::Htj2k);
    assert_eq!(
        raw_group.info().transform,
        BatchWaveletTransform::Reversible53
    );
    assert_eq!(raw_group.info().precision, 8);
    assert!(!raw_group.info().signed);
    let CpuBatchSamples::U8(raw_samples) = raw_group.samples() else {
        panic!("OpenHT raw group must retain u8 samples")
    };
    assert_eq!(raw_samples, &[expected, expected].concat());

    let jph_group = result
        .groups()
        .iter()
        .find(|group| group.info().payload_kind == CompressedPayloadKind::JphFile)
        .expect("JPH group");
    assert_eq!(jph_group.source_indices(), [3]);
    let CpuBatchSamples::U8(jph_samples) = jph_group.samples() else {
        panic!("OpenHT JPH group must retain u8 samples")
    };
    assert_eq!(jph_samples, expected);
}

#[test]
fn independent_odd_openhtj2k_fixture_supports_roi_and_reduction() {
    const WIDTH: u32 = 17;
    const HEIGHT: u32 = 37;
    let encoded = Arc::<[u8]>::from(openhtj2k_refinement_odd_fixture());
    let expected = openhtj2k_refinement_odd_pixels();
    let roi = Rect {
        x: 3,
        y: 5,
        w: 9,
        h: 21,
    };
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder = CpuBatchDecoder::new(options);
    let result = decoder
        .decode(vec![
            EncodedImage::full(Arc::clone(&encoded)),
            EncodedImage::new(Arc::clone(&encoded), DecodeRequest::Region { roi }),
            EncodedImage::new(
                Arc::clone(&encoded),
                DecodeRequest::RegionReduced {
                    roi,
                    scale: Downscale::Half,
                },
            ),
        ])
        .expect("decode independent odd OpenHT request matrix");

    assert!(
        result.errors().is_empty(),
        "request errors: {:?}",
        result.errors()
    );
    assert_eq!(result.groups().len(), 3);
    assert!(result
        .groups()
        .iter()
        .all(|group| group.info().route == BatchCodecRoute::Htj2k));

    let full = result
        .groups()
        .iter()
        .find(|group| group.source_indices() == [0])
        .expect("full independent group");
    assert_eq!(
        full.decoded_rects(),
        [Rect {
            x: 0,
            y: 0,
            w: WIDTH,
            h: HEIGHT,
        }]
    );
    let CpuBatchSamples::U8(full_samples) = full.samples() else {
        panic!("independent odd full group must retain u8 samples")
    };
    assert_eq!(full_samples, expected);

    let region = result
        .groups()
        .iter()
        .find(|group| group.source_indices() == [1])
        .expect("independent ROI group");
    assert_eq!(region.decoded_rects(), [roi]);
    let CpuBatchSamples::U8(region_samples) = region.samples() else {
        panic!("independent odd ROI group must retain u8 samples")
    };
    let expected_region = (roi.y..roi.y + roi.h)
        .flat_map(|y| {
            let start = (y * WIDTH + roi.x) as usize;
            expected[start..start + roi.w as usize].iter().copied()
        })
        .collect::<Vec<_>>();
    assert_eq!(region_samples, &expected_region);

    let scaled_roi = roi.scaled_covering(Downscale::Half);
    let mut scalar = J2kDecoder::new(&encoded).expect("scalar odd OpenHT decoder");
    let mut oracle = vec![0_u8; scaled_roi.w as usize * scaled_roi.h as usize];
    let outcome = scalar
        .decode_region_scaled_into(
            &mut J2kScratchPool::new(),
            &mut oracle,
            scaled_roi.w as usize,
            PixelFormat::Gray8,
            roi,
            Downscale::Half,
        )
        .expect("scalar odd OpenHT region-reduced oracle");
    assert_eq!(outcome.decoded, scaled_roi);
    let region_reduced = result
        .groups()
        .iter()
        .find(|group| group.source_indices() == [2])
        .expect("independent region-reduced group");
    assert_eq!(region_reduced.decoded_rects(), [scaled_roi]);
    let CpuBatchSamples::U8(region_reduced_samples) = region_reduced.samples() else {
        panic!("independent odd region-reduced group must retain u8 samples")
    };
    assert_eq!(region_reduced_samples, &oracle);
}

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
