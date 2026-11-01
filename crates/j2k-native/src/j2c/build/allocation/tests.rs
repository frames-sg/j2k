// SPDX-License-Identifier: MIT OR Apache-2.0

use super::plan::{tag_tree_node_count, IdwtWorkspaceTracker};
use super::reuse::{discard_stale_capacity, ReallocationBudget};
use super::{
    release_unused_roi_workspace, roi_workspace_bytes, BuildWorkspace, DecompositionAllocationPlan,
    RecordCounts, Segment, DEFAULT_MAX_DECODE_BYTES,
};
use crate::error::{DecodeError, ValidationError};
use crate::j2c::build::{Decomposition, SubBand, SubBandType};
use crate::j2c::codestream::WaveletTransform;
use crate::j2c::decode::{DecodeAllocationBudget, DecompositionStorage};
use crate::j2c::rect::IntRect;
use crate::j2c::tag_tree::TagTree;
use alloc::vec::Vec;
use core::mem::size_of;

fn coefficient_plan(baseline: usize, coefficients: usize) -> DecompositionAllocationPlan {
    let mut plan =
        DecompositionAllocationPlan::new(baseline, false, false, BuildWorkspace::CoefficientsOnly)
            .expect("baseline");
    plan.add_component_coefficients(coefficients)
        .expect("target count");
    plan
}

#[test]
fn logical_record_bytes_are_independent_of_the_decode_cap() {
    let count = DEFAULT_MAX_DECODE_BYTES / size_of::<SubBand>() + 1;
    let bytes = RecordCounts {
        sub_bands: count,
        ..RecordCounts::default()
    }
    .checked_bytes()
    .expect("record byte count fits usize");
    assert_eq!(bytes, count * size_of::<SubBand>());
    assert!(bytes > DEFAULT_MAX_DECODE_BYTES);
}

#[test]
fn metadata_plan_retains_logical_samples_without_allocating_pixel_storage() {
    let mut storage = DecompositionStorage {
        exact_integer_decode: true,
        ..DecompositionStorage::default()
    };
    storage.coefficients.try_reserve_exact(8).unwrap();
    storage.coefficients_i64.try_reserve_exact(8).unwrap();
    let baseline = DEFAULT_MAX_DECODE_BYTES - size_of::<SubBand>();
    let mut plan =
        DecompositionAllocationPlan::new(baseline, true, false, BuildWorkspace::MetadataOnly)
            .unwrap();
    plan.sub_bands = 1;
    let logical_samples = DEFAULT_MAX_DECODE_BYTES / size_of::<f32>() + 1;
    plan.add_component_coefficients(logical_samples).unwrap();
    plan.validate_minimum_live_workspace(0)
        .expect("metadata fits without pixel storage");
    discard_stale_capacity(&mut storage, &plan);
    super::reuse::reserve_decomposition_storage(&plan, &mut storage, baseline)
        .expect("only metadata is reserved");
    plan.account_live_workspace(&storage, 0).unwrap();
    assert_eq!(plan.coefficients, logical_samples);
    assert_eq!(plan.total_bytes, DEFAULT_MAX_DECODE_BYTES);
    assert_eq!(storage.coefficients.capacity(), 0);
    assert_eq!(storage.coefficients_i64.capacity(), 0);
    assert_eq!(storage.sub_bands.capacity(), 1);
}

#[test]
fn allocation_plan_tag_tree_count_matches_builder_for_edge_shapes() {
    for (width, height) in [
        (0, 0),
        (0, 4),
        (4, 0),
        (1, 1),
        (1, 3),
        (3, 1),
        (2, 2),
        (3, 5),
        (8, 8),
    ] {
        let mut nodes = Vec::new();
        let _tree = TagTree::new(width, height, &mut nodes);
        assert_eq!(
            nodes.len(),
            tag_tree_node_count(width, height).unwrap(),
            "{width}x{height} tag tree"
        );
    }
}

#[test]
fn same_size_second_tile_reuses_capacity_without_charging_a_second_owner() {
    let mut storage = DecompositionStorage::default();
    storage
        .coefficients
        .try_reserve_exact(2)
        .expect("small test capacity");
    let target = storage.coefficients.capacity();
    let coefficient_bytes = target * size_of::<f32>();
    let baseline = DEFAULT_MAX_DECODE_BYTES - coefficient_bytes;
    let mut plan = coefficient_plan(baseline, target);

    discard_stale_capacity(&mut storage, &plan);
    assert_eq!(storage.coefficients.capacity(), target);
    plan.account_live_workspace(&storage, 0)
        .expect("same-size reuse fits exactly");
    assert_eq!(plan.total_bytes, DEFAULT_MAX_DECODE_BYTES);

    let mut reallocation =
        ReallocationBudget::for_storage(&storage, baseline).expect("live baseline");
    reallocation
        .reserve(&mut storage.coefficients, target)
        .expect("same target is allocation-free");
    assert_eq!(storage.coefficients.capacity(), target);
}

#[test]
fn stale_capacity_is_released_even_when_only_a_later_phase_would_need_the_space() {
    let mut storage = DecompositionStorage::default();
    storage
        .coefficients
        .try_reserve_exact(4)
        .expect("small test capacity");
    storage
        .segments
        .try_reserve_exact(2)
        .expect("small segment capacity");
    let baseline = DEFAULT_MAX_DECODE_BYTES - 2 * size_of::<f32>();
    let mut plan = coefficient_plan(baseline, 1);

    // The logical build has room without consulting a failure fallback, but
    // keeping either stale owner would steal capacity from packet/decode work.
    discard_stale_capacity(&mut storage, &plan);
    assert_eq!(storage.coefficients.capacity(), 0);
    assert_eq!(storage.segments.capacity(), 0);
    plan.account_live_workspace(&storage, 0)
        .expect("logical request remains admissible");
}

#[test]
fn retained_segment_capacity_is_added_once_outside_the_structural_total() {
    let mut storage = DecompositionStorage::default();
    storage
        .segments
        .try_reserve_exact(2)
        .expect("small test capacity");
    let segment_bytes = storage.segments.capacity() * size_of::<Segment<'_>>();
    let baseline = DEFAULT_MAX_DECODE_BYTES - segment_bytes;
    let mut plan = coefficient_plan(baseline, 0);

    plan.account_live_workspace(&storage, 0)
        .expect("segment capacity fits once");
    assert_eq!(plan.total_bytes, baseline);
    storage.structural_workspace_bytes = plan.total_bytes;
    DecodeAllocationBudget::for_storage(&storage)
        .expect("downstream adds the separate segment owner once");
}

#[test]
fn roi_release_restores_the_exact_reserved_boundary() {
    let sub_bands = 7;
    let decompositions = 3;
    let components = 2;
    let bytes = roi_workspace_bytes(sub_bands, decompositions, components).expect("ROI bytes");
    let mut storage = DecompositionStorage::default();
    let empty = IntRect::from_xywh(0, 0, 0, 0);
    for _ in 0..sub_bands {
        storage.sub_bands.push(SubBand {
            sub_band_type: SubBandType::LowLow,
            rect: empty,
            precincts: 0..0,
            coefficients: 0..0,
            storage_rect: empty,
        });
    }
    for _ in 0..decompositions {
        storage.decompositions.push(Decomposition {
            sub_bands: [0; 3],
            rect: empty,
        });
    }
    storage.structural_workspace_bytes = DEFAULT_MAX_DECODE_BYTES;
    release_unused_roi_workspace(&mut storage, components).expect("planned ROI bytes were present");
    assert_eq!(
        storage.structural_workspace_bytes,
        DEFAULT_MAX_DECODE_BYTES - bytes
    );
    let mut budget = DecodeAllocationBudget::from_live_bytes(storage.structural_workspace_bytes)
        .expect("released baseline");
    budget
        .include_elements::<u8>(bytes)
        .expect("exact released boundary is reusable");
    assert!(budget.include_elements::<u8>(1).is_err());
}

#[test]
fn skipped_resolution_idwt_workspace_uses_padded_active_high_water() {
    let ll = IntRect::from_xywh(0, 0, 2, 2);
    let rects = [
        IntRect::from_xywh(0, 0, 4, 3),
        IntRect::from_xywh(0, 0, 8, 5),
        IntRect::from_xywh(0, 0, 16, 9),
    ];

    let mut skipped_one = IdwtWorkspaceTracker::new(Some(2), WaveletTransform::Irreversible97);
    for rect in rects {
        skipped_one.observe(rect);
    }
    assert_eq!(skipped_one.finish(ll).unwrap(), Some(54 + 20));

    let mut full = IdwtWorkspaceTracker::new(Some(3), WaveletTransform::Irreversible97);
    for rect in rects {
        full.observe(rect);
    }
    assert_eq!(full.finish(ll).unwrap(), Some(170 + 54));

    let mut ll_only = IdwtWorkspaceTracker::new(Some(0), WaveletTransform::Irreversible97);
    for rect in rects {
        ll_only.observe(rect);
    }
    assert_eq!(ll_only.finish(ll).unwrap(), Some(4));
}

/// A one-component image of noise, `levels` decompositions deep.
#[cfg(feature = "parallel")]
fn gray_noise_codestream(width: u32, height: u32, levels: u8, reversible: bool) -> Vec<u8> {
    let mut state = 0x9e37_79b9_u32;
    let pixels: Vec<u8> = (0..width * height)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state.to_le_bytes()[3]
        })
        .collect();
    let options = crate::EncodeOptions {
        reversible,
        num_decomposition_levels: levels,
        ..crate::EncodeOptions::default()
    };
    crate::encode(&pixels, width, height, 1, 8, false, &options).expect("encode fixture")
}

/// Images whose two finest IDWT levels `idwt::apply` reconstructs as one
/// pass on a pool of 2, 4 or 8 threads, with and without a level below the
/// pair, for both transforms.
#[cfg(feature = "parallel")]
const PAIRED_IDWT_IMAGES: [(u32, u32, u8, bool); 4] = [
    (512, 512, 2, false),
    (600, 700, 3, true),
    (1024, 768, 4, false),
    (777, 1100, 3, false),
];

/// The paired pass writes only the finer level, and each worker streams
/// the coarser level's rows it reads into its band scratch: what the IDWT
/// holds is the finest level's output, that band scratch, and a ping-pong
/// buffer for the level below the pair alone. The plan must charge all of
/// it, and the output is the serial level-by-level decode's.
#[cfg(feature = "parallel")]
#[test]
fn paired_idwt_levels_hold_no_more_than_the_plan_charges() {
    use crate::j2c::build::CoefficientLayout;
    use crate::j2c::idwt::idwt_buffer_size;
    use crate::{CpuDecodeParallelism, DecodeSettings, DecoderContext, Image};

    for (width, height, levels, reversible) in PAIRED_IDWT_IMAGES {
        let bytes = gray_noise_codestream(width, height, levels, reversible);
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse fixture");
        let mut serial_context = DecoderContext::default();
        serial_context.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
        let serial = image
            .decode_with_context(&mut serial_context)
            .expect("serial decode")
            .data;
        for threads in [2, 4, 8] {
            let label = format!("{width}x{height}, {levels} levels, {threads} threads");
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool");
            pool.install(|| {
                let tiles = crate::j2c::tile::parse(
                    &mut crate::reader::BitReader::new(image.codestream),
                    &image.header,
                    0,
                )
                .expect("parse tile");
                let (_, charged) = super::plan::build_allocation_plan(
                    &tiles[0],
                    0,
                    false,
                    false,
                    BuildWorkspace::DecodePixels {
                        skipped_resolution_levels: 0,
                        coefficient_layout: CoefficientLayout::Component,
                    },
                    None,
                    None,
                )
                .expect("plan");
                let level_rects: Vec<IntRect> = tiles[0]
                    .component_tiles()
                    .next()
                    .expect("component tile")
                    .resolution_tiles()
                    .map(|resolution| resolution.rect)
                    .collect();
                let mut context = DecoderContext::default();
                let paired = image
                    .decode_with_context(&mut context)
                    .expect("paired decode")
                    .data;
                assert_eq!(paired, serial, "{label}");
                let tile_ctx = &context.tile_decode_context;
                let held = tile_ctx.idwt_capacity_bytes().expect("IDWT bytes");
                assert!(
                    held <= charged * size_of::<f32>(),
                    "{label}: the IDWT holds {held} bytes, the plan charged {}",
                    charged * size_of::<f32>()
                );
                // Only the levels below the pair go through the ping-pong
                // buffer, and the last of them is the third-finest level.
                let below_pair = usize::from(levels)
                    .checked_sub(2)
                    .filter(|&index| index > 0)
                    .map_or(0, |index| idwt_buffer_size(level_rects[index]).unwrap().1);
                assert_eq!(
                    tile_ctx.idwt_scratch_buffer.capacity(),
                    below_pair,
                    "{label}: ping-pong buffer"
                );
            });
        }
    }
}

/// The build of a tile whose two finest IDWT levels pair admits it exactly
/// at the budget: from a retained baseline of the budget less its charge,
/// with one byte more refused before any storage is reserved. The IDWT part
/// of that charge is what the admitted decode's IDWT then holds at most.
#[cfg(feature = "parallel")]
#[test]
fn paired_idwt_tile_fits_exactly_the_budget_it_is_charged() {
    use crate::j2c::build::CoefficientLayout;
    use crate::{DecodeSettings, DecoderContext, Image};

    let bytes = gray_noise_codestream(1024, 768, 4, false);
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse fixture");
    let tiles = crate::j2c::tile::parse(
        &mut crate::reader::BitReader::new(image.codestream),
        &image.header,
        0,
    )
    .expect("parse tile");
    let pixels = BuildWorkspace::DecodePixels {
        skipped_resolution_levels: 0,
        coefficient_layout: CoefficientLayout::Component,
    };
    let build = |baseline: usize, workspace| {
        let mut storage = DecompositionStorage::default();
        let result =
            crate::j2c::build::build(&tiles[0], &mut storage, baseline, false, workspace, None);
        (result, storage)
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("pool");
    pool.install(|| {
        let (planned, storage) = build(0, pixels);
        planned.expect("plan from an empty baseline");
        let charge = storage.structural_workspace_bytes;

        let (fits, storage) = build(DEFAULT_MAX_DECODE_BYTES - charge, pixels);
        fits.expect("a tile that fits exactly");
        assert_eq!(storage.structural_workspace_bytes, DEFAULT_MAX_DECODE_BYTES);

        let (refused, storage) = build(DEFAULT_MAX_DECODE_BYTES - charge + 1, pixels);
        assert_eq!(
            refused,
            Err(DecodeError::Validation(ValidationError::ImageTooLarge))
        );
        assert_eq!(
            (
                storage.coefficients.capacity(),
                storage.code_blocks.capacity()
            ),
            (0, 0),
            "refused before allocating"
        );

        let (_, without_idwt) = build(0, BuildWorkspace::CoefficientsOnly);
        let idwt_charge = charge - without_idwt.structural_workspace_bytes;
        let mut context = DecoderContext::default();
        image
            .decode_with_context(&mut context)
            .expect("paired decode");
        let held = context
            .tile_decode_context
            .idwt_capacity_bytes()
            .expect("IDWT bytes");
        assert!(
            held <= idwt_charge,
            "the IDWT holds {held} bytes, the plan charged {idwt_charge}"
        );
    });
}

#[test]
fn impossible_metadata_count_rejects_at_the_logical_preflight() {
    let mut plan = coefficient_plan(0, 0);
    plan.layers = DEFAULT_MAX_DECODE_BYTES / size_of::<super::Layer>() + 1;
    assert_eq!(
        plan.validate_minimum_live_workspace(0),
        Err(DecodeError::Validation(ValidationError::ImageTooLarge))
    );
}

#[test]
fn reusable_growth_drops_the_empty_old_buffer_at_the_transient_boundary() {
    let mut storage = DecompositionStorage::default();
    storage
        .coefficients
        .try_reserve_exact(1)
        .expect("small test capacity");
    let target = storage.coefficients.capacity() + 1;
    let baseline = DEFAULT_MAX_DECODE_BYTES - target * size_of::<f32>();
    let mut reallocation =
        ReallocationBudget::for_storage(&storage, baseline).expect("live baseline");

    reallocation
        .reserve(&mut storage.coefficients, target)
        .expect("dropping the empty old buffer makes the exact final target fit");
    assert!(storage.coefficients.capacity() >= target);
}

#[test]
fn reusable_capacity_growth_rejects_a_logical_target_over_the_aggregate_cap() {
    let mut storage = DecompositionStorage::default();
    storage
        .coefficients
        .try_reserve_exact(1)
        .expect("small test capacity");
    let coefficient_bytes = storage.coefficients.capacity() * size_of::<f32>();
    let baseline = DEFAULT_MAX_DECODE_BYTES - coefficient_bytes;
    let mut reallocation =
        ReallocationBudget::for_storage(&storage, baseline).expect("live baseline");
    let target = storage.coefficients.capacity() + 1;

    assert_eq!(
        reallocation.reserve(&mut storage.coefficients, target),
        Err(DecodeError::Validation(ValidationError::ImageTooLarge))
    );
}

#[test]
fn zeroed_growth_releases_the_old_buffer_before_taking_the_new_one() {
    let mut storage = DecompositionStorage::default();
    storage.coefficients.push(9.0);
    let old_bytes = storage.coefficients.capacity() * size_of::<f32>();
    let target = storage.coefficients.capacity() + 1;
    // Room for the new buffer only once the old one is gone.
    let baseline = DEFAULT_MAX_DECODE_BYTES - target * size_of::<f32>();
    let mut reallocation =
        ReallocationBudget::for_storage(&storage, baseline).expect("live baseline");
    assert_eq!(reallocation.live_bytes(), baseline + old_bytes);

    reallocation
        .reserve_zeroed(&mut storage.coefficients, target)
        .expect("the old buffer is released first");
    assert_eq!(storage.coefficients, vec![0.0; target]);
    assert_eq!(
        reallocation.live_bytes(),
        baseline + storage.coefficients.capacity() * size_of::<f32>()
    );
}

#[test]
fn zeroed_reuse_keeps_the_allocation_and_none_of_its_samples() {
    let mut storage = DecompositionStorage::default();
    storage.coefficients.extend([5.0_f32; 8]);
    let capacity = storage.coefficients.capacity();
    let baseline = DEFAULT_MAX_DECODE_BYTES - capacity * size_of::<f32>();
    let mut reallocation =
        ReallocationBudget::for_storage(&storage, baseline).expect("live baseline");

    reallocation
        .reserve_zeroed(&mut storage.coefficients, 6)
        .expect("the retained allocation has room");
    crate::try_zero_fill_decode_elements(&mut storage.coefficients, 6).expect("zero fill");
    assert_eq!(storage.coefficients.capacity(), capacity);
    assert_eq!(storage.coefficients, vec![0.0; 6]);
    assert_eq!(reallocation.live_bytes(), DEFAULT_MAX_DECODE_BYTES);
}

#[test]
fn zeroed_growth_rejects_a_target_over_the_aggregate_cap() {
    let mut storage = DecompositionStorage::default();
    storage
        .coefficients
        .try_reserve_exact(1)
        .expect("small test capacity");
    let coefficient_bytes = storage.coefficients.capacity() * size_of::<f32>();
    let baseline = DEFAULT_MAX_DECODE_BYTES - coefficient_bytes;
    let mut reallocation =
        ReallocationBudget::for_storage(&storage, baseline).expect("live baseline");
    let target = storage.coefficients.capacity() + 1;

    assert_eq!(
        reallocation.reserve_zeroed(&mut storage.coefficients, target),
        Err(DecodeError::Validation(ValidationError::ImageTooLarge))
    );
    assert_eq!(storage.coefficients.capacity(), 0);
}

#[test]
fn region_decode_scratch_preserves_the_allocation_boundary() {
    use crate::{
        encode, DecodeSettings, DecoderContext, EncodeOptions, EncodeProgressionOrder, Image,
    };
    // Two large precincts of 4x4 blocks, with one pixel in the second.
    // Empty packets need no segments. A retained baseline leaves only a
    // small part of the cap available without allocating a large image.
    let pixels = vec![128; 128 * 64];
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 0,
        code_block_width_exp: 0,
        code_block_height_exp: 0,
        precinct_exponents: vec![(6, 6)],
        progression_order: EncodeProgressionOrder::Rlcp,
        num_layers: 3,
        ..EncodeOptions::default()
    };
    let control = encode(&pixels, 128, 64, 1, 8, false, &options).expect("encode fixture");
    let decode = |bytes: &[u8], baseline| -> crate::Result<Vec<f32>> {
        let image = Image::new_with_retained_baseline(bytes, &DecodeSettings::default(), baseline)?;
        let mut context = DecoderContext::default();
        let mut prepared = image.prepare_region_decoder_with_context(&mut context)?;
        let region = prepared.decode_region_components((64, 0, 1, 1))?;
        Ok(region.planes()[0].samples().to_vec())
    };
    // RLCP keeps the parsed-only precinct, so it has the pre-scratch charge.
    let mut accepted = 0;
    let mut rejected = DEFAULT_MAX_DECODE_BYTES;
    while rejected - accepted > 1 {
        let candidate = usize::midpoint(accepted, rejected);
        match decode(&control, candidate) {
            Ok(pixels) => {
                assert_eq!(pixels, [128.0]);
                accepted = candidate;
            }
            Err(error) => {
                assert!(
                    matches!(
                        error,
                        DecodeError::Validation(ValidationError::ImageTooLarge)
                            | DecodeError::AllocationTooLarge { .. }
                    ),
                    "{error:?}"
                );
                rejected = candidate;
            }
        }
    }
    let cod = control
        .windows(2)
        .position(|w| w == [0xff, 0x52])
        .expect("COD");
    for order in [2, 3, 4] {
        let mut bytes = control.clone();
        // Every packet is empty, so the payload is valid in all orders.
        bytes[cod + 5] = order;
        // Position-major orders also retain their sorted precinct positions
        // while parsing. Give the RLCP control the same remaining budget.
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse fixture");
        let tiles = crate::j2c::tile::parse(
            &mut crate::reader::BitReader::new(image.codestream),
            &image.header,
            0,
        )
        .expect("parse tile");
        let iterator_bytes = crate::j2c::progression::progression_iterator(&tiles[0], 0)
            .expect("position iterator")
            .owned_bytes();
        for baseline in [accepted - 1, accepted, rejected] {
            assert_eq!(
                decode(&bytes, baseline - iterator_bytes),
                decode(&control, baseline),
                "order={order}, baseline={baseline}"
            );
        }
    }
}

/// A layer record keeps its segment range in 8 bytes, tells a layer whose
/// packet is unread from one that contributed no segments, and refuses a
/// range it can't hold rather than truncating it.
#[test]
fn layer_records_keep_their_segment_ranges() {
    use crate::j2c::build::Layer;

    assert_eq!(size_of::<Layer>(), 8);
    assert_eq!(Layer::UNREAD.segments(), None);
    for range in [0..0, 0..3, 7..7, 41..48] {
        let layer = Layer::with_segments(range.clone()).expect("range fits");
        assert_eq!(layer.segments(), Some(range));
    }
    let last = u32::MAX as usize;
    assert_eq!(
        Layer::with_segments(last - 2..last - 1).and_then(Layer::segments),
        Some(last - 2..last - 1)
    );
    assert!(Layer::with_segments(last..last + 1).is_none());
    assert!(Layer::with_segments(0..last).is_none());
    let start = core::hint::black_box(5);
    assert!(Layer::with_segments(start..start - 2).is_none());
}
