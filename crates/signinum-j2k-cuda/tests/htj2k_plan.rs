use signinum_core::{CodecError, PixelFormat, Rect};
use signinum_j2k_cuda::{CudaHtj2kTransform, J2kDecoder, SurfaceResidency};
use signinum_test_support::{
    classic_j2k_gray8_fixture, htj2k_gray8_97_fixture, htj2k_gray8_fixture,
    htj2k_gray8_large_fixture,
    openhtj2k_refinement_odd_fixture as shared_openhtj2k_refinement_odd_fixture,
};

fn ht_gray8_fixture() -> Vec<u8> {
    htj2k_gray8_fixture(8, 8)
}

fn ht_gray8_irreversible_97_fixture() -> Vec<u8> {
    htj2k_gray8_97_fixture(8, 8)
}

fn ht_gray8_large_fixture() -> Vec<u8> {
    htj2k_gray8_large_fixture(256, 256)
}

fn classic_gray8_fixture() -> Vec<u8> {
    classic_j2k_gray8_fixture(8, 8)
}

fn openhtj2k_refinement_odd_fixture() -> &'static [u8] {
    shared_openhtj2k_refinement_odd_fixture()
}

#[test]
fn flat_htj2k_plan_contains_payload_offsets_not_pointers() {
    let bytes = ht_gray8_fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let (cuda_plan, _) = decoder
        .build_cuda_htj2k_grayscale_plan_with_profile(PixelFormat::Gray8)
        .expect("CUDA flat plan");

    assert_eq!(cuda_plan.dimensions(), (8, 8));
    assert_eq!(cuda_plan.output_format(), PixelFormat::Gray8);
    assert_eq!(cuda_plan.transform(), CudaHtj2kTransform::Reversible53);
    assert!(!cuda_plan.payload().is_empty());
    assert!(!cuda_plan.code_blocks().is_empty());
    assert!(!cuda_plan.subbands().is_empty());
    assert_eq!(
        cuda_plan.dispatch_count_hint(),
        cuda_plan.code_blocks().len()
    );

    let block_payload_len = cuda_plan
        .code_blocks()
        .iter()
        .map(|block| block.payload_len as usize)
        .sum();
    assert_eq!(cuda_plan.payload().len(), block_payload_len);

    for block in cuda_plan.code_blocks() {
        let start = usize::try_from(block.payload_offset).expect("payload offset fits usize");
        let end = start + block.payload_len as usize;
        assert!(end <= cuda_plan.payload().len());
        assert_eq!(
            block.payload_len,
            block.cleanup_length + block.refinement_length
        );
    }
}

#[test]
fn flat_htj2k_plan_preserves_refinement_payload_metadata() {
    let mut decoder = J2kDecoder::new(openhtj2k_refinement_odd_fixture()).expect("decoder");
    let (cuda_plan, _) = decoder
        .build_cuda_htj2k_grayscale_plan_with_profile(PixelFormat::Gray8)
        .expect("CUDA refinement flat plan");

    assert_eq!(cuda_plan.code_blocks().len(), 14);
    let mut cursor = 0usize;
    let mut refinement_blocks = 0usize;
    for block in cuda_plan.code_blocks() {
        assert_eq!(block.payload_offset, cursor as u64);
        assert_eq!(
            block.payload_len,
            block.cleanup_length + block.refinement_length
        );
        let end = cursor + block.payload_len as usize;
        assert!(end <= cuda_plan.payload().len());
        cursor = end;
        if block.refinement_length > 0 {
            refinement_blocks += 1;
        }
    }
    assert_eq!(cursor, cuda_plan.payload().len());
    assert_eq!(refinement_blocks, 14);
}

#[test]
fn flat_htj2k_plan_records_irreversible_97_transform() {
    let bytes = ht_gray8_irreversible_97_fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let (cuda_plan, _) = decoder
        .build_cuda_htj2k_grayscale_plan_with_profile(PixelFormat::Gray8)
        .expect("CUDA flat 9/7 plan");

    assert_eq!(cuda_plan.transform(), CudaHtj2kTransform::Irreversible97);
    assert!(!cuda_plan.idwt_steps().is_empty());
    assert!(cuda_plan
        .idwt_steps()
        .iter()
        .all(|step| step.transform == CudaHtj2kTransform::Irreversible97));
}

#[test]
fn flat_htj2k_region_plan_stores_compact_output_rect() {
    let bytes = ht_gray8_fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let (cuda_plan, _) = decoder
        .build_cuda_htj2k_grayscale_region_plan_with_profile(
            PixelFormat::Gray8,
            Rect {
                x: 2,
                y: 1,
                w: 4,
                h: 4,
            },
        )
        .expect("CUDA ROI flat plan");

    assert_eq!(cuda_plan.dimensions(), (4, 4));
    assert_eq!(cuda_plan.output_origin(), (2, 1));
    let [store] = cuda_plan.store_steps() else {
        panic!("expected one ROI store");
    };
    assert_eq!(store.output_width, 4);
    assert_eq!(store.output_height, 4);
    assert_eq!(store.source_x, 2);
    assert_eq!(store.source_y, 1);
    assert_eq!(store.output_x, 0);
    assert_eq!(store.output_y, 0);
    assert_eq!(store.copy_width, 4);
    assert_eq!(store.copy_height, 4);
}

#[test]
fn flat_htj2k_region_plan_prunes_code_blocks() {
    let bytes = ht_gray8_large_fixture();
    let mut full_decoder = J2kDecoder::new(&bytes).expect("full decoder");
    let (full_cuda_plan, _) = full_decoder
        .build_cuda_htj2k_grayscale_scaled_plan_with_profile(PixelFormat::Gray8, (64, 64))
        .expect("CUDA full flat plan");

    let mut roi_decoder = J2kDecoder::new(&bytes).expect("ROI decoder");
    let (roi_cuda_plan, _) = roi_decoder
        .build_cuda_htj2k_grayscale_region_scaled_plan_with_profile(
            PixelFormat::Gray8,
            Rect {
                x: 24,
                y: 24,
                w: 8,
                h: 8,
            },
            (64, 64),
        )
        .expect("CUDA ROI flat plan");

    assert!(
        roi_cuda_plan.code_blocks().len() < full_cuda_plan.code_blocks().len(),
        "ROI should hand CUDA fewer code blocks than a full decode"
    );
    assert_eq!(roi_cuda_plan.dimensions(), (8, 8));
    let [store] = roi_cuda_plan.store_steps() else {
        panic!("expected one ROI store");
    };
    assert_eq!(store.copy_width, 8);
    assert_eq!(store.copy_height, 8);
}

#[test]
fn flat_htj2k_scaled_region_plan_stores_compact_scaled_rect() {
    let bytes = ht_gray8_fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let (cuda_plan, _) = decoder
        .build_cuda_htj2k_grayscale_region_scaled_plan_with_profile(
            PixelFormat::Gray8,
            Rect {
                x: 1,
                y: 0,
                w: 2,
                h: 3,
            },
            (4, 4),
        )
        .expect("CUDA scaled ROI flat plan");

    assert_eq!(cuda_plan.dimensions(), (2, 3));
    assert_eq!(cuda_plan.output_origin(), (1, 0));
    let [store] = cuda_plan.store_steps() else {
        panic!("expected one scaled ROI store");
    };
    assert_eq!(store.output_width, 2);
    assert_eq!(store.output_height, 3);
    assert_eq!(store.source_x, 1);
    assert_eq!(store.source_y, 0);
    assert_eq!(store.output_x, 0);
    assert_eq!(store.output_y, 0);
    assert_eq!(store.copy_width, 2);
    assert_eq!(store.copy_height, 3);
}

#[test]
fn flat_htj2k_plan_rejects_classic_j2k_subband_steps() {
    let bytes = classic_gray8_fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let error = decoder
        .build_cuda_htj2k_grayscale_plan_with_profile(PixelFormat::Gray8)
        .expect_err("classic plans must be rejected");

    assert!(error.is_unsupported());
}

#[test]
fn grayscale_plan_profile_reports_stable_cuda_htj2k_fields() {
    let bytes = ht_gray8_fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");

    let (plan, report) = decoder
        .build_cuda_htj2k_grayscale_plan_with_profile(PixelFormat::Gray8)
        .expect("profiled plan");

    assert_eq!(report.block_count, plan.code_blocks().len());
    assert_eq!(report.payload_bytes, plan.payload().len());
    assert_eq!(report.dispatch_count, 0);
    assert_eq!(report.residency, SurfaceResidency::CudaResidentDecode);
    assert_eq!(report.h2d_us, 0);
    assert_eq!(report.ht_cleanup_us, 0);
    assert_eq!(report.ht_refine_us, 0);
    assert_eq!(report.dequant_us, 0);
    assert_eq!(report.idwt_us, 0);
    assert_eq!(report.mct_us, 0);
    assert_eq!(report.store_us, 0);
}
