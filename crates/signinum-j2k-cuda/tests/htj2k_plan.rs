use signinum_core::{CodecError, PixelFormat};
use signinum_j2k_cuda::{CudaHtj2kDecodePlan, CudaHtj2kTransform, J2kDecoder, SurfaceResidency};
use signinum_j2k_native::{
    encode, encode_htj2k, DecodeSettings, DecoderContext, EncodeOptions, HtOwnedCodeBlockBatchJob,
    HtOwnedSubBandPlan, Image, J2kDirectGrayscalePlan, J2kDirectGrayscaleStep, J2kDirectIdwtStep,
    J2kDirectStoreStep, J2kRect, J2kWaveletTransform,
};

fn ht_gray8_fixture() -> Vec<u8> {
    let pixels: Vec<u8> = (0..64).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, 8, 8, 1, 8, false, &options).expect("encode ht gray8")
}

fn ht_gray8_irreversible_97_fixture() -> Vec<u8> {
    let pixels: Vec<u8> = (0..64).collect();
    let options = EncodeOptions {
        reversible: false,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, 8, 8, 1, 8, false, &options).expect("encode ht gray8 9/7")
}

fn ht_gray8_large_fixture() -> Vec<u8> {
    let mut pixels = Vec::with_capacity(256 * 256);
    for y in 0..256u32 {
        for x in 0..256u32 {
            pixels.push(((x * 3 + y * 5) & 0xff) as u8);
        }
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 3,
        code_block_width_exp: 0,
        code_block_height_exp: 0,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, 256, 256, 1, 8, false, &options).expect("encode large ht gray8")
}

fn classic_gray8_fixture() -> Vec<u8> {
    let pixels: Vec<u8> = (0..64).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 8, 8, 1, 8, false, &options).expect("encode classic gray8")
}

fn openhtj2k_refinement_odd_fixture() -> &'static [u8] {
    include_bytes!("fixtures/htj2k/openhtj2k_ds0_ht_09_b11.j2k")
}

fn one_block_ht_plan(
    cleanup_length: u32,
    refinement_length: u32,
    data: Vec<u8>,
    output_stride: usize,
) -> J2kDirectGrayscalePlan {
    J2kDirectGrayscalePlan {
        dimensions: (1, 1),
        bit_depth: 8,
        steps: vec![
            J2kDirectGrayscaleStep::HtSubBand(HtOwnedSubBandPlan {
                band_id: 0,
                rect: J2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                },
                width: 1,
                height: 1,
                jobs: vec![HtOwnedCodeBlockBatchJob {
                    output_x: 0,
                    output_y: 0,
                    data,
                    cleanup_length,
                    refinement_length,
                    width: 1,
                    height: 1,
                    output_stride,
                    missing_bit_planes: 0,
                    number_of_coding_passes: 1,
                    num_bitplanes: 8,
                    stripe_causal: false,
                    strict: true,
                    dequantization_step: 1.0,
                }],
            }),
            J2kDirectGrayscaleStep::Store(J2kDirectStoreStep {
                input_band_id: 0,
                input_rect: J2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                },
                source_x: 0,
                source_y: 0,
                copy_width: 1,
                copy_height: 1,
                output_width: 1,
                output_height: 1,
                output_x: 0,
                output_y: 0,
                addend: 128.0,
            }),
        ],
    }
}

#[test]
fn flat_htj2k_plan_contains_payload_offsets_not_pointers() {
    let bytes = ht_gray8_fixture();
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse image");
    let mut context = DecoderContext::default();
    let native_plan = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("native direct plan");

    let cuda_plan =
        CudaHtj2kDecodePlan::from_grayscale_direct_plan(&native_plan, PixelFormat::Gray8, (0, 0))
            .expect("CUDA flat plan");

    assert_eq!(cuda_plan.dimensions(), native_plan.dimensions);
    assert_eq!(cuda_plan.output_format(), PixelFormat::Gray8);
    assert_eq!(cuda_plan.transform(), CudaHtj2kTransform::Reversible53);
    assert!(!cuda_plan.payload().is_empty());
    assert!(!cuda_plan.code_blocks().is_empty());
    assert!(!cuda_plan.subbands().is_empty());
    assert_eq!(
        cuda_plan.dispatch_count_hint(),
        cuda_plan.code_blocks().len()
    );

    let native_payload_len: usize = native_plan
        .steps
        .iter()
        .filter_map(|step| match step {
            J2kDirectGrayscaleStep::HtSubBand(subband) => Some(subband),
            _ => None,
        })
        .flat_map(|subband| subband.jobs.iter())
        .map(|job| job.data.len())
        .sum();
    assert_eq!(cuda_plan.payload().len(), native_payload_len);

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
    let image = Image::new(
        openhtj2k_refinement_odd_fixture(),
        &DecodeSettings::default(),
    )
    .expect("image");
    let mut context = DecoderContext::default();
    let native_plan = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("native refinement direct plan");
    let native_jobs = native_plan
        .steps
        .iter()
        .filter_map(|step| match step {
            J2kDirectGrayscaleStep::HtSubBand(subband) => Some(subband),
            _ => None,
        })
        .flat_map(|subband| subband.jobs.iter())
        .collect::<Vec<_>>();

    let cuda_plan =
        CudaHtj2kDecodePlan::from_grayscale_direct_plan(&native_plan, PixelFormat::Gray8, (0, 0))
            .expect("CUDA refinement flat plan");

    assert_eq!(native_jobs.len(), 14);
    assert_eq!(cuda_plan.code_blocks().len(), native_jobs.len());
    let mut cursor = 0usize;
    let mut refinement_blocks = 0usize;
    for (block, job) in cuda_plan.code_blocks().iter().zip(native_jobs) {
        assert_eq!(block.payload_offset, cursor as u64);
        assert_eq!(block.payload_len as usize, job.data.len());
        assert_eq!(block.cleanup_length, job.cleanup_length);
        assert_eq!(block.refinement_length, job.refinement_length);
        assert_eq!(block.number_of_coding_passes, job.number_of_coding_passes);
        assert_eq!(block.missing_bit_planes, job.missing_bit_planes);
        let end = cursor + job.data.len();
        assert_eq!(&cuda_plan.payload()[cursor..end], job.data.as_slice());
        cursor = end;
        if block.refinement_length > 0 {
            refinement_blocks += 1;
        }
    }
    assert_eq!(cursor, cuda_plan.payload().len());
    assert_eq!(refinement_blocks, 14);
}

#[test]
fn flat_htj2k_plan_rejects_block_length_mismatch() {
    let native_plan = one_block_ht_plan(1, 2, vec![0xAA, 0xBB], 1);

    let error =
        CudaHtj2kDecodePlan::from_grayscale_direct_plan(&native_plan, PixelFormat::Gray8, (0, 0))
            .expect_err("mismatched cleanup/refinement lengths must be rejected");

    assert!(error.is_unsupported());
    assert!(
        error
            .to_string()
            .contains("block lengths do not match payload bytes"),
        "unexpected error: {error}"
    );
}

#[test]
fn flat_htj2k_plan_rejects_output_stride_overflow() {
    let native_plan = one_block_ht_plan(1, 0, vec![0xAA], usize::MAX);

    let error =
        CudaHtj2kDecodePlan::from_grayscale_direct_plan(&native_plan, PixelFormat::Gray8, (0, 0))
            .expect_err("unrepresentable output stride must be rejected");

    assert!(error.is_unsupported());
}

#[test]
fn flat_htj2k_plan_records_irreversible_97_transform() {
    let bytes = ht_gray8_irreversible_97_fixture();
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse image");
    let mut context = DecoderContext::default();
    let native_plan = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("native direct plan");

    let cuda_plan =
        CudaHtj2kDecodePlan::from_grayscale_direct_plan(&native_plan, PixelFormat::Gray8, (0, 0))
            .expect("CUDA flat 9/7 plan");

    assert_eq!(cuda_plan.transform(), CudaHtj2kTransform::Irreversible97);
    assert!(!cuda_plan.idwt_steps().is_empty());
    assert!(cuda_plan
        .idwt_steps()
        .iter()
        .all(|step| step.transform == CudaHtj2kTransform::Irreversible97));
}

#[test]
fn flat_htj2k_plan_rejects_mixed_idwt_transforms() {
    let mut native_plan = one_block_ht_plan(1, 0, vec![0xAA], 1);
    let rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 1,
        y1: 1,
    };
    native_plan.steps.insert(
        1,
        J2kDirectGrayscaleStep::Idwt(J2kDirectIdwtStep {
            output_band_id: 4,
            rect,
            transform: J2kWaveletTransform::Reversible53,
            ll_band_id: 0,
            ll: rect,
            hl_band_id: 1,
            hl: rect,
            lh_band_id: 2,
            lh: rect,
            hh_band_id: 3,
            hh: rect,
        }),
    );
    native_plan.steps.insert(
        2,
        J2kDirectGrayscaleStep::Idwt(J2kDirectIdwtStep {
            output_band_id: 8,
            rect,
            transform: J2kWaveletTransform::Irreversible97,
            ll_band_id: 4,
            ll: rect,
            hl_band_id: 5,
            hl: rect,
            lh_band_id: 6,
            lh: rect,
            hh_band_id: 7,
            hh: rect,
        }),
    );

    let error =
        CudaHtj2kDecodePlan::from_grayscale_direct_plan(&native_plan, PixelFormat::Gray8, (0, 0))
            .expect_err("mixed transforms must be rejected");

    assert!(error.is_unsupported());
    assert!(
        error.to_string().contains("mixed DWT transforms"),
        "unexpected error: {error}"
    );
}

#[test]
fn flat_htj2k_region_plan_stores_compact_output_rect() {
    let bytes = ht_gray8_fixture();
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse image");
    let mut context = DecoderContext::default();
    let native_plan = image
        .build_direct_grayscale_plan_region_with_context(&mut context, (2, 1, 6, 5))
        .expect("native ROI direct plan");

    let cuda_plan = CudaHtj2kDecodePlan::from_grayscale_direct_plan_region(
        &native_plan,
        PixelFormat::Gray8,
        (2, 1),
        (4, 4),
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
fn flat_htj2k_region_plan_rejects_store_outside_output_rect() {
    let native_plan = one_block_ht_plan(1, 0, vec![0xAA], 1);

    let error = CudaHtj2kDecodePlan::from_grayscale_direct_plan_region(
        &native_plan,
        PixelFormat::Gray8,
        (1, 1),
        (0, 0),
    )
    .expect_err("store outside compact output rectangle must be rejected");

    assert!(error.is_unsupported());
    assert!(
        error
            .to_string()
            .contains("store does not fit the requested output rectangle"),
        "unexpected error: {error}"
    );
}

#[test]
fn flat_htj2k_region_plan_prunes_code_blocks() {
    let bytes = ht_gray8_large_fixture();
    let image = Image::new(
        &bytes,
        &DecodeSettings {
            target_resolution: Some((64, 64)),
            ..DecodeSettings::default()
        },
    )
    .expect("parse scaled image");
    let mut full_context = DecoderContext::default();
    let full_native_plan = image
        .build_direct_grayscale_plan_with_context(&mut full_context)
        .expect("native full direct plan");
    let full_cuda_plan = CudaHtj2kDecodePlan::from_grayscale_direct_plan(
        &full_native_plan,
        PixelFormat::Gray8,
        (0, 0),
    )
    .expect("CUDA full flat plan");

    let mut roi_context = DecoderContext::default();
    let roi_native_plan = image
        .build_direct_grayscale_plan_region_with_context(&mut roi_context, (24, 24, 8, 8))
        .expect("native ROI direct plan");
    let roi_cuda_plan = CudaHtj2kDecodePlan::from_grayscale_direct_plan_region(
        &roi_native_plan,
        PixelFormat::Gray8,
        (24, 24),
        (8, 8),
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
    let image = Image::new(
        &bytes,
        &DecodeSettings {
            target_resolution: Some((4, 4)),
            ..DecodeSettings::default()
        },
    )
    .expect("parse scaled image");
    let mut context = DecoderContext::default();
    let native_plan = image
        .build_direct_grayscale_plan_region_with_context(&mut context, (1, 0, 2, 3))
        .expect("native scaled ROI direct plan");

    let cuda_plan = CudaHtj2kDecodePlan::from_grayscale_direct_plan_region(
        &native_plan,
        PixelFormat::Gray8,
        (1, 0),
        (2, 3),
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
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse image");
    let mut context = DecoderContext::default();
    let native_plan = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("native direct plan");

    let error =
        CudaHtj2kDecodePlan::from_grayscale_direct_plan(&native_plan, PixelFormat::Gray8, (0, 0))
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
