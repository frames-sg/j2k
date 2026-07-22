// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::{
    encode_htj2k, DecodeSettings, DecoderContext, EncodeOptions, Image, J2kDirectGrayscaleStep,
};

mod prepared;

#[test]
fn referenced_htj2k_payload_ranges_reconstruct_owned_direct_plan_bytes() {
    let pixels = (0_u8..64).collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let bytes = encode_htj2k(&pixels, 8, 8, 1, 8, false, &options).expect("encode HT fixture");
    let image = Image::new(&bytes, &DecodeSettings::strict()).expect("strict HT image");
    let mut owned_context = DecoderContext::default();
    let owned = image
        .build_direct_grayscale_plan_with_context(&mut owned_context)
        .expect("owned direct plan");
    let mut referenced_context = DecoderContext::default();
    let referenced = image
        .build_referenced_htj2k_plan_region_with_context(&mut referenced_context, (0, 0, 8, 8))
        .expect("referenced direct plan");
    let geometry = referenced
        .grayscale_geometry()
        .expect("grayscale referenced plan");
    let mut payload_cursor = 0usize;

    for (owned_step, referenced_step) in owned.steps.iter().zip(&geometry.steps) {
        let (
            J2kDirectGrayscaleStep::HtSubBand(owned_sub_band),
            J2kDirectGrayscaleStep::HtSubBand(referenced_sub_band),
        ) = (owned_step, referenced_step)
        else {
            continue;
        };
        for (owned_job, referenced_job) in owned_sub_band.jobs.iter().zip(&referenced_sub_band.jobs)
        {
            assert!(referenced_job.data.is_empty());
            let payload = referenced
                .payloads()
                .get(payload_cursor)
                .expect("payload for referenced job");
            payload_cursor += 1;
            let mut reconstructed =
                bytes[payload.cleanup.offset..payload.cleanup.end().expect("cleanup end")].to_vec();
            if let Some(refinement) = payload.refinement {
                reconstructed.extend_from_slice(
                    &bytes[refinement.offset..refinement.end().expect("refinement end")],
                );
            }
            assert_eq!(reconstructed, owned_job.data);
        }
    }
    assert_eq!(payload_cursor, referenced.payloads().len());
}
