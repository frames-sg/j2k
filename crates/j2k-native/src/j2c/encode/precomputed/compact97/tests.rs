// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::{J2kPacketizationCodeBlock, J2kPacketizationSubband, J2kSubBandType};
use alloc::vec;

fn vector_with_capacity<T>(capacity: usize) -> Vec<T> {
    let mut values = Vec::new();
    assert!(values.try_reserve_exact(capacity).is_ok());
    values
}

fn basic_compact_request_image() -> PreencodedHtj2k97CompactImage {
    PreencodedHtj2k97CompactImage {
        width: 1,
        height: 1,
        bit_depth: 8,
        signed: false,
        payload: Vec::new(),
        components: vec![PreencodedHtj2k97CompactComponent {
            x_rsiz: 1,
            y_rsiz: 1,
            resolutions: Vec::new(),
        }],
    }
}

#[test]
fn compact_request_rejects_every_ignored_output_option() {
    let image = basic_compact_request_image();
    let mut options = EncodeOptions {
        write_ppm: true,
        ..EncodeOptions::default()
    };
    assert!(validate_compact_request(&image, &options).is_err());
    options.write_ppm = false;
    options.write_ppt = true;
    assert!(validate_compact_request(&image, &options).is_err());
    options.write_ppt = false;
    options.num_layers = 2;
    assert!(validate_compact_request(&image, &options).is_err());
    options.num_layers = 1;
    options.quality_layer_byte_targets.push(1);
    assert!(validate_compact_request(&image, &options).is_err());
    options.quality_layer_byte_targets.clear();
    options.tile_size = Some((1, 1));
    assert!(validate_compact_request(&image, &options).is_err());
    options.tile_size = None;
    options.roi_component_shifts.push(0);
    assert!(validate_compact_request(&image, &options).is_err());
    options.roi_component_shifts.clear();
    options.component_sampling = Some(vec![(1, 1)]);
    assert!(validate_compact_request(&image, &options).is_err());
}

#[test]
fn compact_marker_field_validation_rejects_option_overflow() {
    let options = EncodeOptions {
        code_block_width_exp: u8::MAX,
        precinct_exponents: vec![(15, 15)],
        ..EncodeOptions::default()
    };

    assert_eq!(
        validate_precinct_exponents_for_options(&options, 0),
        Err("code-block width exponent exceeds supported range")
    );

    let excessive_guard_bits = EncodeOptions {
        guard_bits: MAX_QUANTIZATION_GUARD_BITS + 1,
        ..EncodeOptions::default()
    };
    assert!(
        validate_compact_request(&basic_compact_request_image(), &excessive_guard_bits).is_err()
    );
}

#[test]
fn compact_image_retained_owner_counts_every_nested_actual_capacity() {
    let mut payload = vector_with_capacity::<u8>(13);
    payload.push(0x2a);
    let mut code_blocks = vector_with_capacity::<PreencodedHtj2k97CompactCodeBlock>(7);
    code_blocks.push(PreencodedHtj2k97CompactCodeBlock {
        width: 1,
        height: 1,
        payload_range: 0..1,
        cleanup_length: 1,
        refinement_length: 0,
        num_coding_passes: 1,
        num_zero_bitplanes: 0,
    });
    let code_block_capacity = code_blocks.capacity();
    let mut subbands = vector_with_capacity::<PreencodedHtj2k97CompactSubband>(5);
    subbands.push(PreencodedHtj2k97CompactSubband {
        sub_band_type: J2kSubBandType::LowLow,
        num_cbs_x: 1,
        num_cbs_y: 1,
        total_bitplanes: 1,
        code_blocks,
    });
    let subband_capacity = subbands.capacity();
    let mut resolutions = vector_with_capacity::<PreencodedHtj2k97CompactResolution>(3);
    resolutions.push(PreencodedHtj2k97CompactResolution { subbands });
    let resolution_capacity = resolutions.capacity();
    let mut components = vector_with_capacity::<PreencodedHtj2k97CompactComponent>(2);
    components.push(PreencodedHtj2k97CompactComponent {
        x_rsiz: 1,
        y_rsiz: 1,
        resolutions,
    });
    let expected = payload.capacity()
        + components.capacity() * core::mem::size_of::<PreencodedHtj2k97CompactComponent>()
        + resolution_capacity * core::mem::size_of::<PreencodedHtj2k97CompactResolution>()
        + subband_capacity * core::mem::size_of::<PreencodedHtj2k97CompactSubband>()
        + code_block_capacity * core::mem::size_of::<PreencodedHtj2k97CompactCodeBlock>();
    let image = PreencodedHtj2k97CompactImage {
        width: 1,
        height: 1,
        bit_depth: 8,
        signed: false,
        payload,
        components,
    };

    assert_eq!(
        compact_image_retained_bytes(&image).expect("compact retained image bytes"),
        expected
    );
}

#[test]
fn compact_packet_phase_counts_nested_metadata_actual_capacities() {
    let payload = [7_u8];
    let mut code_blocks = vector_with_capacity::<PreparedCompactCodeBlock<'_>>(6);
    code_blocks.push(PreparedCompactCodeBlock {
        data: &payload,
        cleanup_length: 1,
        refinement_length: 0,
        num_coding_passes: 1,
        num_zero_bitplanes: 0,
    });
    let mut subbands = vector_with_capacity::<PreparedCompactSubband<'_>>(4);
    subbands.push(PreparedCompactSubband {
        code_blocks,
        num_cbs_x: 1,
        num_cbs_y: 1,
    });
    let mut prepared_packets = vector_with_capacity::<PreparedCompactResolutionPacket<'_>>(3);
    prepared_packets.push(PreparedCompactResolutionPacket {
        component: 0,
        resolution: 0,
        precinct: 0,
        subbands,
    });
    let packet_descriptors = vector_with_capacity::<J2kPacketizationPacketDescriptor>(5);
    let quant_params = vector_with_capacity::<(u16, u16)>(7);
    let plan = Compact97PacketPlan {
        params: EncodeParams::default(),
        quant_params,
        prepared_packets,
        packet_descriptors,
        retained_input_bytes: 0,
    };
    let plan_owner_bytes = plan
        .plan_owner_retained_bytes()
        .expect("compact plan owner bytes");
    let retained_plan_bytes = plan_owner_bytes
        + plan.packet_descriptors.capacity()
            * core::mem::size_of::<J2kPacketizationPacketDescriptor>();
    let session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())
        .expect("compact metadata session");
    let public_resolutions = construction::try_public_packet_metadata(
        &plan.prepared_packets,
        &session,
        retained_plan_bytes,
    )
    .expect("fallible compact public metadata");
    let public_code_blocks = &public_resolutions[0].subbands[0].code_blocks;
    let expected = plan.quant_params.capacity() * core::mem::size_of::<(u16, u16)>()
        + plan.prepared_packets.capacity()
            * core::mem::size_of::<PreparedCompactResolutionPacket<'_>>()
        + plan.prepared_packets[0].subbands.capacity()
            * core::mem::size_of::<PreparedCompactSubband<'_>>()
        + plan.prepared_packets[0].subbands[0].code_blocks.capacity()
            * core::mem::size_of::<PreparedCompactCodeBlock<'_>>()
        + plan.packet_descriptors.capacity()
            * core::mem::size_of::<J2kPacketizationPacketDescriptor>()
        + public_resolutions.capacity() * core::mem::size_of::<J2kPacketizationResolution<'_>>()
        + public_resolutions[0].subbands.capacity()
            * core::mem::size_of::<J2kPacketizationSubband<'_>>()
        + public_code_blocks.capacity() * core::mem::size_of::<J2kPacketizationCodeBlock<'_>>();
    assert_eq!(
        plan.packet_phase_retained_bytes(
            plan_owner_bytes,
            &public_resolutions,
            public_resolutions.capacity(),
        )
        .expect("compact packet phase bytes"),
        expected
    );
}

#[test]
fn compact_final_codestream_high_water_accepts_exact_cap_and_rejects_one_byte_over() {
    let mut tile_data = vector_with_capacity::<u8>(11);
    tile_data.extend_from_slice(&[1, 2, 3]);
    let quant_params = vector_with_capacity::<(u16, u16)>(5);
    let packetized = Compact97Packetized {
        params: EncodeParams::default(),
        quant_params,
        tile_data,
    };
    let accounted = codestream_write::write_codestream_accounted_with_peak_check(
        &packetized.params,
        &packetized.tile_data,
        &packetized.quant_params,
        |_| Ok(()),
    )
    .expect("accounted compact codestream");
    assert_eq!(
        accounted.writer_peak_bytes,
        accounted.codestream.capacity(),
        "scratch-free single-tile writer peak is the actual output capacity"
    );
    let owner_bytes =
        compact_final_owner_retained_bytes(&packetized).expect("compact final owner bytes");
    let exact_cap = owner_bytes + accounted.writer_peak_bytes;
    let exact = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), exact_cap)
        .expect("exact compact final session");
    reconcile_compact_final_codestream(&exact, &packetized, accounted.writer_peak_bytes)
        .expect("exact compact final high-water");

    let cap = exact_cap - 1;
    let over = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), cap)
        .expect("compact final owners remain below cap");
    let error = reconcile_compact_final_codestream(&over, &packetized, accounted.writer_peak_bytes)
        .expect_err("compact final codestream is one byte over cap");
    assert_eq!(
        error,
        EncodeError::AllocationTooLarge {
            what: FINAL_HIGH_WATER,
            requested: exact_cap,
            cap,
        }
    );
}

#[test]
fn compact_accounted_finalizer_preserves_codestream_byte_parity() {
    let packetized = Compact97Packetized {
        params: EncodeParams::default(),
        quant_params: Vec::new(),
        tile_data: vec![1, 3, 3, 7],
    };
    let expected = codestream_write::write_codestream(
        &packetized.params,
        &packetized.tile_data,
        &packetized.quant_params,
    )
    .expect("legacy codestream writer");
    let actual = codestream_write::write_codestream_accounted_with_peak_check(
        &packetized.params,
        &packetized.tile_data,
        &packetized.quant_params,
        |_| Ok(()),
    )
    .expect("accounted codestream writer");

    assert_eq!(actual.codestream, expected);
}

#[test]
fn compact_final_writer_rejects_preflight_before_reservation() {
    let params = EncodeParams::default();
    let tile_data = [1_u8, 2, 3];
    let mut observed_peak = None;
    let error = codestream_write::write_codestream_accounted_with_peak_check(
        &params,
        &tile_data,
        &[],
        |requested| {
            observed_peak = Some(requested);
            Err(EncodeError::AllocationTooLarge {
                what: FINAL_HIGH_WATER,
                requested,
                cap: requested - 1,
            })
        },
    )
    .expect_err("writer preflight must reject before reservation");
    let requested = observed_peak.expect("preflight callback ran");
    assert_eq!(
        error,
        EncodeError::AllocationTooLarge {
            what: FINAL_HIGH_WATER,
            requested,
            cap: requested - 1,
        }
    );
}
