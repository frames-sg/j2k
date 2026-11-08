// SPDX-License-Identifier: MIT OR Apache-2.0

//! Additive exact-operation ABI: native values, controls, and explicit sRGB.

use j2k_c::*;
use std::ptr;

fn constant_source(bits: u8, signed: bool, value: i16) -> Vec<u8> {
    let mut data = Vec::new();
    for _ in 0..65 * 65 {
        data.push(value.to_le_bytes()[0]);
        if bits > 8 {
            data.push(value.to_le_bytes()[1]);
        }
    }
    let samples = j2k::J2kLosslessSamples::new(&data, 65, 65, 1, bits, signed).unwrap();
    let options = j2k::J2kLosslessEncodeOptions::default()
        .with_cpu_only_backend()
        .with_max_decomposition_levels(Some(4));
    j2k::encode_j2k_lossless(samples, &options)
        .unwrap()
        .codestream
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one FFI lifecycle matrix checks native widths, padding, controls, and final handle cleanup"
)]
fn exact_native_operations_preserve_values_and_honor_controls() {
    for (bits, signed, value, sample_type) in [
        (7, false, 63, J2K_U8),
        (12, false, 2048, J2K_U16_LE),
        (16, true, -1234, J2K_I16_LE),
    ] {
        let bytes = constant_source(bits, signed, value);
        let mut decoder = ptr::null_mut();
        let mut error = J2kCError::default();
        let mut plan = J2kCRegionPlan::default();
        let mut options = J2kCOperationOptions {
            x: 17,
            y: 9,
            width: 19,
            height: 25,
            reduction: 4,
            max_working_memory_bytes: 8 << 20,
            ..Default::default()
        };
        // SAFETY: Live disjoint inputs/outputs, initialized ABI prefixes, serial
        // decoder ownership, and token destruction only after each call returns.
        unsafe {
            assert_eq!(
                j2k_decoder_open(
                    bytes.as_ptr(),
                    bytes.len(),
                    ptr::null(),
                    &raw mut decoder,
                    &raw mut error
                ),
                J2K_OK
            );
            // A controlled operation on an unprepared source prepares it under
            // that operation's working limit; a failed preparation is not kept.
            options.max_working_memory_bytes = 1;
            assert_eq!(
                j2k_decoder_plan_region(decoder, &raw const options, &raw mut plan, &raw mut error),
                J2K_RESOURCE_LIMIT,
                "source preparation obeys the operation's working limit"
            );
            options.max_working_memory_bytes = 8 << 20;
            assert_eq!(
                j2k_decoder_plan_region(decoder, &raw const options, &raw mut plan, &raw mut error),
                J2K_OK
            );
            assert_eq!(
                j2k_decoder_prepare_operations(decoder, &raw mut error),
                J2K_OK,
                "explicit preparation of an already prepared source is a no-op"
            );
            assert_eq!(
                (
                    plan.output.x,
                    plan.output.y,
                    plan.output.width,
                    plan.output.height
                ),
                (1, 0, 2, 3)
            );
            assert_eq!((plan.reduction, plan.max_reduction), (4, 4));
            assert_eq!(
                (plan.output.sample_type, plan.output.precision, plan.route),
                (sample_type, u32::from(bits), J2K_ROUTE_NATIVE_CPU)
            );
            let row = usize::try_from(plan.output.row_bytes).unwrap();
            let stride = row + 3;
            let mut output = vec![0xa5; stride * plan.output.height as usize];
            assert_eq!(
                j2k_decoder_decode_region_into(
                    decoder,
                    &raw const options,
                    output.as_mut_ptr(),
                    output.len(),
                    stride,
                    ptr::null_mut(),
                    &raw mut error
                ),
                J2K_OK
            );
            let sample = if sample_type == J2K_U8 {
                vec![value.to_le_bytes()[0]]
            } else {
                value.to_le_bytes().to_vec()
            };
            for line in output.chunks(stride) {
                assert_eq!(&line[..row], sample.repeat(2));
                assert!(line[row..].iter().all(|&byte| byte == 0xa5));
            }
            options.max_output_bytes = plan.output.buffer_bytes - 1;
            output.fill(0xa5);
            assert_eq!(
                j2k_decoder_decode_region_into(
                    decoder,
                    &raw const options,
                    output.as_mut_ptr(),
                    output.len(),
                    stride,
                    ptr::null_mut(),
                    &raw mut error
                ),
                J2K_RESOURCE_LIMIT
            );
            assert!(output.iter().all(|&byte| byte == 0xa5));
            options.max_output_bytes = 0;
            options.max_working_memory_bytes = 1;
            assert_eq!(
                j2k_decoder_decode_region_into(
                    decoder,
                    &raw const options,
                    output.as_mut_ptr(),
                    output.len(),
                    stride,
                    ptr::null_mut(),
                    &raw mut error
                ),
                J2K_RESOURCE_LIMIT
            );
            options.max_working_memory_bytes = 0;
            let mut cancellation = ptr::null_mut();
            assert_eq!(
                j2k_cancellation_create(&raw mut cancellation, &raw mut error),
                J2K_OK
            );
            assert_eq!(
                j2k_cancellation_cancel(cancellation, &raw mut error),
                J2K_OK
            );
            options.cancellation = cancellation;
            assert_eq!(
                j2k_decoder_decode_region_into(
                    decoder,
                    &raw const options,
                    output.as_mut_ptr(),
                    output.len(),
                    stride,
                    ptr::null_mut(),
                    &raw mut error
                ),
                J2K_CANCELLED
            );
            options.cancellation = ptr::null();
            assert_eq!(
                j2k_cancellation_free(&raw mut cancellation, &raw mut error),
                J2K_OK
            );
            assert!(cancellation.is_null());
            assert_eq!(
                j2k_decoder_set_operation_cache_limit(decoder, 0, &raw mut error),
                J2K_OK
            );
            assert_eq!(
                j2k_decoder_decode_region_into(
                    decoder,
                    &raw const options,
                    output.as_mut_ptr(),
                    output.len(),
                    stride,
                    ptr::null_mut(),
                    &raw mut error
                ),
                J2K_OK
            );
            options.reduction = 5;
            assert_eq!(
                j2k_decoder_plan_region(decoder, &raw const options, &raw mut plan, &raw mut error),
                J2K_UNSUPPORTED
            );
            options.reduction = 32;
            assert_eq!(
                j2k_decoder_plan_region(decoder, &raw const options, &raw mut plan, &raw mut error),
                J2K_INVALID_ARGUMENT
            );
            assert_eq!(j2k_decoder_close(&raw mut decoder, &raw mut error), J2K_OK);
        }
    }
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one FFI lifecycle checks sRGB output, each control and decoder reuse after cancellation"
)]
fn exact_srgb_output_is_explicit_and_honors_operation_controls() {
    let bytes = constant_source(7, false, 63);
    let mut decoder = ptr::null_mut();
    let mut error = J2kCError::default();
    let mut plan = J2kCRegionPlan::default();
    let mut options = J2kCOperationOptions {
        reduction: 4,
        interpretation: J2K_INTERPRETATION_SRGB8,
        ..Default::default()
    };
    // SAFETY: Valid disjoint initialized arguments; the live decoder is used serially.
    unsafe {
        assert_eq!(
            j2k_decoder_open(
                bytes.as_ptr(),
                bytes.len(),
                ptr::null(),
                &raw mut decoder,
                &raw mut error
            ),
            J2K_OK
        );
        assert_eq!(
            j2k_decoder_plan_region(decoder, &raw const options, &raw mut plan, &raw mut error),
            J2K_OK
        );
        assert_eq!(
            (
                plan.output.width,
                plan.output.height,
                plan.output.precision,
                plan.route
            ),
            (5, 5, 8, J2K_ROUTE_REGION_PLANES)
        );
        let mut output = vec![0; usize::try_from(plan.output.buffer_bytes).unwrap()];
        assert_eq!(
            j2k_decoder_decode_region_into(
                decoder,
                &raw const options,
                output.as_mut_ptr(),
                output.len(),
                usize::try_from(plan.output.row_bytes).unwrap(),
                &raw mut plan.output,
                &raw mut error
            ),
            J2K_OK
        );
        assert!(output.iter().all(|&sample| sample == 126));
        // Raw sRGB operations construct their decoder under the operation's
        // working limit and cancellation as well as its output limit.
        options.max_working_memory_bytes = 8 << 20;
        assert_eq!(
            j2k_decoder_plan_region(decoder, &raw const options, &raw mut plan, &raw mut error),
            J2K_OK
        );
        output.fill(0);
        assert_eq!(
            j2k_decoder_decode_region_into(
                decoder,
                &raw const options,
                output.as_mut_ptr(),
                output.len(),
                usize::try_from(plan.output.row_bytes).unwrap(),
                &raw mut plan.output,
                &raw mut error
            ),
            J2K_OK
        );
        assert!(output.iter().all(|&sample| sample == 126));
        options.max_working_memory_bytes = 1;
        assert_eq!(
            j2k_decoder_plan_region(decoder, &raw const options, &raw mut plan, &raw mut error),
            J2K_RESOURCE_LIMIT
        );
        output.fill(0);
        assert_eq!(
            j2k_decoder_decode_region_into(
                decoder,
                &raw const options,
                output.as_mut_ptr(),
                output.len(),
                usize::try_from(plan.output.row_bytes).unwrap(),
                ptr::null_mut(),
                &raw mut error
            ),
            J2K_RESOURCE_LIMIT
        );
        assert!(
            output.iter().all(|&sample| sample == 0),
            "no writes on failure"
        );
        options.max_working_memory_bytes = 0;
        let mut cancellation = ptr::null_mut();
        assert_eq!(
            j2k_cancellation_create(&raw mut cancellation, &raw mut error),
            J2K_OK
        );
        assert_eq!(
            j2k_cancellation_cancel(cancellation, &raw mut error),
            J2K_OK
        );
        options.cancellation = cancellation;
        assert_eq!(
            j2k_decoder_plan_region(decoder, &raw const options, &raw mut plan, &raw mut error),
            J2K_CANCELLED
        );
        options.cancellation = ptr::null();
        assert_eq!(
            j2k_cancellation_free(&raw mut cancellation, &raw mut error),
            J2K_OK
        );
        assert_eq!(
            j2k_decoder_plan_region(decoder, &raw const options, &raw mut plan, &raw mut error),
            J2K_OK,
            "a cancelled operation leaves the decoder usable"
        );
        assert_eq!(j2k_decoder_close(&raw mut decoder, &raw mut error), J2K_OK);
    }
}

#[test]
fn exact_planning_rejects_empty_coverage_and_forged_output_before_preparation() {
    let origin = include_bytes!(
        "../../j2k-test-support/fixtures/openjpeg_origin/rgb_23x17_origin_3_5_ht.j2c"
    );
    let forged = include_bytes!("fixtures/forged_tile_geometry.j2c");
    for (bytes, options, expected) in [
        (
            origin.as_slice(),
            J2kCOperationOptions {
                width: 1,
                height: 1,
                reduction: 1,
                ..Default::default()
            },
            J2K_UNSUPPORTED,
        ),
        (
            forged.as_slice(),
            J2kCOperationOptions::default(),
            J2K_RESOURCE_LIMIT,
        ),
    ] {
        let mut decoder = ptr::null_mut();
        let mut error = J2kCError::default();
        let mut plan = J2kCRegionPlan::default();
        // SAFETY: Independent live buffers and initialized structures; serial handle use.
        unsafe {
            assert_eq!(
                j2k_decoder_open(
                    bytes.as_ptr(),
                    bytes.len(),
                    ptr::null(),
                    &raw mut decoder,
                    &raw mut error
                ),
                J2K_OK
            );
            assert_eq!(
                j2k_decoder_plan_region(decoder, &raw const options, &raw mut plan, &raw mut error),
                expected
            );
            assert_eq!(j2k_decoder_close(&raw mut decoder, &raw mut error), J2K_OK);
        }
    }
}

#[test]
fn normalized_exact_decode_returns_actual_lenient_warnings() {
    let mut bytes = j2k::wrap_j2k_codestream(
        &constant_source(7, false, 63),
        j2k::J2kFileWrapOptions::jp2(),
    )
    .unwrap();
    bytes.extend_from_slice(&[0, 0, 0, 16, b'x', b'm', b'l', b' ']);
    let mut decoder = ptr::null_mut();
    let mut error = J2kCError::default();
    let open = J2kCOpenOptions {
        flags: J2K_OPEN_LENIENT,
        ..Default::default()
    };
    let options = J2kCOperationOptions {
        reduction: 4,
        interpretation: J2K_INTERPRETATION_SRGB8,
        ..Default::default()
    };
    let mut actual = J2kCOutputInfo::default();
    let mut output = [0_u8; 25];
    // SAFETY: Initialized, disjoint storage and a live decoder with serial access.
    unsafe {
        assert_eq!(
            j2k_decoder_open(
                bytes.as_ptr(),
                bytes.len(),
                &raw const open,
                &raw mut decoder,
                &raw mut error
            ),
            J2K_OK
        );
        assert_eq!(
            j2k_decoder_decode_region_into(
                decoder,
                &raw const options,
                output.as_mut_ptr(),
                output.len(),
                5,
                &raw mut actual,
                &raw mut error
            ),
            J2K_OK
        );
        assert_eq!(actual.warnings, J2K_WARNING_LENIENT_RECOVERY);
        assert_eq!(output, [126; 25]);
        assert_eq!(j2k_decoder_close(&raw mut decoder, &raw mut error), J2K_OK);
    }
}
