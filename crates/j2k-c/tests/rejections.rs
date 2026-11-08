// SPDX-License-Identifier: MIT OR Apache-2.0

//! Invalid caller requests and malformed images must fail without partial output.

mod support;
use j2k_c::*;
use std::{
    mem::{offset_of, size_of},
    ptr,
    time::{Duration, Instant},
};

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one rejection table per ABI entry point"
)]
fn null_arguments_versions_reserved_fields_and_indices_are_rejected() {
    let (encoded, _) = support::gray_fixture(false);
    let mut error = J2kCError::default();
    // Every failed open must clear a slot that still holds a stale pointer.
    let failed_open = |input: *const u8, options: *const J2kCOpenOptions, error: *mut J2kCError| {
        let mut handle = ptr::NonNull::<J2kCDecoder>::dangling().as_ptr();
        // SAFETY: Input is null or covers `encoded.len()` bytes; the slot is live and disjoint.
        let status =
            unsafe { j2k_decoder_open(input, encoded.len(), options, &raw mut handle, error) };
        assert!(handle.is_null(), "status {status} left the output slot set");
        status
    };
    assert_eq!(
        failed_open(ptr::null(), ptr::null(), &raw mut error),
        J2K_INVALID_ARGUMENT
    );
    let mut options = J2kCOpenOptions::default();
    options.header.abi_version = 99;
    assert_eq!(
        failed_open(encoded.as_ptr(), &raw const options, &raw mut error),
        J2K_ABI_MISMATCH
    );
    options = J2kCOpenOptions::default();
    options.header.struct_size -= 1;
    assert_eq!(
        failed_open(encoded.as_ptr(), &raw const options, &raw mut error),
        J2K_ABI_MISMATCH
    );
    options = J2kCOpenOptions::default();
    options.flags = 2;
    assert_eq!(
        failed_open(encoded.as_ptr(), &raw const options, &raw mut error),
        J2K_INVALID_ARGUMENT
    );
    options = J2kCOpenOptions::default();
    options.workers = 65;
    assert_eq!(
        failed_open(encoded.as_ptr(), &raw const options, &raw mut error),
        J2K_INVALID_ARGUMENT
    );
    // A misaligned error object cannot receive a diagnostic, but the slot is still cleared.
    let mut storage = [0_u32; size_of::<J2kCError>() / 4 + 1];
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the test passes a deliberately misaligned diagnostic pointer"
    )]
    let misaligned = storage
        .as_mut_ptr()
        .cast::<u8>()
        .wrapping_add(1)
        .cast::<J2kCError>();
    assert_eq!(
        failed_open(encoded.as_ptr(), ptr::null(), misaligned),
        J2K_INVALID_ARGUMENT
    );
    options = J2kCOpenOptions::default();
    options.workers = 64;
    let mut handle = ptr::null_mut();
    // SAFETY: Null arguments are explicitly checked; all other storage is live and disjoint.
    unsafe {
        assert_eq!(
            j2k_decoder_open(
                encoded.as_ptr(),
                encoded.len(),
                &raw const options,
                &raw mut handle,
                &raw mut error
            ),
            J2K_OK
        );
        let mut info = J2kCImageInfo::default();
        info.header.struct_size -= 4;
        assert_eq!(
            j2k_decoder_info(handle, &raw mut info, &raw mut error),
            J2K_ABI_MISMATCH
        );
        assert_eq!(info.width, 0);
        info = J2kCImageInfo::default();
        assert_eq!(
            j2k_decoder_info(handle, &raw mut info, &raw mut error),
            J2K_OK
        );
        assert_eq!(
            (
                info.width,
                info.height,
                info.components,
                info.resolution_levels,
                info.has_signed_components
            ),
            (32, 24, 1, 1, 0)
        );
        let mut component = J2kCComponentInfo::default();
        component.header.struct_size -= 4;
        assert_eq!(
            j2k_decoder_component_info(handle, 0, &raw mut component, &raw mut error),
            J2K_ABI_MISMATCH
        );
        assert_eq!(component.precision, 0);
        component = J2kCComponentInfo::default();
        assert_eq!(
            j2k_decoder_component_info(handle, 0, &raw mut component, &raw mut error),
            J2K_OK
        );
        assert_eq!(
            (
                component.precision,
                component.is_signed,
                component.sampling_x,
                component.sampling_y
            ),
            (8, 0, 1, 1)
        );
        assert_eq!(
            j2k_decoder_component_info(handle, 1, &raw mut component, &raw mut error),
            J2K_INVALID_ARGUMENT
        );
        assert_eq!(component.precision, 8);
        let mut layout = J2kCOutputInfo::default();
        layout.header.struct_size -= 4;
        assert_eq!(
            j2k_decoder_output_info(handle, ptr::null(), &raw mut layout, &raw mut error),
            J2K_ABI_MISMATCH
        );
        assert_eq!(layout.width, 0);
        assert_eq!(j2k_decoder_close(&raw mut handle, &raw mut error), J2K_OK);
        assert_eq!(j2k_decoder_close(&raw mut handle, &raw mut error), J2K_OK);
        assert_eq!(
            j2k_decoder_info(handle, &raw mut info, &raw mut error),
            J2K_INVALID_ARGUMENT
        );
        assert_eq!(error.message[255], 0);
    }
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one table-driven sweep of limits and strides"
)]
fn resource_limits_geometry_and_stride_overflow_are_checked_before_output() {
    let (encoded, _) = support::gray_fixture(true);
    let mut error = J2kCError::default();
    let mut handle = ptr::null_mut();
    let mut options = J2kCOpenOptions {
        max_input_bytes: encoded.len() as u64 - 1,
        ..Default::default()
    };
    // SAFETY: All input, option, handle, and output allocations are live and disjoint.
    unsafe {
        assert_eq!(
            j2k_decoder_open(
                encoded.as_ptr(),
                encoded.len(),
                &raw const options,
                &raw mut handle,
                &raw mut error
            ),
            J2K_RESOURCE_LIMIT
        );
        let zeros = vec![0_u8; (512 << 20) + 1];
        for (length, expected) in [
            (zeros.len() - 1, J2K_INVALID_INPUT),
            (zeros.len(), J2K_RESOURCE_LIMIT),
        ] {
            assert_eq!(
                j2k_decoder_open(
                    zeros.as_ptr(),
                    length,
                    ptr::null(),
                    &raw mut handle,
                    &raw mut error
                ),
                expected
            );
        }
        options.max_input_bytes = encoded.len() as u64;
        options.max_output_bytes = 767;
        assert_eq!(
            j2k_decoder_open(
                encoded.as_ptr(),
                encoded.len(),
                &raw const options,
                &raw mut handle,
                &raw mut error
            ),
            J2K_OK
        );
        let mut info = J2kCOutputInfo::default();
        assert_eq!(
            j2k_decoder_output_info(handle, ptr::null(), &raw mut info, &raw mut error),
            J2K_RESOURCE_LIMIT
        );
        let reduced = J2kCDecodeOptions {
            reduction: 1,
            ..Default::default()
        };
        assert_eq!(
            j2k_decoder_output_info(handle, &raw const reduced, &raw mut info, &raw mut error),
            J2K_UNSUPPORTED
        );
        assert_eq!(j2k_decoder_close(&raw mut handle, &raw mut error), J2K_OK);
        options.max_output_bytes = 768;
        assert_eq!(
            j2k_decoder_open(
                encoded.as_ptr(),
                encoded.len(),
                &raw const options,
                &raw mut handle,
                &raw mut error
            ),
            J2K_OK
        );
        assert_eq!(
            j2k_decoder_output_info(handle, ptr::null(), &raw mut info, &raw mut error),
            J2K_OK
        );
        for (request, expected) in [
            (
                J2kCDecodeOptions {
                    x: 1,
                    ..Default::default()
                },
                J2K_INVALID_ARGUMENT,
            ),
            (
                J2kCDecodeOptions {
                    x: u32::MAX,
                    width: 2,
                    height: 1,
                    ..Default::default()
                },
                J2K_INVALID_ARGUMENT,
            ),
            (
                J2kCDecodeOptions {
                    width: 7,
                    ..Default::default()
                },
                J2K_INVALID_ARGUMENT,
            ),
            (
                J2kCDecodeOptions {
                    x: 16,
                    y: 12,
                    width: 16,
                    height: 12,
                    ..Default::default()
                },
                J2K_OK,
            ),
            (
                J2kCDecodeOptions {
                    reserved: 1,
                    ..Default::default()
                },
                J2K_INVALID_ARGUMENT,
            ),
            // The 32x24 fixture has no wavelet levels to discard.
            (
                J2kCDecodeOptions {
                    reduction: 1,
                    ..Default::default()
                },
                J2K_UNSUPPORTED,
            ),
            (
                J2kCDecodeOptions {
                    reduction: 4,
                    ..Default::default()
                },
                J2K_UNSUPPORTED,
            ),
        ] {
            assert_eq!(
                j2k_decoder_output_info(handle, &raw const request, &raw mut info, &raw mut error),
                expected,
                "{request:?}"
            );
        }
        let mut output = [0xA5; 768];
        // Stride 33 needs 33 * 23 + 32 = 791 bytes, above max_output_bytes.
        for (length, stride, expected) in [
            (output.len(), usize::MAX, J2K_RESOURCE_LIMIT),
            (output.len(), 33, J2K_RESOURCE_LIMIT),
            (output.len(), 31, J2K_INVALID_ARGUMENT),
            (usize::MAX, 32, J2K_INVALID_ARGUMENT),
            (isize::MAX as usize + 1, 32, J2K_INVALID_ARGUMENT),
        ] {
            assert_eq!(
                j2k_decoder_decode_into(
                    handle,
                    ptr::null(),
                    output.as_mut_ptr(),
                    length,
                    stride,
                    &raw mut error
                ),
                expected,
                "length {length}, stride {stride}"
            );
        }
        assert!(output.iter().all(|&value| value == 0xA5));
        assert_eq!(
            j2k_decoder_decode_into(
                handle,
                ptr::null(),
                output.as_mut_ptr(),
                output.len(),
                32,
                &raw mut error
            ),
            J2K_OK
        );
        assert_eq!(j2k_decoder_close(&raw mut handle, &raw mut error), J2K_OK);
        // A fuzzed 2 KiB codestream declaring a 27248 x 840957983 image in 64-pixel-wide
        // tiles. Codec planning once spent about 30 s of CPU per call before the limit applied.
        let forged = include_bytes!("fixtures/forged_tile_geometry.j2c");
        output.fill(0xA5);
        assert_eq!(
            j2k_decoder_open(
                forged.as_ptr(),
                forged.len(),
                ptr::null(),
                &raw mut handle,
                &raw mut error
            ),
            J2K_OK
        );
        let started = Instant::now();
        assert_eq!(
            j2k_decoder_output_info(handle, ptr::null(), &raw mut info, &raw mut error),
            J2K_RESOURCE_LIMIT
        );
        assert_eq!(
            j2k_decoder_decode_into(
                handle,
                ptr::null(),
                output.as_mut_ptr(),
                output.len(),
                64,
                &raw mut error
            ),
            J2K_RESOURCE_LIMIT
        );
        let tall = J2kCDecodeOptions {
            width: 27248,
            height: 500_000_000,
            ..Default::default()
        };
        assert_eq!(
            j2k_decoder_output_info(handle, &raw const tall, &raw mut info, &raw mut error),
            J2K_RESOURCE_LIMIT
        );
        assert!(started.elapsed() < Duration::from_secs(2));
        // A small region of a huge declared image remains a valid request.
        let region = J2kCDecodeOptions {
            width: 64,
            height: 64,
            ..Default::default()
        };
        assert_eq!(
            j2k_decoder_output_info(handle, &raw const region, &raw mut info, &raw mut error),
            J2K_OK
        );
        assert_eq!((info.width, info.height), (64, 64));
        assert!(output.iter().all(|&value| value == 0xA5));
        assert_eq!(j2k_decoder_close(&raw mut handle, &raw mut error), J2K_OK);
    }
}

#[test]
fn truncated_and_foreign_inputs_fail_with_documented_statuses_and_no_output() {
    let (encoded, _) = support::gray_fixture(true);
    let foreign = [0x5A; 16];
    let mut no_components = encoded.clone();
    no_components[40..42].fill(0); // SIZ Csiz = 0
                                   // J2K 0.13.1 reports missing tile data as an unclassified backend failure,
                                   // which docs/api.md maps to J2K_DECODE_FAILED.
    for (input, opened, decoded) in [
        (&encoded[..0], J2K_INVALID_INPUT, None),
        (&encoded[..1], J2K_INVALID_INPUT, None),
        (&encoded[..20], J2K_INVALID_INPUT, None),
        (
            &encoded[..encoded.len() / 2],
            J2K_OK,
            Some(J2K_DECODE_FAILED),
        ),
        (
            &encoded[..encoded.len() - 8],
            J2K_OK,
            Some(J2K_DECODE_FAILED),
        ),
        (&foreign[..], J2K_INVALID_INPUT, None),
        (&no_components[..], J2K_INVALID_INPUT, None),
    ] {
        let mut handle = ptr::null_mut();
        let mut error = J2kCError::default();
        let mut output = [0xA5; 768];
        // SAFETY: Input allocation covers every tested prefix; outputs are valid and disjoint.
        unsafe {
            assert_eq!(
                j2k_decoder_open(
                    input.as_ptr(),
                    input.len(),
                    ptr::null(),
                    &raw mut handle,
                    &raw mut error,
                ),
                opened,
                "open of {} bytes",
                input.len()
            );
            assert_eq!(handle.is_null(), opened != J2K_OK);
            if let Some(expected) = decoded {
                assert_eq!(
                    j2k_decoder_decode_into(
                        handle,
                        ptr::null(),
                        output.as_mut_ptr(),
                        output.len(),
                        32,
                        &raw mut error
                    ),
                    expected,
                    "decode of {} bytes",
                    input.len()
                );
            }
            assert_eq!(output, [0xA5; 768]);
            assert_eq!(j2k_decoder_close(&raw mut handle, ptr::null_mut()), J2K_OK);
        }
    }
}

const HEADER: &str = include_str!("../include/j2k.h");

/// Field names and C offsets of `typedef struct name { ... } name;` in the header.
fn c_layout(name: &str) -> (Vec<(String, usize)>, usize) {
    let start = HEADER
        .find(&format!("typedef struct {name} {{"))
        .unwrap_or_else(|| panic!("{name} is not declared in j2k.h"));
    let body = &HEADER[start..];
    let body = &body[body.find('{').unwrap() + 1..body.find('}').unwrap()];
    let (mut fields, mut offset, mut max_align) = (Vec::new(), 0_usize, 1);
    for declaration in body.lines().filter_map(|line| line.split_once(';')) {
        let mut words = declaration.0.split_whitespace();
        let (ty, field) = (words.next().unwrap(), words.next().unwrap());
        let (field, size, align) = match (ty, field.split_once('[')) {
            ("char", Some((field, length))) => {
                (field, length[..length.len() - 1].parse().unwrap(), 1)
            }
            ("uint32_t", None) => (field, 4, 4),
            ("uint64_t", None) => (field, 8, 8),
            ("j2k_header", None) => (field, 8, 4),
            ("j2k_output_info", None) => (field, 64, 8),
            ("const", None) if field == "j2k_cancellation" => (
                words.next().unwrap().trim_start_matches('*'),
                size_of::<*const J2kCCancellation>(),
                std::mem::align_of::<*const J2kCCancellation>(),
            ),
            _ => panic!("unexpected {name} field: {}", declaration.0),
        };
        offset = offset.next_multiple_of(align);
        fields.push((field.to_owned(), offset));
        offset += size;
        max_align = max_align.max(align);
    }
    (fields, offset.next_multiple_of(max_align))
}

macro_rules! rust_layout {
    ($ty:ty { $($field:ident),* }) => {
        (vec![$((stringify!($field).to_owned(), offset_of!($ty, $field))),*], size_of::<$ty>())
    };
}

/// An ABI-1 image info followed by storage from a newer, larger revision.
#[repr(C)]
struct ExtendedInfo {
    info: J2kCImageInfo,
    tail: [u32; 4],
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one table of every ABI-1 constant and layout"
)]
fn abi_one_layout_matches_the_c_header_and_accepts_extension_storage() {
    assert_eq!(size_of::<J2kCOpenOptions>(), 32);
    assert_eq!(size_of::<J2kCDecodeOptions>(), 32);
    assert_eq!(size_of::<J2kCImageInfo>(), 56);
    assert_eq!(size_of::<J2kCComponentInfo>(), 24);
    assert_eq!(size_of::<J2kCOutputInfo>(), 64);
    assert_eq!(size_of::<J2kCEncodeOptions>(), 32);
    assert_eq!(size_of::<J2kCError>(), 260);
    let defined: Vec<(&str, u32)> = HEADER
        .lines()
        .filter_map(|line| {
            let (name, value) = line.strip_prefix("#define ")?.split_once(" UINT32_C(")?;
            Some((name, value.strip_suffix(')')?.parse().unwrap()))
        })
        .collect();
    assert_eq!(
        defined,
        [
            ("J2K_ABI_VERSION", J2K_ABI_VERSION),
            ("J2K_OK", J2K_OK),
            ("J2K_INVALID_ARGUMENT", J2K_INVALID_ARGUMENT),
            ("J2K_ABI_MISMATCH", J2K_ABI_MISMATCH),
            ("J2K_INVALID_INPUT", J2K_INVALID_INPUT),
            ("J2K_UNSUPPORTED", J2K_UNSUPPORTED),
            ("J2K_BUFFER_TOO_SMALL", J2K_BUFFER_TOO_SMALL),
            ("J2K_RESOURCE_LIMIT", J2K_RESOURCE_LIMIT),
            ("J2K_ALLOCATION_FAILED", J2K_ALLOCATION_FAILED),
            ("J2K_INTERNAL_ERROR", J2K_INTERNAL_ERROR),
            ("J2K_PANIC", J2K_PANIC),
            ("J2K_DECODE_FAILED", J2K_DECODE_FAILED),
            ("J2K_STOPPED", J2K_STOPPED),
            ("J2K_ENCODE_FAILED", J2K_ENCODE_FAILED),
            ("J2K_CANCELLED", J2K_CANCELLED),
            ("J2K_INTERPRETATION_NATIVE", J2K_INTERPRETATION_NATIVE),
            ("J2K_INTERPRETATION_SRGB8", J2K_INTERPRETATION_SRGB8),
            ("J2K_ROUTE_NATIVE_CPU", J2K_ROUTE_NATIVE_CPU),
            ("J2K_ROUTE_REGION_PLANES", J2K_ROUTE_REGION_PLANES),
            ("J2K_OPEN_LENIENT", J2K_OPEN_LENIENT),
            ("J2K_WARNING_LENIENT_RECOVERY", J2K_WARNING_LENIENT_RECOVERY),
            ("J2K_ENCODE_HTJ2K", J2K_ENCODE_HTJ2K),
            ("J2K_ENCODE_SIGNED", J2K_ENCODE_SIGNED),
            ("J2K_ENCODE_FILE", J2K_ENCODE_FILE),
            ("J2K_U8", J2K_U8),
            ("J2K_U16_LE", J2K_U16_LE),
            ("J2K_I16_LE", J2K_I16_LE),
            ("J2K_ALPHA_NONE", J2K_ALPHA_NONE),
            ("J2K_ALPHA_STRAIGHT", J2K_ALPHA_STRAIGHT),
            ("J2K_ALPHA_PREMULTIPLIED", J2K_ALPHA_PREMULTIPLIED),
        ]
    );
    for (name, rust) in [
        (
            "j2k_header",
            rust_layout!(J2kCHeader {
                struct_size,
                abi_version
            }),
        ),
        ("j2k_error", rust_layout!(J2kCError { code, message })),
        (
            "j2k_open_options",
            rust_layout!(J2kCOpenOptions {
                header,
                max_input_bytes,
                max_output_bytes,
                workers,
                flags
            }),
        ),
        (
            "j2k_decode_options",
            rust_layout!(J2kCDecodeOptions {
                header,
                x,
                y,
                width,
                height,
                reduction,
                reserved
            }),
        ),
        (
            "j2k_image_info",
            rust_layout!(J2kCImageInfo {
                header,
                width,
                height,
                components,
                resolution_levels,
                has_signed_components,
                has_icc_profile,
                image_x0,
                image_y0,
                tile_x0,
                tile_y0,
                tile_width,
                tile_height
            }),
        ),
        (
            "j2k_operation_options",
            rust_layout!(J2kCOperationOptions {
                header,
                x,
                y,
                width,
                height,
                reduction,
                interpretation,
                max_working_memory_bytes,
                max_output_bytes,
                cancellation
            }),
        ),
        (
            "j2k_region_plan",
            rust_layout!(J2kCRegionPlan {
                header,
                output,
                source_x,
                source_y,
                source_width,
                source_height,
                reduction,
                max_reduction,
                interpretation,
                route
            }),
        ),
        (
            "j2k_component_info",
            rust_layout!(J2kCComponentInfo {
                header,
                precision,
                is_signed,
                sampling_x,
                sampling_y
            }),
        ),
        (
            "j2k_output_info",
            rust_layout!(J2kCOutputInfo {
                header,
                x,
                y,
                width,
                height,
                channels,
                sample_type,
                precision,
                alpha,
                row_bytes,
                buffer_bytes,
                warnings,
                reserved
            }),
        ),
        (
            "j2k_encode_options",
            rust_layout!(J2kCEncodeOptions {
                header,
                width,
                height,
                components,
                bit_depth,
                flags,
                reserved
            }),
        ),
    ] {
        assert_eq!(c_layout(name), rust, "{name} differs between C and Rust");
    }
    let mut output = ExtendedInfo {
        info: J2kCImageInfo::default(),
        tail: [0x1234_5678; 4],
    };
    output.info.header.struct_size = u32::try_from(size_of::<ExtendedInfo>()).unwrap();
    let (encoded, _) = support::gray_fixture(false);
    let mut handle = ptr::null_mut();
    // SAFETY: The ABI prefix is part of a larger live allocation and all arguments are disjoint.
    unsafe {
        assert_eq!(
            j2k_decoder_open(
                encoded.as_ptr(),
                encoded.len(),
                ptr::null(),
                &raw mut handle,
                ptr::null_mut()
            ),
            J2K_OK
        );
        assert_eq!(
            j2k_decoder_info(handle, &raw mut output.info, ptr::null_mut()),
            J2K_OK
        );
        assert_eq!(output.tail, [0x1234_5678; 4]);
        assert_eq!(j2k_decoder_close(&raw mut handle, ptr::null_mut()), J2K_OK);
    }
}

#[test]
fn valid_high_precision_input_is_inspectable_but_not_silently_narrowed() {
    let pixels: Vec<u8> = (0..32 * 24)
        .flat_map(|i| {
            let bytes = u32::try_from(i * 331).unwrap().to_le_bytes();
            [bytes[0], bytes[1], bytes[2]]
        })
        .collect();
    let encoded = support::encode(&pixels, 32, 24, 1, 18, false, false);
    let mut handle = ptr::null_mut();
    let mut component = J2kCComponentInfo::default();
    let mut output = J2kCOutputInfo::default();
    // SAFETY: Disjoint, initialized output allocations and one exclusively used handle.
    unsafe {
        assert_eq!(
            j2k_decoder_open(
                encoded.as_ptr(),
                encoded.len(),
                ptr::null(),
                &raw mut handle,
                ptr::null_mut()
            ),
            J2K_OK
        );
        assert_eq!(
            j2k_decoder_component_info(handle, 0, &raw mut component, ptr::null_mut()),
            J2K_OK
        );
        assert_eq!(component.precision, 18);
        // ICC-tagged input is inspectable and flagged, but pixel decode rejects it.
        let mut info = J2kCImageInfo::default();
        let icc = j2k::wrap_j2k_codestream(
            &support::gray_fixture(false).0,
            j2k::J2kFileWrapOptions::jp2().with_color(j2k::J2kFileColorSpec::IccProfile(&[0; 128])),
        )
        .unwrap();
        let mut icc_handle = ptr::null_mut();
        assert_eq!(
            j2k_decoder_open(
                icc.as_ptr(),
                icc.len(),
                ptr::null(),
                &raw mut icc_handle,
                ptr::null_mut()
            ),
            J2K_OK
        );
        assert_eq!(
            j2k_decoder_info(icc_handle, &raw mut info, ptr::null_mut()),
            J2K_OK
        );
        assert_eq!(info.has_icc_profile, 1);
        assert_eq!(
            j2k_decoder_output_info(icc_handle, ptr::null(), &raw mut output, ptr::null_mut()),
            J2K_UNSUPPORTED
        );
        assert_eq!(
            j2k_decoder_close(&raw mut icc_handle, ptr::null_mut()),
            J2K_OK
        );
        assert_eq!(
            j2k_decoder_output_info(handle, ptr::null(), &raw mut output, ptr::null_mut()),
            J2K_UNSUPPORTED
        );
        assert_eq!(j2k_decoder_close(&raw mut handle, ptr::null_mut()), J2K_OK);
    }
}
