// SPDX-License-Identifier: MIT OR Apache-2.0

//! Numerical behavior of the library's native sample and resolution contracts.

#[path = "support/encode.rs"]
mod support;
use j2k_c::*;
use std::ptr;

struct Decoded {
    image: J2kCImageInfo,
    component: J2kCComponentInfo,
    layout: J2kCOutputInfo,
    samples: Vec<u8>,
}

/// Decode through an odd, padded stride; return tightly packed rows.
fn decode(bytes: &[u8], options: J2kCDecodeOptions) -> Decoded {
    let mut handle = ptr::null_mut();
    let mut image = J2kCImageInfo::default();
    let mut component = J2kCComponentInfo::default();
    let mut layout = J2kCOutputInfo::default();
    let mut error = J2kCError::default();
    // SAFETY: Each input/output allocation is valid and disjoint; handle use is serial.
    unsafe {
        assert_eq!(
            j2k_decoder_open(
                bytes.as_ptr(),
                bytes.len(),
                ptr::null(),
                &raw mut handle,
                &raw mut error
            ),
            J2K_OK
        );
        assert_eq!(
            j2k_decoder_info(handle, &raw mut image, &raw mut error),
            J2K_OK
        );
        assert_eq!(
            j2k_decoder_component_info(handle, 0, &raw mut component, &raw mut error),
            J2K_OK
        );
        assert_eq!(
            j2k_decoder_output_info(handle, &raw const options, &raw mut layout, &raw mut error),
            J2K_OK,
            "{}",
            std::ffi::CStr::from_ptr(error.message.as_ptr()).to_string_lossy()
        );
        let row = usize::try_from(layout.row_bytes).unwrap();
        let stride = row + 3;
        let mut output = vec![0xA5; stride * (layout.height as usize - 1) + row];
        assert_eq!(
            j2k_decoder_decode_into(
                handle,
                &raw const options,
                output.as_mut_ptr(),
                output.len(),
                stride,
                &raw mut error
            ),
            J2K_OK
        );
        assert_eq!(j2k_decoder_close(&raw mut handle, &raw mut error), J2K_OK);
        let mut samples = Vec::with_capacity(usize::try_from(layout.buffer_bytes).unwrap());
        for line in output.chunks(stride) {
            samples.extend_from_slice(&line[..row]);
            assert!(
                line[row..].iter().all(|&byte| byte == 0xA5),
                "stride padding was overwritten"
            );
        }
        Decoded {
            image,
            component,
            layout,
            samples,
        }
    }
}

#[test]
fn native_precision_signedness_and_rgb_order_are_preserved() {
    for ht in [false, true] {
        for (channels, bits, signed) in [
            (3, 8, false),
            (1, 12, false),
            (3, 16, false),
            (1, 8, true),
            (1, 16, true),
            (3, 12, true),
        ] {
            let mut input = Vec::new();
            let mut expected = Vec::new();
            for i in 0..32 * 24 * channels {
                let value: i32 =
                    ((i * 37 + 11) % (1 << bits)) - if signed { 1 << (bits - 1) } else { 0 };
                if bits <= 8 {
                    input.push(value.to_le_bytes()[0]);
                } else {
                    input.extend_from_slice(&value.to_le_bytes()[..2]);
                }
                if signed || bits > 8 {
                    expected.extend_from_slice(&value.to_le_bytes()[..2]);
                } else {
                    expected.push(value.to_le_bytes()[0]);
                }
            }
            let encoded = support::encode(
                &input,
                32,
                24,
                u16::try_from(channels).unwrap(),
                bits,
                signed,
                ht,
            );
            let decoded = decode(&encoded, J2kCDecodeOptions::default());
            assert_eq!(
                (decoded.layout.channels, decoded.layout.precision),
                (u32::try_from(channels).unwrap(), u32::from(bits))
            );
            assert_eq!(
                (
                    decoded.image.has_signed_components,
                    decoded.component.is_signed,
                    decoded.component.precision
                ),
                (u32::from(signed), u32::from(signed), u32::from(bits))
            );
            assert_eq!(
                decoded.layout.sample_type,
                if signed {
                    J2K_I16_LE
                } else if bits > 8 {
                    J2K_U16_LE
                } else {
                    J2K_U8
                }
            );
            assert_eq!(
                decoded.samples, expected,
                "ht={ht}, channels={channels}, bits={bits}, signed={signed}"
            );
        }
    }
}

#[test]
fn wavelet_reduction_uses_covering_geometry_and_preserves_a_constant_image() {
    // J2K deliberately disables wavelets below 64 pixels on either axis.
    let pixels = [42_u8; 64 * 64];
    for ht in [false, true] {
        let encoded = support::encode(&pixels, 64, 64, 1, 8, false, ht);
        let options = J2kCDecodeOptions {
            x: 3,
            y: 5,
            width: 7,
            height: 6,
            reduction: 1,
            ..Default::default()
        };
        let decoded = decode(&encoded, options);
        assert_eq!(decoded.image.resolution_levels, 4);
        let layout = decoded.layout;
        assert_eq!(
            (layout.x, layout.y, layout.width, layout.height),
            (1, 2, 4, 4)
        );
        assert_eq!(decoded.samples, [42; 16]);
        for reduction in 1..=3 {
            let decoded = decode(
                &encoded,
                J2kCDecodeOptions {
                    reduction,
                    ..Default::default()
                },
            );
            assert_eq!(
                (decoded.layout.width, decoded.layout.height),
                (64 >> reduction, 64 >> reduction)
            );
            assert!(decoded.samples.iter().all(|&value| value == 42));
        }
    }
}

#[test]
fn jp2_and_jph_wrappers_preserve_rgba_samples_and_alpha_metadata() {
    use j2k::{
        J2kChannelAssociation, J2kChannelDefinition, J2kChannelType, J2kFileBoxMetadata,
        J2kFileWrapOptions,
    };
    let pixels: Vec<u8> = (0..32 * 24)
        .flat_map(|i| {
            [
                u8::try_from(i % 128).unwrap(),
                u8::try_from((i * 2) % 128).unwrap(),
                u8::try_from((i * 3) % 128).unwrap(),
                200,
            ]
        })
        .collect();
    for ht in [false, true] {
        let encoded = support::encode(&pixels, 32, 24, 4, 8, false, ht);
        for premultiplied in [false, true] {
            let mut definitions: Vec<_> = (0..3)
                .map(|index| J2kChannelDefinition {
                    channel_index: index,
                    channel_type: J2kChannelType::Color,
                    association: J2kChannelAssociation::Color { index: index + 1 },
                })
                .collect();
            definitions.push(J2kChannelDefinition {
                channel_index: 3,
                channel_type: if premultiplied {
                    J2kChannelType::PremultipliedOpacity
                } else {
                    J2kChannelType::Opacity
                },
                association: J2kChannelAssociation::WholeImage,
            });
            let options = if ht {
                J2kFileWrapOptions::jph()
            } else {
                J2kFileWrapOptions::jp2()
            }
            .with_metadata(J2kFileBoxMetadata {
                channel_definitions: &definitions,
                ..J2kFileBoxMetadata::empty()
            });
            let wrapped = j2k::wrap_j2k_codestream(&encoded, options).unwrap();
            let decoded = decode(&wrapped, J2kCDecodeOptions::default());
            assert_eq!(decoded.layout.channels, 4);
            assert_eq!(
                decoded.layout.alpha,
                if premultiplied {
                    J2K_ALPHA_PREMULTIPLIED
                } else {
                    J2K_ALPHA_STRAIGHT
                }
            );
            assert_eq!(decoded.samples, pixels);
        }
    }
}

/// An image whose reference-grid origin is odd at every resolution reduces to
/// the dimensions and samples `OpenJPEG` reports: each reduction covers
/// `ceil(origin / 2^r)..ceil((origin + size) / 2^r)`, not `ceil(size / 2^r)`.
#[test]
fn odd_origin_reductions_match_openjpeg_dimensions_and_samples() {
    use j2k_test_support::{
        OPENJPEG_ORIGIN_23X17_AT_3_5, OPENJPH_ORIGIN_23X17_AT_3_5, ORIGIN_REDUCED_RGB,
        ORIGIN_SOURCE_23X17_RGB,
    };
    for codestream in [OPENJPEG_ORIGIN_23X17_AT_3_5, OPENJPH_ORIGIN_23X17_AT_3_5] {
        let full = decode(codestream, J2kCDecodeOptions::default());
        assert_eq!((full.layout.width, full.layout.height), (23, 17));
        assert_eq!(full.samples, ORIGIN_SOURCE_23X17_RGB);
        for (reduction, ((width, height), expected)) in (1..).zip(ORIGIN_REDUCED_RGB) {
            let options = J2kCDecodeOptions {
                reduction,
                ..J2kCDecodeOptions::default()
            };
            let decoded = decode(codestream, options);
            assert_eq!(
                (decoded.layout.x, decoded.layout.y),
                (0, 0),
                "reduction {reduction}"
            );
            assert_eq!(
                (decoded.layout.width, decoded.layout.height),
                (width, height),
                "reduction {reduction}"
            );
            assert_eq!(decoded.samples, expected, "reduction {reduction}");
        }
    }
}
