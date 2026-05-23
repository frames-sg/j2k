// SPDX-License-Identifier: Apache-2.0

use signinum_j2k_native::{encode_precomputed_htj2k_53, DecodeSettings, EncodeOptions, Image};
use signinum_transcode::htj2k_wavelet::{
    ComponentSampling, WaveletBand53, WaveletComponent53, WaveletImage53, WaveletLevel53,
    WaveletToPrecomputedError,
};

#[test]
fn wavelet_image_validates_one_level_band_dimensions() {
    let component = WaveletComponent53 {
        width: 8,
        height: 8,
        bit_depth: 8,
        is_signed: false,
        sampling: ComponentSampling {
            x_rsiz: 1,
            y_rsiz: 1,
        },
        final_ll: band(4, 4),
        levels: vec![WaveletLevel53 {
            hl: band(4, 4),
            lh: band(4, 4),
            hh: band(4, 4),
        }],
    };
    let image = WaveletImage53 {
        components: vec![component],
    };

    image.validate().expect("valid one-level geometry");
}

#[test]
fn wavelet_image_validates_multilevel_band_dimensions() {
    let component = WaveletComponent53 {
        width: 8,
        height: 8,
        bit_depth: 8,
        is_signed: false,
        sampling: ComponentSampling {
            x_rsiz: 2,
            y_rsiz: 2,
        },
        final_ll: band(2, 2),
        levels: vec![
            WaveletLevel53 {
                hl: band(4, 4),
                lh: band(4, 4),
                hh: band(4, 4),
            },
            WaveletLevel53 {
                hl: band(2, 2),
                lh: band(2, 2),
                hh: band(2, 2),
            },
        ],
    };
    let image = WaveletImage53 {
        components: vec![component],
    };

    image.validate().expect("valid multilevel geometry");
}

#[test]
fn wavelet_image_rejects_invalid_band_lengths() {
    let component = WaveletComponent53 {
        width: 8,
        height: 8,
        bit_depth: 8,
        is_signed: false,
        sampling: ComponentSampling {
            x_rsiz: 1,
            y_rsiz: 1,
        },
        final_ll: band(4, 4),
        levels: vec![WaveletLevel53 {
            hl: WaveletBand53 {
                width: 4,
                height: 4,
                coefficients: vec![0; 15],
            },
            lh: band(4, 4),
            hh: band(4, 4),
        }],
    };
    let image = WaveletImage53 {
        components: vec![component],
    };

    assert!(image.validate().is_err());
}

#[test]
fn wavelet_image_converts_to_encodable_precomputed_htj2k_image() {
    let image = WaveletImage53 {
        components: vec![
            component(16, 16, 1, 1),
            component(8, 8, 2, 2),
            component(8, 8, 2, 2),
        ],
    };

    let precomputed = image
        .to_precomputed_htj2k_53(16, 16)
        .expect("convert wavelet image to precomputed HTJ2K");

    assert_eq!((precomputed.width, precomputed.height), (16, 16));
    assert_eq!(precomputed.bit_depth, 8);
    assert!(!precomputed.signed);
    assert_eq!(precomputed.components.len(), 3);
    assert_eq!(
        precomputed
            .components
            .iter()
            .map(|component| (component.x_rsiz, component.y_rsiz))
            .collect::<Vec<_>>(),
        vec![(1, 1), (2, 2), (2, 2)]
    );

    let bytes = encode_precomputed_htj2k_53(&precomputed, &precomputed_options())
        .expect("converted wavelet image encodes as HTJ2K");
    let decoded = Image::new(&bytes, &DecodeSettings::default())
        .expect("native parser accepts converted HTJ2K")
        .decode_native()
        .expect("native decoder accepts converted HTJ2K");

    assert_eq!((decoded.width, decoded.height), (16, 16));
    assert_eq!(decoded.num_components, 3);
}

#[test]
fn wavelet_image_precomputed_conversion_rejects_reference_grid_mismatch() {
    let image = WaveletImage53 {
        components: vec![component(8, 8, 1, 1)],
    };

    let err = image
        .to_precomputed_htj2k_53(16, 16)
        .expect_err("component dimensions must match reference grid sampling");

    assert!(matches!(
        err,
        WaveletToPrecomputedError::ComponentGeometry {
            component: 0,
            expected_width: 16,
            expected_height: 16,
            actual_width: 8,
            actual_height: 8,
        }
    ));
}

fn band(width: usize, height: usize) -> WaveletBand53<i32> {
    WaveletBand53 {
        width,
        height,
        coefficients: vec![0; width * height],
    }
}

fn component(width: usize, height: usize, x_rsiz: u16, y_rsiz: u16) -> WaveletComponent53<i32> {
    WaveletComponent53 {
        width,
        height,
        bit_depth: 8,
        is_signed: false,
        sampling: ComponentSampling { x_rsiz, y_rsiz },
        final_ll: band(width.div_ceil(2), height.div_ceil(2)),
        levels: vec![WaveletLevel53 {
            hl: band(width / 2, height.div_ceil(2)),
            lh: band(width.div_ceil(2), height / 2),
            hh: band(width / 2, height / 2),
        }],
    }
}

fn precomputed_options() -> EncodeOptions {
    EncodeOptions {
        num_decomposition_levels: 1,
        reversible: true,
        use_ht_block_coding: true,
        use_mct: false,
        validate_high_throughput_codestream: false,
        ..EncodeOptions::default()
    }
}
