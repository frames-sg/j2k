// SPDX-License-Identifier: Apache-2.0

use signinum_transcode::htj2k_wavelet::{
    ComponentSampling, WaveletBand53, WaveletComponent53, WaveletImage53, WaveletLevel53,
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

fn band(width: usize, height: usize) -> WaveletBand53<i32> {
    WaveletBand53 {
        width,
        height,
        coefficients: vec![0; width * height],
    }
}
