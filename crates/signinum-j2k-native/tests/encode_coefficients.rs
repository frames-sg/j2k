use signinum_j2k_native::{
    encode_precomputed_htj2k_53, DecodeSettings, EncodeOptions, Image, J2kForwardDwt53Level,
    J2kForwardDwt53Output, PrecomputedHtj2k53Component, PrecomputedHtj2k53Image,
};

#[test]
fn precomputed_zero_grayscale_53_coefficients_decode_with_native_decoder() {
    let image = PrecomputedHtj2k53Image {
        width: 8,
        height: 8,
        bit_depth: 8,
        signed: false,
        components: vec![PrecomputedHtj2k53Component {
            x_rsiz: 1,
            y_rsiz: 1,
            dwt: zero_dwt53(8, 8),
        }],
    };
    let bytes = encode_precomputed_htj2k_53(&image, &precomputed_options())
        .expect("encode precomputed HTJ2K coefficients");
    let decoded = Image::new(&bytes, &DecodeSettings::default())
        .expect("native parser accepts codestream")
        .decode_native()
        .expect("native decoder accepts codestream");

    assert_eq!((decoded.width, decoded.height), (8, 8));
    assert_eq!(decoded.num_components, 1);
    assert!(decoded.data.iter().all(|&sample| sample == 128));
}

#[test]
fn precomputed_encode_writes_component_sampling_in_siz() {
    let image = PrecomputedHtj2k53Image {
        width: 16,
        height: 16,
        bit_depth: 8,
        signed: false,
        components: vec![
            PrecomputedHtj2k53Component {
                x_rsiz: 1,
                y_rsiz: 1,
                dwt: zero_dwt53(16, 16),
            },
            PrecomputedHtj2k53Component {
                x_rsiz: 2,
                y_rsiz: 2,
                dwt: zero_dwt53(8, 8),
            },
            PrecomputedHtj2k53Component {
                x_rsiz: 2,
                y_rsiz: 2,
                dwt: zero_dwt53(8, 8),
            },
        ],
    };
    let bytes = encode_precomputed_htj2k_53(&image, &precomputed_options())
        .expect("encode precomputed subsampled HTJ2K coefficients");
    let siz = find_marker(&bytes, 0x51).expect("SIZ marker");
    let component_info = siz + 40;

    assert_eq!(bytes[component_info + 1], 1);
    assert_eq!(bytes[component_info + 2], 1);
    assert_eq!(bytes[component_info + 4], 2);
    assert_eq!(bytes[component_info + 5], 2);
    assert_eq!(bytes[component_info + 7], 2);
    assert_eq!(bytes[component_info + 8], 2);
}

#[test]
fn precomputed_encode_rejects_component_sampling_geometry_mismatch() {
    let image = PrecomputedHtj2k53Image {
        width: 16,
        height: 16,
        bit_depth: 8,
        signed: false,
        components: vec![PrecomputedHtj2k53Component {
            x_rsiz: 2,
            y_rsiz: 2,
            dwt: zero_dwt53(16, 16),
        }],
    };

    let err = encode_precomputed_htj2k_53(&image, &precomputed_options())
        .expect_err("component DWT geometry must match SIZ sampling");

    assert_eq!(err, "precomputed DWT component dimensions mismatch");
}

#[test]
fn precomputed_encode_rejects_recursive_level_geometry_mismatch() {
    let mut dwt = zero_dwt53(8, 8);
    dwt.levels[0].low_width = 3;
    let image = PrecomputedHtj2k53Image {
        width: 8,
        height: 8,
        bit_depth: 8,
        signed: false,
        components: vec![PrecomputedHtj2k53Component {
            x_rsiz: 1,
            y_rsiz: 1,
            dwt,
        }],
    };

    let err = encode_precomputed_htj2k_53(&image, &precomputed_options())
        .expect_err("level geometry must match recursive 5/3 expectations");

    assert_eq!(err, "precomputed DWT recursive geometry mismatch");
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

fn zero_dwt53(width: u32, height: u32) -> J2kForwardDwt53Output {
    let low_width = width.div_ceil(2);
    let low_height = height.div_ceil(2);
    let high_width = width / 2;
    let high_height = height / 2;

    J2kForwardDwt53Output {
        ll: vec![0.0; (low_width * low_height) as usize],
        ll_width: low_width,
        ll_height: low_height,
        levels: vec![J2kForwardDwt53Level {
            hl: vec![0.0; (high_width * low_height) as usize],
            lh: vec![0.0; (low_width * high_height) as usize],
            hh: vec![0.0; (high_width * high_height) as usize],
            width,
            height,
            low_width,
            low_height,
            high_width,
            high_height,
        }],
    }
}

fn find_marker(codestream: &[u8], marker: u8) -> Option<usize> {
    codestream
        .windows(2)
        .position(|window| window == [0xff, marker])
}
