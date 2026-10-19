use j2k_native::{
    encode, irreversible_quantization_profile_scale_ceiling, DecodeSettings, EncodeError,
    EncodeOptions, Image,
};

fn pixels() -> Vec<u8> {
    (0..64 * 64)
        .flat_map(|i| {
            let x = i % 64;
            let y = i / 64;
            let value = u8::try_from((x * 17 + y * 29) % 180 + 30).unwrap();
            [value, value, value]
        })
        .collect()
}

fn options(ht: bool) -> EncodeOptions {
    EncodeOptions {
        reversible: false,
        use_mct: false,
        use_ht_block_coding: ht,
        num_decomposition_levels: 2,
        irreversible_quantization_scale: 32.0,
        ..EncodeOptions::default()
    }
}

#[test]
fn all_ones_profile_preserves_default_codestream_and_component_profile_changes_decode() {
    let pixels = pixels();
    for ht in [false, true] {
        let mut options = options(ht);
        let default = encode(&pixels, 64, 64, 3, 8, false, &options).unwrap();
        options.irreversible_quantization_profile = Some(vec![vec![1.0; 7]; 3]);
        assert_eq!(
            encode(&pixels, 64, 64, 3, 8, false, &options).unwrap(),
            default
        );
        options.irreversible_quantization_profile.as_mut().unwrap()[1][4] = 8.0;
        let weighted = encode(&pixels, 64, 64, 3, 8, false, &options).unwrap();
        assert!(weighted.windows(2).any(|marker| marker == [0xff, 0x5d]));
        let baseline = Image::new(&default, &DecodeSettings::default())
            .unwrap()
            .decode()
            .unwrap();
        let decoded = Image::new(&weighted, &DecodeSettings::default())
            .unwrap()
            .decode()
            .unwrap();
        for component in [0, 2] {
            assert!(baseline
                .iter()
                .skip(component)
                .step_by(3)
                .eq(decoded.iter().skip(component).step_by(3)));
        }
        assert!(baseline
            .iter()
            .skip(1)
            .step_by(3)
            .zip(decoded.iter().skip(1).step_by(3))
            .any(|(a, b)| a != b));
    }
}

#[test]
fn invalid_profile_shape_values_and_reversible_use_are_rejected() {
    let pixels = pixels();
    for profile in [
        vec![vec![1.0; 7]; 2],
        vec![vec![1.0; 6]; 3],
        vec![vec![0.0; 7]; 3],
        vec![vec![f32::NAN; 7]; 3],
    ] {
        let mut options = options(false);
        options.irreversible_quantization_profile = Some(profile);
        assert!(encode(&pixels, 64, 64, 3, 8, false, &options).is_err());
    }
    let mut options = options(false);
    options.reversible = true;
    options.irreversible_quantization_profile = Some(vec![vec![1.0; 7]; 3]);
    assert!(encode(&pixels, 64, 64, 3, 8, false, &options).is_err());
}

#[test]
fn component_profile_survives_multitile_options_copy() {
    let pixels = pixels();
    let mut options = options(true);
    options.tile_size = Some((32, 32));
    options.code_block_width_exp = 3;
    options.code_block_height_exp = 3;
    let default = encode(&pixels, 64, 64, 3, 8, false, &options).unwrap();
    options.irreversible_quantization_profile = Some(vec![vec![1.0; 7]; 3]);
    options.irreversible_quantization_profile.as_mut().unwrap()[2][0] = 4.0;
    let weighted = encode(&pixels, 64, 64, 3, 8, false, &options).unwrap();
    assert_ne!(weighted, default);
    Image::new(&weighted, &DecodeSettings::default())
        .unwrap()
        .decode()
        .unwrap();
}

#[test]
fn unrepresentable_profile_weights_are_input_errors() {
    let pixels = vec![128_u8; 64 * 64 * 3];
    for ht in [false, true] {
        for weight in [1e-20, f32::MIN_POSITIVE, 1e9, f32::MAX] {
            for profile in [vec![vec![weight; 7]; 3], {
                let mut profile = vec![vec![1.0; 7]; 3];
                profile[0][0] = weight;
                profile
            }] {
                let mut options = options(ht);
                options.irreversible_quantization_profile = Some(profile);
                assert!(
                    matches!(
                        encode(&pixels, 64, 64, 3, 8, false, &options),
                        Err(EncodeError::InvalidInput { .. })
                    ),
                    "ht={ht} weight={weight}"
                );
            }
        }
    }
}

#[test]
fn profile_scale_ceiling_is_the_last_scale_encode_accepts() {
    let pixels = pixels();
    let mut options = options(true);
    assert_eq!(
        irreversible_quantization_profile_scale_ceiling(8, &options),
        None
    );
    options.irreversible_quantization_profile = Some(vec![vec![1.0; 7]; 3]);
    assert_eq!(
        irreversible_quantization_profile_scale_ceiling(8, &options),
        None
    );
    options.irreversible_quantization_profile =
        Some(vec![vec![1.0, 3.0, 3.0, 3.0, 5.0, 5.0, 5.0]; 3]);
    let ceiling = irreversible_quantization_profile_scale_ceiling(8, &options).unwrap();
    assert!(ceiling > options.irreversible_quantization_scale);
    options.irreversible_quantization_scale = ceiling;
    let encoded = encode(&pixels, 64, 64, 3, 8, false, &options).unwrap();
    Image::new(&encoded, &DecodeSettings::default())
        .unwrap()
        .decode()
        .unwrap();
    options.irreversible_quantization_scale = ceiling.next_up();
    assert!(matches!(
        encode(&pixels, 64, 64, 3, 8, false, &options),
        Err(EncodeError::InvalidInput { .. })
    ));
}

#[test]
fn native_multitile_profile_rows_follow_the_edge_tile_depth() {
    // The 1-px and 16-px last tile columns cap the depth for every tile.
    for (width, levels) in [(65_u32, 0_usize), (80, 4)] {
        let pixels: Vec<u8> = (0..width * 64 * 3)
            .map(|i| u8::try_from(i % 251).unwrap())
            .collect();
        let mut options = options(false);
        options.num_decomposition_levels = 5;
        options.tile_size = Some((64, 64));
        let default = encode(&pixels, width, 64, 3, 8, false, &options).unwrap();
        options.irreversible_quantization_profile = Some(vec![vec![1.0; 1 + 3 * levels]; 3]);
        assert_eq!(
            encode(&pixels, width, 64, 3, 8, false, &options).unwrap(),
            default,
            "{width} px all-ones profile"
        );
        options.irreversible_quantization_profile.as_mut().unwrap()[1][0] = 4.0;
        let weighted = encode(&pixels, width, 64, 3, 8, false, &options).unwrap();
        assert_ne!(weighted, default, "{width} px weighted profile");
        Image::new(&weighted, &DecodeSettings::default())
            .unwrap()
            .decode()
            .unwrap();
        options.irreversible_quantization_profile = Some(vec![vec![2.0; 16]; 3]);
        assert!(
            matches!(
                encode(&pixels, width, 64, 3, 8, false, &options),
                Err(EncodeError::InvalidInput { .. })
            ),
            "{width} px rows sized for the requested depth"
        );
    }
}
