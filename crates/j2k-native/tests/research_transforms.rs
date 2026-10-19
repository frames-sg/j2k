use j2k_native::{
    forward_dwt97_reference, forward_ict_reference, inverse_dwt97_reference, inverse_ict_reference,
};

#[test]
#[expect(
    clippy::cast_precision_loss,
    reason = "plane coordinates stay below 100 and convert to f32 exactly"
)]
fn edited_float_coefficients_reconstruct_odd_and_even_planes() {
    for (width, height, levels) in [(17, 19, 0), (17, 19, 3), (96, 80, 3)] {
        let pixels: Vec<f32> = (0..width * height)
            .map(|i| {
                0.4 + 0.2 * ((i % width) as f32 * 0.17).sin()
                    + 0.1 * ((i / width) as f32 * 0.23).cos()
            })
            .collect();
        let mut bands = forward_dwt97_reference(&pixels, width, height, levels).unwrap();
        let restored = inverse_dwt97_reference(&bands).unwrap();
        let error = restored
            .iter()
            .zip(&pixels)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(error < 3e-6, "{width}x{height}: {error}");
        if let Some(finest) = bands.levels.last_mut() {
            finest.hl.fill(0.0);
            finest.lh.fill(0.0);
            finest.hh.fill(0.0);
            let altered = inverse_dwt97_reference(&bands).unwrap();
            assert!(altered
                .iter()
                .zip(&pixels)
                .any(|(a, b)| (a - b).abs() > 1e-5));
        }
    }
}

#[test]
fn float_color_roundtrip_keeps_unclipped_values() {
    let rgb = vec![
        vec![-0.1, 0.2, 1.1],
        vec![0.4, 0.1, 0.8],
        vec![0.9, 0.3, 0.2],
    ];
    let actual = inverse_ict_reference(forward_ict_reference(rgb.clone())).unwrap();
    for (a, b) in actual.iter().flatten().zip(rgb.iter().flatten()) {
        // Allow the color error from the codec's rounded ICT constants.
        assert!((a - b).abs() < 3e-5, "{a} versus {b}");
    }
    assert!(actual[0][0] < 0.0 && actual[0][2] > 1.0);
}

#[test]
fn malformed_coefficient_geometry_is_rejected() {
    let mut bands = forward_dwt97_reference(&vec![0.5; 17 * 19], 17, 19, 2).unwrap();
    bands.levels[0].hl.pop();
    assert!(inverse_dwt97_reference(&bands).is_err());
    assert!(inverse_ict_reference(vec![vec![0.0], vec![1.0], vec![]]).is_err());
}
