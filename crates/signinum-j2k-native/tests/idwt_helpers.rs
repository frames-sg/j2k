use signinum_j2k_native::idwt_band_index;

#[test]
fn idwt_band_index_matches_low_and_high_band_coordinate_mapping() {
    assert_eq!(idwt_band_index(0, 0, true), 0);
    assert_eq!(idwt_band_index(0, 1, true), 1);
    assert_eq!(idwt_band_index(0, 1, false), 0);
    assert_eq!(idwt_band_index(0, 2, false), 1);
}

#[test]
fn idwt_band_index_accounts_for_odd_output_origins() {
    assert_eq!(idwt_band_index(3, 0, true), 0);
    assert_eq!(idwt_band_index(3, 1, true), 0);
    assert_eq!(idwt_band_index(3, 2, true), 1);
    assert_eq!(idwt_band_index(3, 0, false), 0);
    assert_eq!(idwt_band_index(3, 1, false), 1);
}
