use j2k_codec_math::dwt;

#[test]
fn generated_source_comparison_ignores_only_line_endings() {
    assert!(generated_source_matches(
        "alpha\r\nbeta\r\n",
        "alpha\nbeta\n"
    ));
    assert!(!generated_source_matches(
        "alpha\r\ngamma\r\n",
        "alpha\nbeta\n"
    ));
}

#[test]
fn metal_dwt97_fragment_matches_rust_constants() {
    let expected = [
        "// Generated from crates/j2k-codec-math/src/dwt.rs.".to_string(),
        format!(
            "constant float CODEC_MATH_DWT97_ALPHA = {}f;",
            compact_f32(dwt::DWT97_ALPHA_F32)
        ),
        format!(
            "constant float CODEC_MATH_DWT97_BETA = {}f;",
            compact_f32(dwt::DWT97_BETA_F32)
        ),
        format!(
            "constant float CODEC_MATH_DWT97_GAMMA = {}f;",
            compact_f32(dwt::DWT97_GAMMA_F32)
        ),
        format!(
            "constant float CODEC_MATH_DWT97_DELTA = {}f;",
            compact_f32(dwt::DWT97_DELTA_F32)
        ),
        format!(
            "constant float CODEC_MATH_DWT97_KAPPA = {}f;",
            compact_f32(dwt::DWT97_KAPPA_F32)
        ),
        "constant float CODEC_MATH_DWT97_INV_KAPPA = 1.0f / CODEC_MATH_DWT97_KAPPA;".to_string(),
        format!(
            "constant float CODEC_MATH_IDWT97_NEG_ALPHA = {}f;",
            compact_f32(dwt::IDWT97_NEG_ALPHA_F32)
        ),
        format!(
            "constant float CODEC_MATH_IDWT97_NEG_BETA = {}f;",
            compact_f32(dwt::IDWT97_NEG_BETA_F32)
        ),
        format!(
            "constant float CODEC_MATH_IDWT97_NEG_GAMMA = {}f;",
            compact_f32(dwt::IDWT97_NEG_GAMMA_F32)
        ),
        format!(
            "constant float CODEC_MATH_IDWT97_NEG_DELTA = {}f;",
            compact_f32(dwt::IDWT97_NEG_DELTA_F32)
        ),
    ]
    .join("\n")
        + "\n";

    assert!(
        generated_source_matches(
            include_str!("../generated/dwt97_constants.metal"),
            &expected
        ),
        "generated Metal constants differ from the Rust source constants"
    );
}

#[test]
fn rust_dwt97_fragment_matches_rust_constants() {
    let expected = [
        "// Generated from crates/j2k-codec-math/src/dwt.rs.".to_string(),
        format!(
            "pub const CODEC_MATH_DWT97_ALPHA: f32 = {};",
            compact_f32(dwt::DWT97_ALPHA_F32)
        ),
        format!(
            "pub const CODEC_MATH_DWT97_BETA: f32 = {};",
            compact_f32(dwt::DWT97_BETA_F32)
        ),
        format!(
            "pub const CODEC_MATH_DWT97_GAMMA: f32 = {};",
            compact_f32(dwt::DWT97_GAMMA_F32)
        ),
        format!(
            "pub const CODEC_MATH_DWT97_DELTA: f32 = {};",
            compact_f32(dwt::DWT97_DELTA_F32)
        ),
        format!(
            "pub const CODEC_MATH_DWT97_KAPPA: f32 = {};",
            compact_f32(dwt::DWT97_KAPPA_F32)
        ),
        "pub const CODEC_MATH_DWT97_INV_KAPPA: f32 = 1.0 / CODEC_MATH_DWT97_KAPPA;".to_string(),
        format!(
            "pub const CODEC_MATH_IDWT97_NEG_ALPHA: f32 = {};",
            compact_f32(dwt::IDWT97_NEG_ALPHA_F32)
        ),
        format!(
            "pub const CODEC_MATH_IDWT97_NEG_BETA: f32 = {};",
            compact_f32(dwt::IDWT97_NEG_BETA_F32)
        ),
        format!(
            "pub const CODEC_MATH_IDWT97_NEG_GAMMA: f32 = {};",
            compact_f32(dwt::IDWT97_NEG_GAMMA_F32)
        ),
        format!(
            "pub const CODEC_MATH_IDWT97_NEG_DELTA: f32 = {};",
            compact_f32(dwt::IDWT97_NEG_DELTA_F32)
        ),
    ]
    .join("\n")
        + "\n";

    assert!(
        generated_source_matches(include_str!("../generated/dwt97_constants.rs"), &expected),
        "generated Rust constants differ from the source constants"
    );
}

fn compact_f32(value: f32) -> String {
    format!("{value:?}")
}

fn generated_source_matches(actual: &str, expected: &str) -> bool {
    actual.replace("\r\n", "\n") == expected.replace("\r\n", "\n")
}

/// The integers of the Metal initializer `name[...] = { ... };`, without
/// vector constructors, casts, or `u` suffixes.
fn metal_array(source: &str, name: &str) -> Vec<u32> {
    let start = source
        .find(&format!("{name}["))
        .unwrap_or_else(|| panic!("generated Metal omits {name}"));
    let body = &source[start..];
    let body = &body[body.find('{').expect("initializer opens") + 1..];
    let body = &body[..body.find("};").expect("initializer closes")];
    body.split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|token| token.starts_with(|first: char| first.is_ascii_digit()))
        .map(|token| {
            let token = token.trim_end_matches('u');
            token.strip_prefix("0x").map_or_else(
                || token.parse().expect("decimal literal"),
                |hex| u32::from_str_radix(hex, 16).expect("hex literal"),
            )
        })
        .collect()
}

#[test]
fn metal_classic_fragment_matches_rust_tables() {
    use j2k_codec_math::classic;

    let source = include_str!("../generated/classic_tables.metal");
    let states = classic::MQ_STATES
        .iter()
        .flat_map(|state| {
            [
                state.qe,
                u32::from(state.nmps),
                u32::from(state.nlps),
                u32::from(state.switch),
            ]
        })
        .collect::<Vec<_>>();
    assert_eq!(metal_array(source, "J2K_QE_TABLE"), states);
    let packed = classic::MQ_STATES
        .iter()
        .map(|state| {
            state.qe
                | u32::from(state.nmps) << 16
                | u32::from(state.nlps) << 22
                | u32::from(state.switch) << 28
        })
        .collect::<Vec<_>>();
    assert_eq!(metal_array(source, "J2K_PACKED_QE_TABLE"), packed);
    let sign = classic::SIGN_CONTEXT_LOOKUP
        .iter()
        .flat_map(|&(context, xor)| [u32::from(context), u32::from(xor)])
        .collect::<Vec<_>>();
    assert_eq!(metal_array(source, "SIGN_CONTEXT_LOOKUP"), sign);
    let tables = [
        ("ZERO_CTX_LL_LH_LOOKUP", &classic::ZERO_CTX_LL_LH_LOOKUP),
        ("ZERO_CTX_HL_LOOKUP", &classic::ZERO_CTX_HL_LOOKUP),
        ("ZERO_CTX_HH_LOOKUP", &classic::ZERO_CTX_HH_LOOKUP),
    ];
    for (name, table) in tables {
        let expected = table.iter().copied().map(u32::from).collect::<Vec<_>>();
        assert_eq!(metal_array(source, name), expected, "{name}");
    }
    // Nine-bit stripe neighborhoods: bit 4 is the sample itself; the other
    // bits are the eight-bit neighbor index with horizontal pairs swapped.
    let plain = metal_array(source, "PLAIN_ZERO_CONTEXTS");
    assert_eq!(plain.len(), 512);
    for (neighborhood, &value) in plain.iter().enumerate() {
        let index = [
            (0, 1),
            (1, 0),
            (2, 3),
            (3, 2),
            (5, 4),
            (6, 5),
            (7, 6),
            (8, 7),
        ]
        .into_iter()
        .filter(|&(from, _)| neighborhood >> from & 1 != 0)
        .fold(0_usize, |index, (_, to)| index | 1 << to);
        let expected = u32::from(classic::ZERO_CTX_LL_LH_LOOKUP[index])
            | u32::from(classic::ZERO_CTX_HL_LOOKUP[index]) << 4
            | u32::from(classic::ZERO_CTX_HH_LOOKUP[index]) << 8;
        assert_eq!(value, expected, "PLAIN_ZERO_CONTEXTS[{neighborhood}]");
    }
}

#[test]
fn metal_htj2k_fragment_matches_rust_constants() {
    use j2k_codec_math::htj2k;

    let source = include_str!("../generated/htj2k_constants.metal");
    assert_eq!(
        metal_array(source, "J2K_HT_MEL_EXP"),
        htj2k::MEL_EXPONENTS.map(u32::from)
    );
    assert!(source.contains(&format!(
        "constant uint J2K_HT_MAX_SCRATCH = {}u;",
        htj2k::HT_MAX_SCRATCH
    )));
}

#[test]
fn metal_mct_fragment_matches_rust_constants() {
    use j2k_codec_math::mct;

    let source = include_str!("../generated/mct_constants.metal");
    for (name, value) in [
        ("ICT_FWD_Y_R", mct::ICT_FWD_Y_R),
        ("ICT_FWD_Y_G", mct::ICT_FWD_Y_G),
        ("ICT_FWD_Y_B", mct::ICT_FWD_Y_B),
        ("ICT_FWD_CB_R", mct::ICT_FWD_CB_R),
        ("ICT_FWD_CB_G", mct::ICT_FWD_CB_G),
        ("ICT_FWD_CB_B", mct::ICT_FWD_CB_B),
        ("ICT_FWD_CR_R", mct::ICT_FWD_CR_R),
        ("ICT_FWD_CR_G", mct::ICT_FWD_CR_G),
        ("ICT_FWD_CR_B", mct::ICT_FWD_CR_B),
        ("ICT_INV_R_CR", mct::ICT_INV_R_CR),
        ("ICT_INV_G_CB", mct::ICT_INV_G_CB),
        ("ICT_INV_G_CR", mct::ICT_INV_G_CR),
        ("ICT_INV_B_CB", mct::ICT_INV_B_CB),
    ] {
        let line = format!(
            "constant float CODEC_MATH_{name} = {}f;",
            compact_f32(value)
        );
        assert!(source.contains(&line), "generated Metal omits `{line}`");
    }
}
