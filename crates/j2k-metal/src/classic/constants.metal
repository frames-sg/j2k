constant uint J2K_CLASSIC_STATUS_OK = 0u;
constant uint J2K_CLASSIC_STATUS_FAIL = 1u;
constant uint J2K_CLASSIC_STATUS_UNSUPPORTED = 2u;
constant uint J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES = 1u << 0;
constant uint J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS = 1u << 1;
constant uint J2K_CLASSIC_STYLE_VERTICALLY_CAUSAL_CONTEXT = 1u << 2;
constant uint J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS = 1u << 3;
constant uint J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS = 1u << 4;

constant uint J2K_CLASSIC_MAX_WIDTH = 64u;
constant uint J2K_CLASSIC_MAX_HEIGHT = 64u;
constant uint J2K_CLASSIC_PADDING = 1u;
constant uint J2K_CLASSIC_MAX_PADDED_WIDTH = J2K_CLASSIC_MAX_WIDTH + J2K_CLASSIC_PADDING * 2u;
constant uint J2K_CLASSIC_MAX_PADDED_HEIGHT = J2K_CLASSIC_MAX_HEIGHT + J2K_CLASSIC_PADDING * 2u;
constant uint J2K_CLASSIC_MAX_COEFF_COUNT = J2K_CLASSIC_MAX_PADDED_WIDTH * J2K_CLASSIC_MAX_PADDED_HEIGHT;
// One word covers four coefficients and their 3-column by 6-row context window.
// Sign, magnitude-refinement, and significance-propagation bits occupy the
// remaining positions so the plain decoder can retain one word while scanning.
constant uint J2K_CLASSIC_MAX_FLAG_WORDS = J2K_CLASSIC_MAX_PADDED_WIDTH * ((J2K_CLASSIC_MAX_HEIGHT + 3u) / 4u);
constant uint J2K_CLASSIC_SIGMA_THIS = 1u << 4u;
constant uint J2K_CLASSIC_SIGMA_NEIGHBORS = 0x1EFu;
constant uint J2K_CLASSIC_SIGMA_ROWS =
    J2K_CLASSIC_SIGMA_THIS |
    (J2K_CLASSIC_SIGMA_THIS << 3u) |
    (J2K_CLASSIC_SIGMA_THIS << 6u) |
    (J2K_CLASSIC_SIGMA_THIS << 9u);
constant uint J2K_CLASSIC_MU_THIS = 1u << 20u;
constant uint J2K_CLASSIC_PI_THIS = 1u << 21u;
constant uint J2K_CLASSIC_PI_ALL =
    J2K_CLASSIC_PI_THIS |
    (J2K_CLASSIC_PI_THIS << 3u) |
    (J2K_CLASSIC_PI_THIS << 6u) |
    (J2K_CLASSIC_PI_THIS << 9u);
constant uchar J2K_SIG_SHIFT = 7u;
constant uchar J2K_MAG_REF_SHIFT = 6u;
constant uchar J2K_SIGN_SHIFT = 5u;
constant uchar J2K_STATE_MARKER_MASK = uchar(0x1Fu);
