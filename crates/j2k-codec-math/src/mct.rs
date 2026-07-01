//! JPEG 2000 multi-component transform constants.

/// Forward ICT coefficient for R into Y.
pub const ICT_FWD_Y_R: f32 = 0.299;
/// Forward ICT coefficient for G into Y.
pub const ICT_FWD_Y_G: f32 = 0.587;
/// Forward ICT coefficient for B into Y.
pub const ICT_FWD_Y_B: f32 = 0.114;
/// Forward ICT coefficient for R into Cb.
pub const ICT_FWD_CB_R: f32 = -0.16875;
/// Forward ICT coefficient for G into Cb.
pub const ICT_FWD_CB_G: f32 = -0.33126;
/// Forward ICT coefficient for B into Cb.
pub const ICT_FWD_CB_B: f32 = 0.5;
/// Forward ICT coefficient for R into Cr.
pub const ICT_FWD_CR_R: f32 = 0.5;
/// Forward ICT coefficient for G into Cr.
pub const ICT_FWD_CR_G: f32 = -0.41869;
/// Forward ICT coefficient for B into Cr.
pub const ICT_FWD_CR_B: f32 = -0.08131;

/// Inverse ICT coefficient for Cr into R.
pub const ICT_INV_R_CR: f32 = 1.402;
/// Inverse ICT coefficient for Cb into G.
pub const ICT_INV_G_CB: f32 = -0.34413;
/// Inverse ICT coefficient for Cr into G.
pub const ICT_INV_G_CR: f32 = -0.71414;
/// Inverse ICT coefficient for Cb into B.
pub const ICT_INV_B_CB: f32 = 1.772;

/// Reversible color transform quarter scale.
pub const RCT_QUARTER: f32 = 0.25;
