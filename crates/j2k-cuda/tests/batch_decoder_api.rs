// SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "batch_decoder_api/basic_contracts.rs"]
mod basic_contracts;

#[cfg(feature = "cuda-runtime")]
#[path = "batch_decoder_api/async_resident.rs"]
mod async_resident;
#[cfg(feature = "cuda-runtime")]
#[path = "batch_decoder_api/classic_native.rs"]
mod classic_native;
#[cfg(feature = "cuda-runtime")]
#[path = "batch_decoder_api/exact_color.rs"]
mod exact_color;
#[cfg(feature = "cuda-runtime")]
#[path = "batch_decoder_api/external_lifecycle.rs"]
mod external_lifecycle;
#[cfg(feature = "cuda-runtime")]
#[path = "batch_decoder_api/multitile.rs"]
mod multitile;
#[cfg(feature = "cuda-runtime")]
#[path = "batch_decoder_api/refinement_overlap.rs"]
mod refinement_overlap;
#[cfg(feature = "cuda-runtime")]
#[path = "batch_decoder_api/rgba.rs"]
mod rgba;
#[cfg(feature = "cuda-runtime")]
#[path = "batch_decoder_api/session_soak.rs"]
mod session_soak;
#[cfg(feature = "cuda-runtime")]
#[path = "batch_decoder_api/signed_rgb.rs"]
mod signed_rgb;
#[cfg(feature = "cuda-runtime")]
#[path = "batch_decoder_api/support.rs"]
mod support;
