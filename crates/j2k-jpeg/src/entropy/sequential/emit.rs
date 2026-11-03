// SPDX-License-Identifier: MIT OR Apache-2.0

mod four_component;
mod output;
mod rgb;
mod rgb444;
mod types;
mod upsample;

#[cfg(test)]
pub(super) use self::upsample::component_row_triplet;
pub(super) use self::{
    output::emit_stripe,
    rgb::emit_stripe_rgb,
    rgb444::FastRgbRowsEmitter,
    types::StripeEmit,
    upsample::{rgb420_chroma_rows, valid_component_rows},
};
