// SPDX-License-Identifier: MIT OR Apache-2.0

mod four_component;
mod output;
mod rgb;
mod rgb444;
mod types;
mod upsample;

pub(super) use self::{
    output::emit_stripe,
    rgb::emit_stripe_rgb,
    rgb444::emit_stripe_rgb_444,
    types::StripeEmit,
    upsample::{component_row_triplet, valid_component_rows},
};
