// SPDX-License-Identifier: MIT OR Apache-2.0

mod availability;
mod decision;
mod eligibility;
pub(crate) mod promotion;
mod rejection;
mod telemetry;

pub(crate) use availability::auto_cuda_available;
pub(crate) use decision::{auto_decode_uses_cuda, auto_repeated_decode_uses_cuda};
pub(crate) use eligibility::{inputs_repeat_one_slice, AutoDecodeOperation};

#[cfg(test)]
mod tests;
