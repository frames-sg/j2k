// SPDX-License-Identifier: MIT OR Apache-2.0

pub(super) mod decode_case;
mod fixture;
pub(super) mod input_selection;
pub(super) mod workload;

pub(super) const LOW_BATCH_SIZES: &[usize] = &[1, 8];
pub(super) const BATCH_SIZES: &[usize] = &[LOW_BATCH_SIZES[0], LOW_BATCH_SIZES[1], 32, 64];
pub(super) const GENERATED_BATCH_SIZE: usize = 64;
