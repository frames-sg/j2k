//! Scalar HTJ2K block encoding.

mod cleanup;
mod distribution;
mod emit;
mod facade;
mod quad;
mod refinement;
mod writers;

pub(crate) use distribution::collect_encode_distribution;
pub(crate) use facade::{encode_code_block, encode_code_block_with_passes};

#[cfg(test)]
mod golden_tests;
#[cfg(test)]
mod tests;
