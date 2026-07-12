//! Scalar HTJ2K block encoding.

mod allocation;
mod cleanup;
mod distribution;
mod emit;
mod facade;
mod quad;
mod refinement;
mod writers;

pub(crate) use allocation::ht_worker_allocation;
pub(crate) use distribution::collect_encode_distribution;
#[cfg(test)]
pub(crate) use facade::{encode_code_block, encode_code_block_with_passes};
pub(crate) use facade::{try_encode_code_block, try_encode_code_block_with_passes};

#[cfg(test)]
mod golden_tests;
#[cfg(test)]
mod tests;
