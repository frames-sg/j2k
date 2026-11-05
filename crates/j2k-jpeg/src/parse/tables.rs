// SPDX-License-Identifier: MIT OR Apache-2.0

//! Versioned JPEG DQT/DHT definitions and active table-slot state.

mod dht;
mod dqt;
mod state;
mod types;

pub(crate) use dht::parse_dht;
pub(crate) use dqt::parse_dqt;
pub(crate) use state::{HuffmanTables, ProgressiveTableState, QuantTables};
#[cfg(test)]
pub(crate) use types::raw_table;
pub(crate) use types::{HuffmanTableRole, HuffmanValues, RawHuffmanTable};

#[cfg(test)]
mod tests;
