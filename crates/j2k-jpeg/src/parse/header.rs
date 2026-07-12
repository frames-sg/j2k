// SPDX-License-Identifier: MIT OR Apache-2.0

//! JPEG header parsing and compact progressive-script ownership.

mod inspect;
mod markers;
mod progressive;
mod types;
mod validation;
mod walk;

pub(crate) use inspect::parse_info;
pub(crate) use types::ParsedHeader;
pub(crate) use walk::{parse_header, parse_header_with_external_live};

#[cfg(test)]
pub(crate) use types::ParsedProgressiveScan;

#[cfg(test)]
mod tests;
