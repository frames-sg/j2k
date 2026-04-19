// SPDX-License-Identifier: Apache-2.0

//! JPEG 2000 inspect support for slidecodec.

pub mod error;
pub use error::J2kError;

pub mod view;
pub use view::{J2kDecoder, J2kView};

pub(crate) mod parse;
