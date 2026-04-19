// SPDX-License-Identifier: Apache-2.0

use crate::{parse::parse_info, J2kError};
use slidecodec_core::Info;

#[derive(Debug)]
pub struct J2kView<'a> {
    bytes: &'a [u8],
    info: Info,
}

impl<'a> J2kView<'a> {
    pub fn parse(input: &'a [u8]) -> Result<Self, J2kError> {
        let info = parse_info(input)?;
        Ok(Self { bytes: input, info })
    }

    pub fn info(&self) -> &Info {
        &self.info
    }

    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }
}

#[derive(Debug)]
pub struct J2kDecoder<'a> {
    bytes: &'a [u8],
    info: Info,
}

impl<'a> J2kDecoder<'a> {
    pub fn inspect(input: &'a [u8]) -> Result<Info, J2kError> {
        parse_info(input)
    }

    pub fn new(input: &'a [u8]) -> Result<Self, J2kError> {
        Self::from_view(J2kView::parse(input)?)
    }

    pub fn from_view(view: J2kView<'a>) -> Result<Self, J2kError> {
        Ok(Self {
            bytes: view.bytes,
            info: view.info,
        })
    }

    pub fn info(&self) -> &Info {
        &self.info
    }

    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }
}
