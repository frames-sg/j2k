// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::JpegError;
use crate::info::{ColorSpace, DecodeOptions, Info, RestartIndex, SofKind};
use crate::parse::header::{parse_header, ParsedHeader};
use j2k_core::{CompressedPayloadKind, PassthroughCandidate};

use super::core_traits::jpeg_passthrough_syntax;
use super::{find_component_index, restart_index_for_stream};

/// A parsed borrowed view of a JPEG stream.
#[derive(Debug)]
pub struct JpegView<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) header: ParsedHeader,
    pub(super) info: Info,
    pub(super) options: DecodeOptions,
}

impl<'a> JpegView<'a> {
    /// Parse the stream into a borrowed view that can later build a decoder.
    ///
    /// # Errors
    ///
    /// Returns an error when the JPEG header is malformed or unsupported.
    pub fn parse(input: &'a [u8]) -> Result<Self, JpegError> {
        Self::parse_with_options(input, DecodeOptions::default())
    }

    /// Parse the stream with explicit decode options.
    ///
    /// # Errors
    ///
    /// Returns an error when the JPEG header is malformed or unsupported.
    pub fn parse_with_options(input: &'a [u8], options: DecodeOptions) -> Result<Self, JpegError> {
        let header = parse_header(input)?;
        let mut info = header.info();
        options.apply_to_info(&mut info);
        Ok(Self {
            bytes: input,
            header,
            info,
            options,
        })
    }

    /// Header-derived metadata for the parsed stream.
    #[must_use]
    pub fn info(&self) -> &Info {
        &self.info
    }

    /// Original compressed bytes backing this view.
    #[must_use]
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Return a byte-preserving passthrough candidate for active DICOM/WSI
    /// transfer syntaxes.
    ///
    /// Progressive JPEG is intentionally not exposed here because the active
    /// conversion path should transcode it rather than introduce a retired or
    /// unsupported destination syntax.
    #[must_use]
    pub fn passthrough_candidate(&self) -> Option<PassthroughCandidate<'a>> {
        jpeg_passthrough_syntax(&self.info).map(|transfer_syntax| {
            PassthroughCandidate::new(
                self.bytes,
                transfer_syntax,
                CompressedPayloadKind::JpegInterchange,
                self.info.to_core_info(),
            )
        })
    }

    /// Build a restart-marker byte-offset index for the first scan.
    ///
    /// Offsets are absolute byte positions in the original JPEG byte slice.
    /// Returns `Ok(None)` when the stream has no non-zero DRI marker.
    ///
    /// # Errors
    ///
    /// Returns an error when restart-marker syntax is malformed.
    pub fn restart_index(&self) -> Result<Option<RestartIndex>, JpegError> {
        restart_index_for_stream(
            self.bytes,
            self.header.sos_offset,
            &self.info,
            self.info.restart_interval,
        )
    }

    pub(crate) fn has_lossless_subsampled_color_capability_shape(&self) -> bool {
        if self.info.sof_kind != SofKind::Lossless
            || !matches!(self.info.color_space, ColorSpace::Rgb | ColorSpace::YCbCr)
            || !matches!(self.info.bit_depth, 8 | 16)
            || self.info.sampling.len() != 3
            || !self
                .info
                .sampling
                .components()
                .iter()
                .any(|&(h, v)| h != 1 || v != 1)
            || self.header.scan_count != 1
        {
            return false;
        }

        let Some(scan) = self.header.scan.as_ref() else {
            return false;
        };
        if !(1..=7).contains(&scan.ss)
            || scan.se != 0
            || scan.ah != 0
            || scan.al != 0
            || scan.components.len() != 3
        {
            return false;
        }

        scan.components.iter().all(|scan_component| {
            find_component_index(&self.header.component_ids, scan_component.id).is_some()
                && self.header.huffman_tables.dc[scan_component.dc_table as usize].is_some()
        })
    }
}
