// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, fmt::Write as _};

use serde::{Deserialize, Serialize};

use crate::{manifest::validate_sha256, EncoderReferenceDecoder};

use super::EncoderCaseReport;
use crate::report::{markdown_cell, report_error, ReportError};

/// Exact T.804 reference implementation used for supported Annex D cases.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderReferenceIdentity {
    /// Reference-software standard.
    pub standard: String,
    /// Reference decoder implementation.
    pub implementation: String,
    /// Exact implementation version.
    pub version: String,
}

/// Executable provenance for a supplemental encoder interoperability decoder.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderSupplementalReferenceIdentity {
    /// Matrix decoder selection represented by this identity.
    pub decoder: EncoderReferenceDecoder,
    /// Exact evidence scope, including exclusions from formal reference-software evidence.
    pub scope: String,
    /// Decoder implementation.
    pub implementation: String,
    /// Exact implementation version.
    pub version: String,
    /// Upstream source repository.
    pub source_url: String,
    /// Exact source revision used to build the executable.
    pub source_commit: String,
    /// SHA-256 of the executable used by this run.
    pub executable_sha256: String,
}

pub(super) fn validate_reference_decoders(
    primary: &EncoderReferenceIdentity,
    supplemental: &[EncoderSupplementalReferenceIdentity],
    cases: &[EncoderCaseReport],
) -> Result<(), ReportError> {
    if [&primary.standard, &primary.implementation, &primary.version]
        .into_iter()
        .any(String::is_empty)
    {
        return report_error("encoder reference decoder identity must not be empty");
    }

    let selected = cases
        .iter()
        .map(|case| case.reference_decoder)
        .filter(|decoder| *decoder != EncoderReferenceDecoder::OpenJpeg)
        .collect::<BTreeSet<_>>();
    let identities = supplemental
        .iter()
        .map(|identity| identity.decoder)
        .collect::<BTreeSet<_>>();
    if supplemental.len() != identities.len()
        || !supplemental
            .windows(2)
            .all(|pair| pair[0].decoder < pair[1].decoder)
    {
        return report_error("supplemental reference decoder identities must be sorted and unique");
    }
    for identity in supplemental {
        validate_supplemental_identity(identity)?;
    }
    if selected != identities {
        return report_error("selected OpenHTJ2K identity is missing or unused");
    }
    Ok(())
}

pub(super) fn push_supplemental_reference_markdown(
    supplemental: &[EncoderSupplementalReferenceIdentity],
    markdown: &mut String,
) {
    if supplemental.is_empty() {
        return;
    }
    markdown.push_str("\nSupplemental reference decoders:\n");
    for identity in supplemental {
        let _ = write!(
            markdown,
            "\n- {} {}: {}; source {} at `{}`; executable SHA-256 `{}`.\n",
            markdown_cell(&identity.implementation),
            markdown_cell(&identity.version),
            markdown_cell(&identity.scope),
            markdown_cell(&identity.source_url),
            identity.source_commit,
            identity.executable_sha256,
        );
    }
}

pub(super) const fn reference_decoder_name(decoder: EncoderReferenceDecoder) -> &'static str {
    match decoder {
        EncoderReferenceDecoder::OpenJpeg => "OpenJPEG",
        EncoderReferenceDecoder::OpenHtj2k => "OpenHTJ2K",
    }
}

fn validate_supplemental_identity(
    identity: &EncoderSupplementalReferenceIdentity,
) -> Result<(), ReportError> {
    if identity.decoder == EncoderReferenceDecoder::OpenJpeg
        || [
            &identity.scope,
            &identity.implementation,
            &identity.version,
            &identity.source_url,
        ]
        .into_iter()
        .any(String::is_empty)
        || !identity.scope.contains("not T.804")
        || !identity.source_url.starts_with("https://")
        || identity.source_commit.len() != 40
        || !identity
            .source_commit
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return report_error("supplemental reference decoder identity is incomplete");
    }
    validate_sha256(
        &identity.executable_sha256,
        "supplemental reference decoder executable",
    )
    .map_err(|error| ReportError::Validation(error.to_string()))
}
