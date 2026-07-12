// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded typed fields for direct and hybrid Metal profiling rows.

use std::{cell::RefCell, fmt};

use j2k_core::PixelFormat;
use j2k_profile::{same_summary_labels, ProfileField, ProfileResult};

const METAL_PROFILE_DECODE_LABEL_ENV: &str = "J2K_METAL_PROFILE_DECODE_LABEL";

thread_local! {
    static METAL_DIRECT_PROFILE_SUMMARY: RefCell<j2k_profile::ProfileSummary> =
        RefCell::new(new_metal_direct_profile_summary().emit_on_drop());
}

fn new_metal_direct_profile_summary() -> j2k_profile::ProfileSummary {
    match same_summary_labels(&[
        "pipeline",
        "label",
        "stage",
        "processor",
        "metric",
        "metric_kind",
        "aggregation",
        "fmt",
        "batch_count",
    ])
    .and_then(j2k_profile::ProfileSummary::new)
    {
        Ok(summary) => summary,
        Err(error) => {
            j2k_profile::emit_profile_error("metal_direct_summary_init", &error);
            j2k_profile::ProfileSummary::default()
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum MetalProfileFormat<'a> {
    Pixel(PixelFormat),
    Family(&'a str),
}

impl fmt::Display for MetalProfileFormat<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pixel(pixel_format) => write!(formatter, "{pixel_format:?}"),
            Self::Family(family) => formatter.write_str(family),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MetalDirectProfileRow<'a> {
    pub(crate) pipeline: &'a str,
    pub(crate) label: &'a str,
    pub(crate) stage: &'a str,
    pub(crate) processor: &'a str,
    pub(crate) metric: &'a str,
    pub(crate) metric_kind: &'a str,
    pub(crate) aggregation: &'a str,
    pub(crate) fmt: MetalProfileFormat<'a>,
    pub(crate) batch_count: usize,
    pub(crate) elapsed_us: u128,
}

pub(crate) fn emit_metal_profile_row(
    codec: &str,
    op: &str,
    path: &str,
    row: &MetalDirectProfileRow<'_>,
) {
    let fields = match metal_direct_profile_fields(row) {
        Ok(fields) => fields,
        Err(error) => {
            j2k_profile::emit_profile_error("metal_direct_fields", &error);
            return;
        }
    };
    j2k_profile::emit_profile_fields(
        super::metal_profile_stage_mode(),
        &METAL_DIRECT_PROFILE_SUMMARY,
        codec,
        op,
        path,
        &fields,
    );
}

fn metal_direct_profile_fields(
    row: &MetalDirectProfileRow<'_>,
) -> ProfileResult<[ProfileField; 10]> {
    Ok([
        ProfileField::label("pipeline", row.pipeline)?,
        ProfileField::label("label", row.label)?,
        ProfileField::label("stage", row.stage)?,
        ProfileField::label("processor", row.processor)?,
        ProfileField::label("metric", row.metric)?,
        ProfileField::label("metric_kind", row.metric_kind)?,
        ProfileField::label("aggregation", row.aggregation)?,
        ProfileField::label("fmt", row.fmt)?,
        ProfileField::metric("batch_count", row.batch_count)?,
        ProfileField::metric("elapsed_us", row.elapsed_us)?,
    ])
}

pub(crate) fn decode_profile_label() -> ProfileResult<String> {
    match std::env::var(METAL_PROFILE_DECODE_LABEL_ENV) {
        Ok(label) => bounded_profile_label(&label),
        Err(std::env::VarError::NotPresent) => bounded_profile_label("unlabeled"),
        Err(std::env::VarError::NotUnicode(_)) => Err(j2k_profile::ProfileError::InvalidInput {
            what: "Metal profile decode label is not Unicode",
        }),
    }
}

fn bounded_profile_label(label: &str) -> ProfileResult<String> {
    let label = if label.is_empty() { "unlabeled" } else { label };
    ProfileField::label("label", SanitizedProfileLabel(label)).map(ProfileField::into_value)
}

struct SanitizedProfileLabel<'a>(&'a str);

impl fmt::Display for SanitizedProfileLabel<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for character in self.0.chars() {
            let sanitized =
                if character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.') {
                    character
                } else {
                    '_'
                };
            fmt::Write::write_char(formatter, sanitized)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_row(label: &str) -> MetalDirectProfileRow<'_> {
        MetalDirectProfileRow {
            pipeline: "direct",
            label,
            stage: "hybrid",
            processor: "cpu",
            metric: "elapsed",
            metric_kind: "timing",
            aggregation: "sum",
            fmt: MetalProfileFormat::Family("rgb8"),
            batch_count: 1,
            elapsed_us: 7,
        }
    }

    #[test]
    fn direct_profile_emitter_records_summary_rows_in_summary_mode() {
        let _guard = super::super::force_metal_profile_stage_mode_for_test(
            j2k_profile::ProfileStageMode::Summary,
        );
        METAL_DIRECT_PROFILE_SUMMARY.with(|summary| {
            summary
                .borrow_mut()
                .take_formatted_rows()
                .expect("clear Metal direct summary rows");
        });

        emit_metal_profile_row("j2k-metal", "direct", "decode", &test_row("unit"));

        let rows = METAL_DIRECT_PROFILE_SUMMARY.with(|summary| {
            summary
                .borrow_mut()
                .take_formatted_rows()
                .expect("format Metal direct summary rows")
        });
        assert_eq!(
            rows,
            vec![
                "j2k_profile_summary codec=j2k-metal op=direct path=decode pipeline=direct label=unit stage=hybrid processor=cpu metric=elapsed metric_kind=timing aggregation=sum fmt=rgb8 batch_count=1 count=1 elapsed_us_sum=7 elapsed_us_avg=7"
            ]
        );
    }

    #[test]
    fn profile_label_sanitization_is_bounded_and_fallible() {
        assert_eq!(
            bounded_profile_label("release candidate/0.7").expect("bounded label"),
            "release_candidate_0.7"
        );

        let oversized = "x".repeat(j2k_profile::ProfileLimits::default().max_token_bytes() + 1);
        assert!(matches!(
            bounded_profile_label(&oversized),
            Err(j2k_profile::ProfileError::LimitExceeded {
                what: "field value",
                ..
            })
        ));
    }

    #[test]
    fn direct_profile_fields_reject_oversized_values() {
        let oversized = "x".repeat(j2k_profile::ProfileLimits::default().max_token_bytes() + 1);
        assert!(matches!(
            metal_direct_profile_fields(&test_row(&oversized)),
            Err(j2k_profile::ProfileError::LimitExceeded {
                what: "field value",
                ..
            })
        ));
    }
}
