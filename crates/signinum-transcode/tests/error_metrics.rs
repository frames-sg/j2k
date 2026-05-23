// SPDX-License-Identifier: Apache-2.0

use signinum_transcode::metrics::error_metrics_i32;
use signinum_transcode::TranscodeValidationClassification;

#[test]
fn error_metrics_report_exact_rate_max_abs_error_and_histogram() {
    let actual = [10, 11, 12, 13, 14];
    let expected = [10, 12, 12, 12, 16];

    let metrics = error_metrics_i32(&actual, &expected).expect("matching lengths");

    assert_eq!(metrics.total, 5);
    assert_eq!(metrics.exact_matches, 2);
    assert!((metrics.exact_match_rate() - 0.4).abs() <= f64::EPSILON);
    assert_eq!(metrics.max_abs_error, 2);
    assert_eq!(metrics.absolute_error_count(0), 2);
    assert_eq!(metrics.absolute_error_count(1), 2);
    assert_eq!(metrics.absolute_error_count(2), 1);
    assert!(!metrics.is_one_lsb_bounded(0.999));
}

#[test]
fn error_metrics_accept_one_lsb_bounded_claim_when_thresholds_pass() {
    let actual = [10, 11, 12, 13];
    let expected = [10, 11, 12, 14];

    let metrics = error_metrics_i32(&actual, &expected).expect("matching lengths");

    assert!(metrics.is_one_lsb_bounded(0.75));
    assert!(!metrics.is_one_lsb_bounded(0.999));
}

#[test]
fn transcode_validation_classification_applies_thresholds() {
    let exact = error_metrics_i32(&[4, 5, 6], &[4, 5, 6]).expect("matching lengths");
    assert_eq!(
        TranscodeValidationClassification::classify_metrics(&exact),
        TranscodeValidationClassification::Exact
    );

    let mut actual = vec![7; 1000];
    let expected = vec![7; 1000];
    actual[999] = 8;
    let one_lsb = error_metrics_i32(&actual, &expected).expect("matching lengths");
    assert_eq!(
        TranscodeValidationClassification::classify_metrics(&one_lsb),
        TranscodeValidationClassification::OneLsbBounded
    );

    actual[998] = 8;
    let outside = error_metrics_i32(&actual, &expected).expect("matching lengths");
    assert_eq!(
        TranscodeValidationClassification::classify_metrics(&outside),
        TranscodeValidationClassification::OutsideThreshold
    );
}

#[test]
fn error_metrics_reject_mismatched_lengths() {
    let err = error_metrics_i32(&[1, 2, 3], &[1, 2]).unwrap_err();

    assert_eq!(err.actual_len(), 3);
    assert_eq!(err.expected_len(), 2);
}
