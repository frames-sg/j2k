// SPDX-License-Identifier: MIT OR Apache-2.0

pub(super) fn assert_restart_counts(
    consumers: &str,
    lossless: &str,
    progressive: &str,
    sequential_restart: &str,
    sequential_dct: &str,
    checkpoint: &str,
) {
    assert!(
        !consumers.contains("ensure_bits(1)"),
        "restart consumers must not recreate the ignored boundary-probe Result"
    );
    assert_eq!(
        consumers.matches(".consume_restart_marker(").count(),
        5,
        "the five entropy restart call sites must delegate to BitReader"
    );
    assert_eq!(lossless.matches(".consume_restart_marker(").count(), 2);
    assert_eq!(progressive.matches(".consume_restart_marker(").count(), 1);
    assert_eq!(
        sequential_restart
            .matches(".consume_restart_marker(")
            .count(),
        1
    );
    assert_eq!(
        sequential_dct.matches(".consume_restart_marker(").count(),
        0
    );
    assert_eq!(checkpoint.matches(".consume_restart_marker(").count(), 1);

    assert_eq!(
        consumers.matches(".reset_at_restart()").count(),
        1,
        "only lossless may discard padding before the shared validated reset"
    );
    assert_eq!(
        lossless.matches("JpegError::RestartMismatch").count(),
        1,
        "the raw restart-index scanner remains a separate byte-offset abstraction"
    );
    for (name, source) in [
        ("progressive scan", progressive),
        ("sequential restart", sequential_restart),
        ("sequential DCT", sequential_dct),
        ("checkpoint builder", checkpoint),
    ] {
        assert!(
            !source.contains("JpegError::RestartMismatch"),
            "{name} must leave restart validation to BitReader"
        );
    }
}
