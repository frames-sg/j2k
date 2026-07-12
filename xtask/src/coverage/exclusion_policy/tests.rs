// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    require_primary_evidence, supplemental_evidence, validate_evidence_test_source,
    CoverageExclusion, EvidenceClass, EvidenceTest, ExclusionMatcher,
};

mod matchers;

#[test]
fn evidence_resolution_ignores_comments_and_string_literals() {
    let spoofed = r#"
// fn hardware_parity(
const SPOOF: &str = "fn hardware_parity(";
fn helper() {}
"#;

    let error = validate_evidence_test_source(
        "tests/spoof.rs",
        "hardware_parity",
        EvidenceClass::Primary,
        spoofed,
    )
    .unwrap_err();
    assert!(
        error.contains("no matching Rust function symbol"),
        "{error}"
    );
}

#[test]
fn evidence_resolution_requires_one_unconditional_runnable_test() {
    let helper = "fn hardware_parity() {}\n";
    assert!(validate_evidence_test_source(
        "tests/helper.rs",
        "hardware_parity",
        EvidenceClass::Primary,
        helper,
    )
    .unwrap_err()
    .contains("#[test]"));

    let ignored = "#[test]\n#[ignore]\nfn hardware_parity() {}\n";
    assert!(validate_evidence_test_source(
        "tests/ignored.rs",
        "hardware_parity",
        EvidenceClass::Primary,
        ignored,
    )
    .unwrap_err()
    .contains("must not be ignored"));

    let runnable = "#[test]\nfn hardware_parity() {}\n";
    validate_evidence_test_source(
        "tests/runnable.rs",
        "hardware_parity",
        EvidenceClass::Primary,
        runnable,
    )
    .unwrap();
}

#[test]
fn direct_and_inherited_cfg_require_supplemental_classification() {
    for conditional in [
        "#[cfg(feature = \"gpu\")]\n#[test]\nfn hardware_parity() {}\n",
        "#![cfg(target_os = \"macos\")]\n#[test]\nfn hardware_parity() {}\n",
        "#[cfg(any())]\nmod gated { #[test] fn hardware_parity() {} }\n",
        "#[cfg_attr(feature = \"gpu\", cfg(any()))]\nmod gated { #[test] fn hardware_parity() {} }\n",
    ] {
        let error = validate_evidence_test_source(
            "tests/conditional.rs",
            "hardware_parity",
            EvidenceClass::Primary,
            conditional,
        )
        .unwrap_err();
        assert!(error.contains("conditionally compiled supplemental"), "{error}");
        validate_evidence_test_source(
            "tests/conditional.rs",
            "hardware_parity",
            EvidenceClass::Supplemental,
            conditional,
        )
        .unwrap();
    }
}

#[test]
fn exact_enclosing_cfg_test_is_harness_plumbing() {
    let source = "#[cfg(test)]\nmod tests { #[test] fn hardware_parity() {} }\n";
    validate_evidence_test_source(
        "src/lib.rs",
        "hardware_parity",
        EvidenceClass::Primary,
        source,
    )
    .unwrap();

    let error = validate_evidence_test_source(
        "src/lib.rs",
        "hardware_parity",
        EvidenceClass::Supplemental,
        source,
    )
    .unwrap_err();
    assert!(error.contains("unconditional primary"), "{error}");
}

#[test]
fn supplemental_only_exclusion_evidence_is_rejected() {
    const EVIDENCE: &[EvidenceTest] = &[supplemental_evidence("tests/gpu.rs", "hardware_parity")];
    let exclusion = CoverageExclusion {
        id: "supplemental-only",
        reason: "adversarial test",
        matcher: ExclusionMatcher::WholeFile {
            path: "tests/gpu.rs",
        },
        evidence: EVIDENCE,
    };

    let error = require_primary_evidence(&exclusion).unwrap_err();
    assert!(
        error.contains("at least one unconditional primary"),
        "{error}"
    );
}
