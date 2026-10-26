// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::Path;

use super::{
    primary_evidence, python_primary_evidence, python_supplemental_evidence,
    require_primary_evidence, supplemental_evidence, validate_evidence_test_path,
    validate_evidence_test_source, validate_exclusion_matcher,
    validate_python_evidence_test_source, CoverageExclusion, EvidenceClass, EvidenceLanguage,
    EvidenceTest, ExclusionMatcher,
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
fn external_module_evidence_inherits_parent_conditional_compilation() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let path = "crates/j2k-metal/tests/device.rs";
    let name = "full_classic_grayscale_decode_to_metal_matches_host_decode";

    validate_evidence_test_path(&root, path, name, EvidenceClass::Supplemental).unwrap();
    let error = validate_evidence_test_path(&root, path, name, EvidenceClass::Primary).unwrap_err();
    assert!(
        error.contains("conditionally compiled supplemental"),
        "{error}"
    );
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

#[test]
fn python_evidence_resolves_one_unittest_method_and_its_skip_conditions() {
    let runnable = "import unittest\n\nclass AdapterTests(unittest.TestCase):\n    def test_batch(self):\n        pass\n";
    validate_python_evidence_test_source("test_batch", EvidenceClass::Primary, runnable).unwrap();

    for (source, expected) in [
        ("class AdapterTests(unittest.TestCase):\n    def test_other(self):\n        pass\n", "no matching"),
        (
            "class AdapterTests(unittest.TestCase):\n    def test_batch(self):\n        pass\n    def test_batch(self):\n        pass\n",
            "ambiguous",
        ),
        ("class Helper:\n    def test_batch(self):\n        pass\n", "unittest.TestCase"),
        ("def test_batch(self):\n    pass\n", "no matching"),
        (
            "class AdapterTests(unittest.TestCase):\n    @unittest.skip(\"later\")\n    def test_batch(self):\n        pass\n",
            "must not be skipped",
        ),
        (
            "class AdapterTests(unittest.TestCase):\n    @unittest.expectedFailure\n    def test_batch(self):\n        pass\n",
            "must not be skipped",
        ),
    ] {
        let error = validate_python_evidence_test_source("test_batch", EvidenceClass::Primary, source)
            .unwrap_err();
        assert!(error.contains(expected), "{error}");
    }
    assert!(
        validate_python_evidence_test_source("batch", EvidenceClass::Primary, runnable)
            .unwrap_err()
            .contains("`test_` method")
    );

    for conditional in [
        "@unittest.skipUnless(AVAILABLE, \"needs hardware\")\nclass DeviceTests(unittest.TestCase):\n    def test_batch(self):\n        pass\n",
        "@unittest.skipUnless(\n    AVAILABLE, \"needs hardware\"\n)\n\nclass DeviceTests(unittest.TestCase):\n    def test_batch(self):\n        pass\n",
        "class DeviceTests(unittest.TestCase):\n    def test_first(self):\n        pass\n\n    @unittest.skipIf(\n        MISSING, \"needs hardware\"\n    )\n    def test_batch(self):\n        pass\n",
    ] {
        let error =
            validate_python_evidence_test_source("test_batch", EvidenceClass::Primary, conditional)
                .unwrap_err();
        assert!(error.contains("conditionally compiled supplemental"), "{error}");
        validate_python_evidence_test_source("test_batch", EvidenceClass::Supplemental, conditional)
            .unwrap();
    }
}

#[test]
fn python_extension_exclusion_covers_only_the_extension_sources() {
    let rule = |path| {
        super::matching_exclusion(path, 1, &[])
            .unwrap()
            .map(|rule| rule.id)
    };
    for path in [
        "crates/j2k-python/src/decode.rs",
        "crates/j2k-python/src/tensor.rs",
    ] {
        assert_eq!(rule(path), Some("python-extension-bindings"), "{path}");
    }
    for path in [
        "crates/j2k-python/build.rs",
        "crates/j2k-python-other/src/lib.rs",
        "crates/j2k/src/lib.rs",
        "crates/j2k-ml/src/lib.rs",
    ] {
        assert_eq!(rule(path), None, "{path}");
    }
}

#[test]
fn evidence_constructors_record_class_and_language() {
    let built = [
        primary_evidence("tests/gpu.rs", "rust_primary"),
        supplemental_evidence("tests/gpu.rs", "rust_supplemental"),
        python_primary_evidence("tests/test_gpu.py", "test_primary"),
        python_supplemental_evidence("tests/test_gpu.py", "test_supplemental"),
    ];

    assert_eq!(
        built.map(|evidence| (evidence.class, evidence.language)),
        [
            (EvidenceClass::Primary, EvidenceLanguage::Rust),
            (EvidenceClass::Supplemental, EvidenceLanguage::Rust),
            (EvidenceClass::Primary, EvidenceLanguage::Python),
            (EvidenceClass::Supplemental, EvidenceLanguage::Python),
        ]
    );
}

#[test]
fn path_prefix_exclusions_must_name_a_crate_source_directory() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let exclusion = |prefix| CoverageExclusion {
        id: "prefix-test",
        reason: "test fixture",
        matcher: ExclusionMatcher::PathPrefix { prefix },
        evidence: &[],
    };

    validate_exclusion_matcher(&root, &exclusion("crates/j2k-python/src/")).unwrap();
    let error = validate_exclusion_matcher(&root, &exclusion("crates/j2k-python/")).unwrap_err();
    assert!(error.contains("crate source directory"), "{error}");
    assert!(validate_exclusion_matcher(&root, &exclusion("crates/j2k-missing/src/")).is_err());
}
