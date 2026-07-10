use serde_json::Value;

pub(crate) const PUBLICATION_GATE_KEYS: [&str; 3] = [
    "publication_eligible",
    "publication_blockers",
    "benchmark_complete",
];

#[derive(Debug, Clone, PartialEq, Eq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each boolean records an independent fail-closed publication-gate condition"
)]
struct PublicationGateEvaluation {
    metadata_present: bool,
    eligible: bool,
    blockers_clean: bool,
    benchmark_complete: bool,
    blockers: String,
}

impl PublicationGateEvaluation {
    fn from_metadata(metadata: Option<&Value>) -> Self {
        let Some(metadata) = metadata else {
            return Self {
                metadata_present: false,
                eligible: false,
                blockers_clean: false,
                benchmark_complete: false,
                blockers: "not-recorded".to_string(),
            };
        };
        let blockers = metadata
            .get("publication_blockers")
            .and_then(Value::as_str)
            .unwrap_or("not-recorded")
            .to_string();
        Self {
            metadata_present: true,
            eligible: metadata.get("publication_eligible").and_then(Value::as_str) == Some("true"),
            blockers_clean: blockers == "none",
            benchmark_complete: metadata.get("benchmark_complete").and_then(Value::as_str)
                == Some("true"),
            blockers,
        }
    }

    fn collect_issues(&self, label: &str, issues: &mut Vec<String>) {
        if !self.metadata_present {
            issues.push(format!("{label} metadata missing"));
            return;
        }
        if !self.eligible {
            issues.push(format!(
                "{label} publication_eligible=false blockers={}",
                self.blockers
            ));
        }
        if !self.blockers_clean {
            issues.push(format!("{label} publication_blockers={}", self.blockers));
        }
        if !self.benchmark_complete {
            issues.push(format!("{label} benchmark_complete is not true"));
        }
    }
}

pub(crate) fn collect_publication_gate_issues(
    label: &str,
    metadata: Option<&Value>,
    issues: &mut Vec<String>,
) {
    PublicationGateEvaluation::from_metadata(metadata).collect_issues(label, issues);
}

#[cfg(test)]
mod tests {
    use super::collect_publication_gate_issues;
    use serde_json::json;

    #[test]
    fn publication_gate_accepts_clean_writer_metadata() {
        let metadata = json!({
            "publication_eligible": "true",
            "publication_blockers": "none",
            "benchmark_complete": "true",
        });
        let mut issues = Vec::new();

        collect_publication_gate_issues("cpu-fixture-compare", Some(&metadata), &mut issues);

        assert!(issues.is_empty(), "unexpected issues: {issues:?}");
    }

    #[test]
    fn publication_gate_rejects_failed_writer_metadata() {
        let metadata = json!({
            "publication_eligible": "false",
            "publication_blockers": "generated-fixtures-included",
            "benchmark_complete": "false",
        });
        let mut issues = Vec::new();

        collect_publication_gate_issues("cpu-fixture-compare", Some(&metadata), &mut issues);

        assert_eq!(
            issues,
            vec![
                "cpu-fixture-compare publication_eligible=false blockers=generated-fixtures-included",
                "cpu-fixture-compare publication_blockers=generated-fixtures-included",
                "cpu-fixture-compare benchmark_complete is not true",
            ]
        );
    }
}
