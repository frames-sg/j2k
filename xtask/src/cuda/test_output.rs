//! Strict parsing for Rust test summaries emitted by CUDA validation commands.

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct TestSummary {
    pub(super) passed: usize,
    pub(super) failed: usize,
    pub(super) ignored: usize,
    pub(super) measured: usize,
    pub(super) filtered_out: usize,
}

impl TestSummary {
    fn parse(line: &str) -> Option<Self> {
        let counts = line.trim().strip_prefix("test result: ok.")?;
        let mut fields = counts.split(';').map(str::trim);
        let passed = parse_count(fields.next()?, "passed")?;
        let failed = parse_count(fields.next()?, "failed")?;
        let ignored = parse_count(fields.next()?, "ignored")?;
        let measured = parse_count(fields.next()?, "measured")?;
        let filtered_out = parse_count(fields.next()?, "filtered out")?;
        if fields
            .next()
            .is_some_and(|timing| !timing.starts_with("finished in "))
            || fields.next().is_some()
        {
            return None;
        }
        Some(Self {
            passed,
            failed,
            ignored,
            measured,
            filtered_out,
        })
    }

    pub(super) const fn add(self, other: Self) -> Self {
        Self {
            passed: self.passed + other.passed,
            failed: self.failed + other.failed,
            ignored: self.ignored + other.ignored,
            measured: self.measured + other.measured,
            filtered_out: self.filtered_out + other.filtered_out,
        }
    }
}

fn parse_count(field: &str, suffix: &str) -> Option<usize> {
    field
        .strip_suffix(suffix)?
        .trim()
        .trim_end_matches('.')
        .parse()
        .ok()
}

pub(super) fn successful_test_summaries(output: &str) -> Vec<TestSummary> {
    output.lines().filter_map(TestSummary::parse).collect()
}
