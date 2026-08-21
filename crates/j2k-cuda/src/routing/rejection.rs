// SPDX-License-Identifier: MIT OR Apache-2.0

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AutoCudaRejection {
    Ineligible,
    NotBenchmarkQualified,
}
