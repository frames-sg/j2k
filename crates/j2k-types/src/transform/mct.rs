// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multi-component transform job contracts.

/// Forward reversible color transform job.
#[derive(Debug)]
pub struct J2kForwardRctJob<'a> {
    /// First component plane, updated in place.
    pub plane0: &'a mut [f32],
    /// Second component plane, updated in place.
    pub plane1: &'a mut [f32],
    /// Third component plane, updated in place.
    pub plane2: &'a mut [f32],
}

/// Forward irreversible color transform job.
#[derive(Debug)]
pub struct J2kForwardIctJob<'a> {
    /// First component plane, updated in place.
    pub plane0: &'a mut [f32],
    /// Second component plane, updated in place.
    pub plane1: &'a mut [f32],
    /// Third component plane, updated in place.
    pub plane2: &'a mut [f32],
}
