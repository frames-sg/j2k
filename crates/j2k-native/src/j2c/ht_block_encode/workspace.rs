// SPDX-License-Identifier: MIT OR Apache-2.0

use super::writers::{MagSgnEncoder, MelEncoder, VlcEncoder};
use super::writers::{MEL_SIZE, MS_SIZE, VLC_SIZE};
use crate::EncodeResult;

/// Reusable scalar cleanup-pass reservoirs owned once per CPU worker.
pub(crate) struct HtEncodeWorkspace {
    pub(super) mel: MelEncoder,
    pub(super) vlc: VlcEncoder,
    pub(super) mag_sgn: MagSgnEncoder,
}

impl HtEncodeWorkspace {
    pub(crate) const ALLOCATION_BYTES: usize = MEL_SIZE + VLC_SIZE + MS_SIZE;

    pub(crate) fn try_new() -> EncodeResult<Self> {
        Ok(Self {
            mel: MelEncoder::try_new()?,
            vlc: VlcEncoder::try_new()?,
            mag_sgn: MagSgnEncoder::try_new()?,
        })
    }

    pub(super) fn reset_cleanup(&mut self) {
        self.mel.reset();
        self.vlc.reset();
        self.mag_sgn.reset();
    }
}
