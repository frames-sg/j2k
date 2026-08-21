// SPDX-License-Identifier: MIT OR Apache-2.0

//! Encode-stage dispatch accounting.

/// Encode-stage dispatch counters reported by an accelerator.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct J2kEncodeDispatchReport {
    /// Pixel deinterleave/level-shift dispatch count.
    pub deinterleave: usize,
    /// Forward RCT kernel dispatch count.
    pub forward_rct: usize,
    /// Forward ICT kernel dispatch count.
    pub forward_ict: usize,
    /// Forward reversible 5/3 DWT kernel dispatch count.
    pub forward_dwt53: usize,
    /// Forward irreversible 9/7 DWT kernel dispatch count.
    pub forward_dwt97: usize,
    /// Subband quantization dispatch count.
    pub quantize_subband: usize,
    /// Tier-1 code-block encode dispatch count.
    pub tier1_code_block: usize,
    /// HTJ2K code-block encode dispatch count.
    pub ht_code_block: usize,
    /// Packetization dispatch count.
    pub packetization: usize,
}

impl J2kEncodeDispatchReport {
    /// Return the saturating per-stage delta from `before` to `self`.
    #[must_use]
    pub fn saturating_delta(self, before: Self) -> Self {
        Self {
            deinterleave: self.deinterleave.saturating_sub(before.deinterleave),
            forward_rct: self.forward_rct.saturating_sub(before.forward_rct),
            forward_ict: self.forward_ict.saturating_sub(before.forward_ict),
            forward_dwt53: self.forward_dwt53.saturating_sub(before.forward_dwt53),
            forward_dwt97: self.forward_dwt97.saturating_sub(before.forward_dwt97),
            quantize_subband: self
                .quantize_subband
                .saturating_sub(before.quantize_subband),
            tier1_code_block: self
                .tier1_code_block
                .saturating_sub(before.tier1_code_block),
            ht_code_block: self.ht_code_block.saturating_sub(before.ht_code_block),
            packetization: self.packetization.saturating_sub(before.packetization),
        }
    }

    /// Return total dispatches across all encode stages.
    #[must_use]
    pub fn total(self) -> usize {
        self.forward_rct
            .saturating_add(self.deinterleave)
            .saturating_add(self.forward_ict)
            .saturating_add(self.forward_dwt53)
            .saturating_add(self.forward_dwt97)
            .saturating_add(self.quantize_subband)
            .saturating_add(self.tier1_code_block)
            .saturating_add(self.ht_code_block)
            .saturating_add(self.packetization)
    }

    /// Return whether at least one encode stage dispatched.
    #[must_use]
    pub fn any(self) -> bool {
        self.total() > 0
    }
}
