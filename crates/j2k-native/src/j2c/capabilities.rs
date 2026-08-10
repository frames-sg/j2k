// SPDX-License-Identifier: MIT OR Apache-2.0

//! JPEG 2000 CAP and CPF marker parsing shared by decode and inspection.

use alloc::vec::Vec;

mod magnitude;
pub(crate) use magnitude::encode_magnitude_bound;
pub use magnitude::required_magnitude_bound;

const HTJ2K_PCAP_MASK: u32 = 1 << 17;
pub(crate) const HTJ2K_RSIZ_MASK: u16 = 1 << 14;
const CCAP15_RESERVED_MASK: u16 = 0x07C0;

/// Part 15 code-block set declared by `Ccap15`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Htj2kCapabilityMode {
    /// Every code-block uses HT coding.
    HtOnly,
    /// HT coding is declared, but the codestream may use classic coding.
    HtDeclared,
    /// Classic and HT code-block coding may be mixed.
    Mixed,
}

/// Corresponding-profile words parsed from one CPF marker segment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct J2kCorrespondingProfile {
    words: Vec<u16>,
}

impl J2kCorrespondingProfile {
    /// Return the `Pcpf_i` words in codestream order.
    #[must_use]
    pub fn words(&self) -> &[u16] {
        &self.words
    }

    /// Return `CPFnum` when its little-word-endian representation fits `u64`.
    #[must_use]
    pub fn number_u64(&self) -> Option<u64> {
        profile_number_u64(self.words.len(), self.words.iter().copied())
    }
}

/// Parsed Part 15 CAP/CPF facts plus the default COD HT style bits.
#[derive(Clone, Debug, Eq, PartialEq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the booleans expose independent Part 15 CAP and COD capability facts"
)]
pub struct Htj2kCapabilities {
    pcap: u32,
    ccap15: u16,
    mode: Htj2kCapabilityMode,
    multiple_ht_sets: bool,
    roi: bool,
    heterogeneous: bool,
    ht_irreversible: bool,
    magnitude_bound: u8,
    quality_layers: u8,
    default_ht_block_coding: bool,
    default_mixed_block_coding: bool,
    corresponding_profile: Option<J2kCorrespondingProfile>,
}

impl Htj2kCapabilities {
    /// Raw `Pcap` word from CAP.
    #[must_use]
    pub const fn pcap(&self) -> u32 {
        self.pcap
    }

    /// Raw Part 15 capability word.
    #[must_use]
    pub const fn ccap15(&self) -> u16 {
        self.ccap15
    }

    /// Declared HT code-block set.
    #[must_use]
    pub const fn mode(&self) -> Htj2kCapabilityMode {
        self.mode
    }

    /// Whether `Ccap15` advertises multiple HT sets.
    #[must_use]
    pub const fn multiple_ht_sets(&self) -> bool {
        self.multiple_ht_sets
    }

    /// Whether `Ccap15` advertises RGN use.
    #[must_use]
    pub const fn roi(&self) -> bool {
        self.roi
    }

    /// Whether `Ccap15` advertises heterogeneous HT sets.
    #[must_use]
    pub const fn heterogeneous(&self) -> bool {
        self.heterogeneous
    }

    /// Whether `Ccap15` advertises irreversible HT coding.
    #[must_use]
    pub const fn ht_irreversible(&self) -> bool {
        self.ht_irreversible
    }

    /// Decoded `BMAGB` magnitude bound in the range 8 through 74.
    #[must_use]
    pub const fn magnitude_bound(&self) -> u8 {
        self.magnitude_bound
    }

    /// Number of quality layers declared by the main COD marker.
    #[must_use]
    pub const fn quality_layers(&self) -> u8 {
        self.quality_layers
    }

    /// Whether the main COD marker selects HT block coding by default.
    #[must_use]
    pub const fn default_ht_block_coding(&self) -> bool {
        self.default_ht_block_coding
    }

    /// Whether the main COD marker permits mixed classic/HT block coding.
    #[must_use]
    pub const fn default_mixed_block_coding(&self) -> bool {
        self.default_mixed_block_coding
    }

    /// Corresponding profile advertised by CPF, when present.
    #[must_use]
    pub const fn corresponding_profile(&self) -> Option<&J2kCorrespondingProfile> {
        self.corresponding_profile.as_ref()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CapabilityMarkerError {
    Cap(&'static str),
    Cpf(&'static str),
    Allocation { bytes: usize },
}

impl CapabilityMarkerError {
    pub(crate) const fn marker_label(self) -> &'static str {
        match self {
            Self::Cap(what) | Self::Cpf(what) => what,
            Self::Allocation { .. } => "CPF profile allocation",
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the booleans preserve independent validated Ccap15 flag bits"
)]
struct Htj2kCapabilityCore {
    pcap: u32,
    ccap15: u16,
    mode: Htj2kCapabilityMode,
    multiple_ht_sets: bool,
    roi: bool,
    heterogeneous: bool,
    ht_irreversible: bool,
    magnitude_bound: u8,
}

#[derive(Default)]
pub(crate) struct CapabilityMarkerState<'a> {
    saw_cap: bool,
    htj2k: Option<Htj2kCapabilityCore>,
    cpf_payload: Option<&'a [u8]>,
}

impl<'a> CapabilityMarkerState<'a> {
    pub(crate) fn record_cap(&mut self, payload: &'a [u8]) -> Result<(), CapabilityMarkerError> {
        if self.saw_cap {
            return Err(CapabilityMarkerError::Cap("duplicate CAP"));
        }
        self.saw_cap = true;
        self.htj2k = parse_cap(payload)?;
        Ok(())
    }

    pub(crate) fn record_cpf(&mut self, payload: &'a [u8]) -> Result<(), CapabilityMarkerError> {
        if self.cpf_payload.is_some() {
            return Err(CapabilityMarkerError::Cpf("duplicate CPF"));
        }
        validate_cpf(payload)?;
        self.cpf_payload = Some(payload);
        Ok(())
    }

    pub(crate) fn validate_rsiz(&self, rsiz: u16) -> Result<(), CapabilityMarkerError> {
        match (rsiz & HTJ2K_RSIZ_MASK != 0, self.htj2k.is_some()) {
            (true, false) => Err(CapabilityMarkerError::Cap(
                "SIZ advertises Part 15 without Pcap15",
            )),
            (false, true) => Err(CapabilityMarkerError::Cap(
                "Pcap15 is present without the Part 15 SIZ capability",
            )),
            _ => Ok(()),
        }?;
        if self.cpf_payload.is_some() && self.htj2k.is_none() {
            return Err(CapabilityMarkerError::Cpf(
                "CPF is present without Part 15 capabilities",
            ));
        }
        Ok(())
    }

    pub(crate) fn high_throughput(&self) -> bool {
        self.htj2k.is_some()
    }

    pub(crate) fn to_public(
        &self,
        quality_layers: u8,
        default_ht_block_coding: bool,
        default_mixed_block_coding: bool,
    ) -> Result<Option<Htj2kCapabilities>, CapabilityMarkerError> {
        let Some(core) = self.htj2k else {
            return Ok(None);
        };
        let corresponding_profile = self
            .cpf_payload
            .map(J2kCorrespondingProfile::from_payload)
            .transpose()?;
        Ok(Some(Htj2kCapabilities {
            pcap: core.pcap,
            ccap15: core.ccap15,
            mode: core.mode,
            multiple_ht_sets: core.multiple_ht_sets,
            roi: core.roi,
            heterogeneous: core.heterogeneous,
            ht_irreversible: core.ht_irreversible,
            magnitude_bound: core.magnitude_bound,
            quality_layers,
            default_ht_block_coding,
            default_mixed_block_coding,
            corresponding_profile,
        }))
    }
}

impl J2kCorrespondingProfile {
    fn from_payload(payload: &[u8]) -> Result<Self, CapabilityMarkerError> {
        let word_count = payload.len() / 2;
        let bytes = word_count
            .checked_mul(core::mem::size_of::<u16>())
            .ok_or(CapabilityMarkerError::Allocation { bytes: usize::MAX })?;
        let mut words = Vec::new();
        words
            .try_reserve_exact(word_count)
            .map_err(|_| CapabilityMarkerError::Allocation { bytes })?;
        words.extend(
            payload
                .chunks_exact(2)
                .map(|word| u16::from_be_bytes([word[0], word[1]])),
        );
        Ok(Self { words })
    }
}

fn parse_cap(payload: &[u8]) -> Result<Option<Htj2kCapabilityCore>, CapabilityMarkerError> {
    if payload.len() < 4 {
        return Err(CapabilityMarkerError::Cap(
            "CAP payload is shorter than Pcap",
        ));
    }
    let pcap = u32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let expected_len = 4usize
        .checked_add(pcap.count_ones() as usize * 2)
        .ok_or(CapabilityMarkerError::Cap("CAP payload length overflows"))?;
    if payload.len() != expected_len {
        return Err(CapabilityMarkerError::Cap(
            "CAP payload does not match the Pcap capability count",
        ));
    }
    if pcap & HTJ2K_PCAP_MASK == 0 {
        return Ok(None);
    }
    let preceding_capabilities = (pcap >> 18).count_ones() as usize;
    let offset = 4 + preceding_capabilities * 2;
    let ccap15 = u16::from_be_bytes([payload[offset], payload[offset + 1]]);
    if ccap15 & CCAP15_RESERVED_MASK != 0 {
        return Err(CapabilityMarkerError::Cap("CAP reserved Ccap15 bits"));
    }
    let mode = match ccap15 >> 14 {
        0 => Htj2kCapabilityMode::HtOnly,
        2 => Htj2kCapabilityMode::HtDeclared,
        3 => Htj2kCapabilityMode::Mixed,
        _ => return Err(CapabilityMarkerError::Cap("CAP reserved HT mode")),
    };
    Ok(Some(Htj2kCapabilityCore {
        pcap,
        ccap15,
        mode,
        multiple_ht_sets: ccap15 & (1 << 13) != 0,
        roi: ccap15 & (1 << 12) != 0,
        heterogeneous: ccap15 & (1 << 11) != 0,
        ht_irreversible: ccap15 & (1 << 5) != 0,
        magnitude_bound: decode_magnitude_bound((ccap15 & 0x1F) as u8),
    }))
}

fn validate_cpf(payload: &[u8]) -> Result<(), CapabilityMarkerError> {
    if payload.is_empty() || !payload.len().is_multiple_of(2) {
        return Err(CapabilityMarkerError::Cpf(
            "CPF payload must contain complete profile words",
        ));
    }
    if payload[payload.len() - 2..] == [0, 0] {
        return Err(CapabilityMarkerError::Cpf(
            "CPF final profile word must be non-zero",
        ));
    }
    Ok(())
}

fn profile_number_u64(word_count: usize, words: impl Iterator<Item = u16>) -> Option<u64> {
    if word_count > 4 {
        return None;
    }
    let mut encoded = 0u64;
    for (index, word) in words.enumerate() {
        encoded = encoded.checked_add(u64::from(word) << (index * 16))?;
    }
    encoded.checked_sub(1)
}

const fn decode_magnitude_bound(p: u8) -> u8 {
    match p {
        0 => 8,
        1..=19 => p + 8,
        20..=30 => 4 * (p - 19) + 27,
        _ => 74,
    }
}
