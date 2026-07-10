// SPDX-License-Identifier: MIT OR Apache-2.0

//! JPEG 2000 marker codes (Table A.2).

/// Start of codestream - 'SOC'.
pub(crate) const SOC: u8 = 0x4F;
/// Start of tile-part - 'SOT'.
pub(crate) const SOT: u8 = 0x90;
/// Start of data - 'SOD'.
pub(crate) const SOD: u8 = 0x93;
/// End of codestream - 'EOC'.
pub(crate) const EOC: u8 = 0xD9;

/// Extended capabilities - 'CAP'.
pub(crate) const CAP: u8 = 0x50;
/// Image and tile size - 'SIZ'.
pub(crate) const SIZ: u8 = 0x51;

/// Coding style default - 'COD'.
pub(crate) const COD: u8 = 0x52;
/// Coding component - 'COC'.
pub(crate) const COC: u8 = 0x53;
/// Region-of-interest - 'RGN'.
pub(crate) const RGN: u8 = 0x5E;
/// Quantization default - 'QCD'.
pub(crate) const QCD: u8 = 0x5C;
/// Quantization component - 'QCC'.
pub(crate) const QCC: u8 = 0x5D;
/// Progression order change - 'POC'.
pub(crate) const POC: u8 = 0x5F;

/// Tile-part lengths - 'TLM'.
pub(crate) const TLM: u8 = 0x55;
/// Packet length, main header - 'PLM'.
pub(crate) const PLM: u8 = 0x57;
/// Packet length, tile-part header - 'PLT'.
pub(crate) const PLT: u8 = 0x58;
/// Corresponding profile - 'CPF'.
pub(crate) const CPF: u8 = 0x59;
/// Packed packet headers, main header - 'PPM'.
pub(crate) const PPM: u8 = 0x60;
/// Packed packet headers, tile-part header - 'PPT'.
pub(crate) const PPT: u8 = 0x61;

/// Start of packet - 'SOP'.
pub(crate) const SOP: u8 = 0x91;
/// End of packet header - 'EPH'.
pub(crate) const EPH: u8 = 0x92;

/// Component registration - 'CRG'.
pub(crate) const CRG: u8 = 0x63;
/// Comment - 'COM'.
pub(crate) const COM: u8 = 0x64;

#[expect(
    dead_code,
    reason = "not all marker codes are used in every decoding path yet"
)]
pub(crate) fn to_string(marker: u8) -> &'static str {
    match marker {
        // Delimiting markers.
        SOC => "SOC",
        SOT => "SOT",
        SOD => "SOD",
        EOC => "EOC",

        // Fixed information.
        CAP => "CAP",
        SIZ => "SIZ",

        // Functional markers.
        COD => "COD",
        COC => "COC",
        RGN => "RGN",
        QCD => "QCD",
        QCC => "QCC",
        POC => "POC",

        // Pointer markers.
        TLM => "TLM",
        PLM => "PLM",
        PLT => "PLT",
        CPF => "CPF",
        PPM => "PPM",
        PPT => "PPT",

        // In-bit-stream markers.
        SOP => "SOP",
        EPH => "EPH",

        // Informational markers.
        CRG => "CRG",
        COM => "COM",

        _ => "UNKNOWN",
    }
}
