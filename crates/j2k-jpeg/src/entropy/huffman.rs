// SPDX-License-Identifier: MIT OR Apache-2.0

//! Huffman decoder with fused symbol-and-magnitude lookups.
//!
//! Small baseline scans use a 256-entry table indexed by the next 8 bits.
//! Larger sequential scans use a 4096-entry 12-bit table, and densely coded AC
//! scans can also select 32768 entries and 15 bits. The prepared plan chooses
//! the primary width once; all tables remain immutable and cache-shareable.
//!
//! Full decode tables retain the fixed 15-bit raw-symbol and fused lookups.
//! Small baseline images use 8-bit raw-symbol fallbacks and the 8-bit primary
//! lookup, omitting wider lookups they cannot select. Full tables also retain
//! a 256-entry primary companion so a stronger cached table satisfies a later
//! narrow request without another allocation.
//! Progressive-only tables compact their raw-symbol prefix lookup to the
//! longest code; every path preserves input and error behavior.
//!
//! Built once from [`crate::parse::tables::RawHuffmanTable`]; read many times
//! by [`crate::entropy::block::decode_block_with_activity`].

use crate::allocation::{checked_add_allocation_bytes, try_reserve_for_len_with_live_budget};
use crate::error::{HuffmanFailure, JpegError};
use crate::internal::bit_reader::BitReader;
use crate::parse::tables::{HuffmanTableRole, HuffmanValues, RawHuffmanTable};
use crate::standard_huffman::{
    STD_CHROMA_AC_BITS, STD_CHROMA_AC_VALUES, STD_CHROMA_DC_BITS, STD_CHROMA_DC_VALUES,
    STD_LUMA_AC_BITS, STD_LUMA_AC_VALUES, STD_LUMA_DC_BITS, STD_LUMA_DC_VALUES,
};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::{mem::size_of, num::NonZeroU32};

/// Number of fast-lookup entries. One per possible 15-bit peek value.
const FAST_BITS: u8 = 15;
const FAST_ENTRIES: usize = 1 << FAST_BITS;
const NARROW_RAW_BITS: u8 = 8;
const NARROW_RAW_ENTRIES: usize = 1 << NARROW_RAW_BITS;
// Empty raw tables retain one addressable lookup bit; non-empty Raw tables use
// their longest canonical code length up to the 15-bit lookup cap.
const MIN_DYNAMIC_RAW_BITS: u8 = 1;
const NARROW_PRIMARY_BITS: u8 = 8;
pub(crate) const NARROW_PRIMARY_ENTRIES: usize = 1 << NARROW_PRIMARY_BITS;
const FULL_PRIMARY_BITS: u8 = 12;
pub(crate) const FULL_PRIMARY_ENTRIES: usize = 1 << FULL_PRIMARY_BITS;
static EMPTY_PACKED_LOOKUP: [u32; FAST_ENTRIES] = [0; FAST_ENTRIES];

const AC_FAST_KIND_SHIFT: u32 = 28;
pub(crate) const AC_FAST_KIND_MASK: u32 = 0xF << AC_FAST_KIND_SHIFT;
pub(crate) const AC_FAST_VALUE: u32 = 1 << AC_FAST_KIND_SHIFT;
pub(crate) const AC_FAST_EOB: u32 = 2 << AC_FAST_KIND_SHIFT;
pub(crate) const AC_FAST_ZRL: u32 = 3 << AC_FAST_KIND_SHIFT;
const AC_FAST_LEN_MASK: u32 = 0x0F;
const AC_FAST_RUN_MASK: u32 = 0xF0;
const AC_FAST_VALUE_SHIFT: u32 = 8;

const DC_FAST_LEN_MASK: u32 = 0x0F;
const DC_FAST_VALUE_SHIFT: u32 = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HuffmanTable {
    Static(&'static StaticCompiledHuffmanTable),
    Dynamic(Arc<CompiledHuffmanTable>),
}

/// Lookup payload requested by a prepared decode plan. The declaration order
/// is the capability order used when a cache entry satisfies a later request.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum HuffmanLookupRequirement {
    Raw,
    FusedNarrow,
    FusedFull,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct CompiledHuffmanTable {
    /// Raw-symbol prefix lookup. The paired metadata shift selects its width;
    /// `bit_length == 0` means the canonical fallback must continue decoding.
    fast: Vec<(u8, u8)>,
    /// Slow path, indexed by code length `l` ∈ `1..=16`:
    /// - `min_code[l]`: smallest `l`-bit code; `i32::MAX` if no `l`-bit code.
    /// - `max_code[l]`: largest `l`-bit code; `-1` if no `l`-bit code.
    /// - `val_offset[l]`: index into `values` where `l`-bit symbols begin,
    ///   pre-adjusted by subtracting `min_code[l]` so
    ///   `symbol = values[code + val_offset[l]]`.
    metadata: HuffmanMetadata,
    primary: Vec<u32>,
    /// Full tables retain this narrow companion so cached Full capability can
    /// serve a later 8-bit primary request without rebuilding the table.
    primary8: Vec<u32>,
    /// Wide fused AC entries. DC tables leave this empty because every DC
    /// decode uses `primary` or the raw-symbol table.
    packed: Vec<u32>,
    retained_allocation_bytes: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct StaticCompiledHuffmanTable {
    fast: &'static [(u8, u8); FAST_ENTRIES],
    metadata: HuffmanMetadata,
    primary: &'static [u32; FULL_PRIMARY_ENTRIES],
    primary8: &'static [u32; NARROW_PRIMARY_ENTRIES],
    packed: Option<&'static [u32; FAST_ENTRIES]>,
}

#[derive(Debug, PartialEq, Eq)]
struct HuffmanMetadata {
    /// Exact source identity retained beside the already-owned values and role.
    bits: [u8; 16],
    min_code: [i32; 17],
    max_code: [i32; 17],
    val_offset: [i32; 17],
    values: HuffmanValues,
    role: HuffmanTableRole,
    raw_lookup_shift: u8,
}

/// Thin borrowed view over a variable-width raw-symbol lookup. The pointer is
/// derived from the complete slice and the marker retains its borrow lifetime.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
struct RawSymbolLookup<'a> {
    ptr: *const (u8, u8),
    marker: core::marker::PhantomData<&'a [(u8, u8)]>,
}

impl RawSymbolLookup<'_> {
    #[expect(
        clippy::inline_always,
        reason = "progressive raw Huffman lookup is in every symbol loop"
    )]
    #[inline(always)]
    unsafe fn get_unchecked(self, index: usize) -> (u8, u8) {
        // SAFETY: the caller shifts a 15-bit peek by the paired metadata shift;
        // construction allocates exactly the resulting index range.
        unsafe { *self.ptr.add(index) }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RawHuffmanDecoder<'a> {
    fast: RawSymbolLookup<'a>,
    metadata: &'a HuffmanMetadata,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DcHuffmanTable<'a, const N: usize = FULL_PRIMARY_ENTRIES> {
    fast: RawSymbolLookup<'a>,
    metadata: &'a HuffmanMetadata,
    primary: &'a [u32; N],
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AcHuffmanTable<'a, const N: usize = FULL_PRIMARY_ENTRIES> {
    fast: RawSymbolLookup<'a>,
    metadata: &'a HuffmanMetadata,
    primary: &'a [u32; N],
    packed: &'a [u32; FAST_ENTRIES],
    wide: bool,
    wide_available: bool,
}

/// Checked handle into the one compiled-table arena retained by a decoder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PreparedHuffmanTableId(NonZeroU32);

/// Compiled table handles are immutable and shared by contexts and plans. The
/// handle arena and each referenced payload are charged conservatively per
/// owner so cache eviction cannot make a live plan disappear from accounting.
#[derive(Debug)]
pub(crate) struct PreparedHuffmanTables {
    entries: Arc<Vec<HuffmanTable>>,
    retained_table_bytes: usize,
}

pub(crate) type CanonicalHuffmanDerivation = j2k_codec_math::jpeg::CanonicalHuffmanDerivation;

pub(crate) fn derive_canonical_huffman(
    raw: &RawHuffmanTable,
) -> Result<CanonicalHuffmanDerivation, JpegError> {
    j2k_codec_math::jpeg::derive_canonical_huffman(&raw.bits, raw.values.len()).map_err(|_| {
        JpegError::HuffmanDecode {
            mcu: 0,
            reason: HuffmanFailure::CodeOverflow,
        }
    })
}

struct CompactRawCanonicalDerivation {
    min_code: [i32; 17],
    max_code: [i32; 17],
    val_offset: [i32; 17],
    lookup_bits: u8,
}

fn canonical_huffman_error() -> JpegError {
    JpegError::HuffmanDecode {
        mcu: 0,
        reason: HuffmanFailure::CodeOverflow,
    }
}

/// Validate an untrusted Raw-only table and retain only the canonical bounds
/// used by the decoder fallback. Prefix codes are replayed directly into the
/// lookup after its fallible allocation succeeds.
fn derive_compact_raw_canonical_huffman(
    raw: &RawHuffmanTable,
) -> Result<CompactRawCanonicalDerivation, JpegError> {
    let values_len = raw.values.len();
    if values_len > 256 {
        return Err(canonical_huffman_error());
    }

    // Preserve the full derivation's error precedence: validate the complete
    // BITS/HUFFVAL count before checking whether the canonical tree overflows.
    // Sixteen byte-sized counts sum to at most 4,080, so u16 is exact.
    let value_count = raw.bits.iter().copied().map(u16::from).sum::<u16>();
    if usize::from(value_count) != values_len {
        return Err(canonical_huffman_error());
    }

    let mut longest_len = MIN_DYNAMIC_RAW_BITS;
    let mut min_code = [i32::MAX; 17];
    let mut max_code = [-1i32; 17];
    let mut val_offset = [0i32; 17];
    let mut code = 0u32;
    let mut cursor = 0usize;
    let mut len = 1u8;
    while len <= 16 {
        if len != 1 {
            code <<= 1;
        }
        let count = usize::from(raw.bits[usize::from(len - 1)]);
        if count != 0 {
            longest_len = len;
            let next_code = code
                .checked_add(u32::try_from(count).map_err(|_| canonical_huffman_error())?)
                .ok_or_else(canonical_huffman_error)?;
            // Equality is the complete decoder table accepted by the full
            // canonical derivation. Only an over-subscribed tree is invalid.
            if next_code > (1u32 << len) {
                return Err(canonical_huffman_error());
            }
            let index = usize::from(len);
            let first_code = i32::try_from(code).map_err(|_| canonical_huffman_error())?;
            min_code[index] = first_code;
            max_code[index] =
                i32::try_from(next_code - 1).map_err(|_| canonical_huffman_error())?;
            val_offset[index] =
                i32::try_from(cursor).map_err(|_| canonical_huffman_error())? - first_code;
            code = next_code;
            cursor += count;
        }
        len += 1;
    }

    Ok(CompactRawCanonicalDerivation {
        min_code,
        max_code,
        val_offset,
        lookup_bits: longest_len.clamp(MIN_DYNAMIC_RAW_BITS, FAST_BITS),
    })
}

fn charge_live_bytes(
    live_bytes: &mut usize,
    additional: usize,
    cap: usize,
) -> Result<(), JpegError> {
    let requested = live_bytes
        .checked_add(additional)
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap,
        })?;
    if requested > cap {
        return Err(JpegError::MemoryCapExceeded { requested, cap });
    }
    *live_bytes = requested;
    Ok(())
}

const fn fill_fast_range(output: &mut [(u8, u8)], start: usize, end: usize, value: (u8, u8)) {
    if start >= end {
        return;
    }
    let (up_to_end, _) = output.split_at_mut(end);
    let (_, range) = up_to_end.split_at_mut(start);
    let mut index = 0;
    while index < range.len() {
        range[index] = value;
        index += 1;
    }
}

fn fill_raw_lookup_from_bits(
    output: &mut [(u8, u8)],
    lookup_bits: u8,
    raw: &RawHuffmanTable,
    canonical: &CompactRawCanonicalDerivation,
) -> Result<(), JpegError> {
    debug_assert_eq!(output.len(), 1usize << lookup_bits);
    let values = raw.values.as_slice();
    let mut value_index = 0usize;
    let mut len = 1u8;
    while len <= 16 {
        let count = usize::from(raw.bits[usize::from(len - 1)]);
        if count != 0 && len <= lookup_bits {
            let first_code = usize::try_from(canonical.min_code[usize::from(len)])
                .map_err(|_| canonical_huffman_error())?;
            let shift = usize::from(lookup_bits - len);
            let repeats = 1usize << shift;
            let mut offset = 0usize;
            while offset < count {
                let start = (first_code + offset) << shift;
                let end = start + repeats;
                // Compact canonical validation proves both the HUFFVAL index
                // and the complete prefix range are within these slices.
                fill_fast_range(output, start, end, (values[value_index + offset], len));
                offset += 1;
            }
        }
        value_index += count;
        len += 1;
    }
    debug_assert_eq!(value_index, values.len());
    Ok(())
}

const fn fill_u32_range(output: &mut [u32], start: usize, end: usize, value: u32) {
    if start >= end {
        return;
    }
    let (up_to_end, _) = output.split_at_mut(end);
    let (_, range) = up_to_end.split_at_mut(start);
    let mut index = 0;
    while index < range.len() {
        range[index] = value;
        index += 1;
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "a baseline DC symbol limits magnitude to 15 bits, which fits i16 and i32"
)]
const fn fill_dc_lookup(output: &mut [u32], lookup_bits: u8, code: u16, len: u8, symbol: u8) {
    if len > lookup_bits || symbol > 15 {
        return;
    }
    let total_len = len + symbol;
    if total_len > lookup_bits {
        return;
    }
    let code_start = (code as usize) << (lookup_bits - len);
    if symbol == 0 {
        fill_u32_range(
            output,
            code_start,
            code_start + (1usize << (lookup_bits - len)),
            pack_dc_value(total_len, 0),
        );
        return;
    }
    let repeats = 1usize << (lookup_bits - total_len);
    let mut magnitude = 0usize;
    while magnitude < (1usize << symbol) {
        let start = code_start + magnitude * repeats;
        let value = huff_extend(magnitude as i32, symbol) as i16;
        fill_u32_range(
            output,
            start,
            start + repeats,
            pack_dc_value(total_len, value),
        );
        magnitude += 1;
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "the AC size nibble limits magnitude to 15 bits, which fits i16 and i32"
)]
const fn fill_ac_lookup(output: &mut [u32], lookup_bits: u8, code: u16, len: u8, symbol: u8) {
    if len > lookup_bits {
        return;
    }
    let code_start = (code as usize) << (lookup_bits - len);
    let code_end = code_start + (1usize << (lookup_bits - len));
    let prefix = u32::from_le_bytes([len, symbol, 0, 0]);
    let run = symbol >> 4;
    let magnitude_bits = symbol & 0x0f;
    if magnitude_bits == 0 {
        let packed = if run == 0 {
            pack_ac_eob(len)
        } else if run == 15 {
            pack_ac_zrl(len)
        } else {
            prefix
        };
        fill_u32_range(output, code_start, code_end, packed);
        return;
    }
    let total_len = len + magnitude_bits;
    if total_len > lookup_bits {
        fill_u32_range(output, code_start, code_end, prefix);
        return;
    }
    let repeats = 1usize << (lookup_bits - total_len);
    let mut magnitude = 0usize;
    while magnitude < (1usize << magnitude_bits) {
        let start = code_start + magnitude * repeats;
        let value = huff_extend(magnitude as i32, magnitude_bits) as i16;
        fill_u32_range(
            output,
            start,
            start + repeats,
            pack_ac_value(total_len, run, value),
        );
        magnitude += 1;
    }
}

/// One const-capable range filler serves compile-time standard tables and the
/// fallible runtime storage used for custom DHT definitions.
#[expect(
    clippy::too_many_arguments,
    reason = "one const table builder fills the related static lookup layouts in place"
)]
const fn fill_lookup_tables(
    fast: &mut [(u8, u8)],
    fast_bits: u8,
    primary: &mut [u32],
    primary_bits: u8,
    primary8: &mut [u32],
    packed: &mut [u32],
    values: &[u8],
    canonical: &CanonicalHuffmanDerivation,
    is_ac: bool,
) {
    let mut index = 0usize;
    while index < canonical.huffsize_len {
        let len = canonical.huffsize[index];
        let code = canonical.huffcode[index];
        let symbol = values[index];
        if len <= fast_bits {
            let start = (code as usize) << (fast_bits - len);
            let end = start + (1usize << (fast_bits - len));
            fill_fast_range(fast, start, end, (symbol, len));
        }
        if is_ac && !packed.is_empty() {
            fill_ac_lookup(packed, FAST_BITS, code, len, symbol);
        }
        if !primary.is_empty() {
            if is_ac {
                fill_ac_lookup(primary, primary_bits, code, len, symbol);
            } else {
                fill_dc_lookup(primary, primary_bits, code, len, symbol);
            }
        }
        if !primary8.is_empty() {
            if is_ac {
                fill_ac_lookup(primary8, NARROW_PRIMARY_BITS, code, len, symbol);
            } else {
                fill_dc_lookup(primary8, NARROW_PRIMARY_BITS, code, len, symbol);
            }
        }
        index += 1;
    }
}

struct StaticDcLookups {
    fast: [(u8, u8); FAST_ENTRIES],
    primary: [u32; FULL_PRIMARY_ENTRIES],
    primary8: [u32; NARROW_PRIMARY_ENTRIES],
}

struct StaticAcLookups {
    fast: [(u8, u8); FAST_ENTRIES],
    primary: [u32; FULL_PRIMARY_ENTRIES],
    primary8: [u32; NARROW_PRIMARY_ENTRIES],
    packed: [u32; FAST_ENTRIES],
}

#[expect(
    clippy::large_stack_arrays,
    reason = "this const builder is evaluated into static standard-table storage"
)]
const fn standard_dc_lookups<const N: usize>(bits: &[u8; 16], values: &[u8; N]) -> StaticDcLookups {
    let Ok(canonical) = j2k_codec_math::jpeg::derive_canonical_huffman(bits, N) else {
        panic!("invalid built-in DC Huffman table");
    };
    let mut output = StaticDcLookups {
        fast: [(0, 0); FAST_ENTRIES],
        primary: [0; FULL_PRIMARY_ENTRIES],
        primary8: [0; NARROW_PRIMARY_ENTRIES],
    };
    fill_lookup_tables(
        &mut output.fast,
        FAST_BITS,
        &mut output.primary,
        FULL_PRIMARY_BITS,
        &mut output.primary8,
        &mut [],
        values,
        &canonical,
        false,
    );
    output
}

#[expect(
    clippy::large_stack_arrays,
    reason = "this const builder is evaluated into static standard-table storage"
)]
const fn standard_ac_lookups<const N: usize>(bits: &[u8; 16], values: &[u8; N]) -> StaticAcLookups {
    let Ok(canonical) = j2k_codec_math::jpeg::derive_canonical_huffman(bits, N) else {
        panic!("invalid built-in AC Huffman table");
    };
    let mut output = StaticAcLookups {
        fast: [(0, 0); FAST_ENTRIES],
        primary: [0; FULL_PRIMARY_ENTRIES],
        primary8: [0; NARROW_PRIMARY_ENTRIES],
        packed: [0; FAST_ENTRIES],
    };
    fill_lookup_tables(
        &mut output.fast,
        FAST_BITS,
        &mut output.primary,
        FULL_PRIMARY_BITS,
        &mut output.primary8,
        &mut output.packed,
        values,
        &canonical,
        true,
    );
    output
}

const fn standard_metadata<const N: usize>(
    bits: &[u8; 16],
    values: &[u8; N],
    role: HuffmanTableRole,
) -> HuffmanMetadata {
    let Ok(canonical) = j2k_codec_math::jpeg::derive_canonical_huffman(bits, N) else {
        panic!("invalid built-in Huffman table");
    };
    HuffmanMetadata {
        bits: *bits,
        min_code: canonical.min_code,
        max_code: canonical.max_code,
        val_offset: canonical.val_offset,
        values: HuffmanValues::from_array(values),
        role,
        raw_lookup_shift: 0,
    }
}

static LUMA_DC_LOOKUPS: StaticDcLookups =
    standard_dc_lookups(&STD_LUMA_DC_BITS, &STD_LUMA_DC_VALUES);
static CHROMA_DC_LOOKUPS: StaticDcLookups =
    standard_dc_lookups(&STD_CHROMA_DC_BITS, &STD_CHROMA_DC_VALUES);
static LUMA_AC_LOOKUPS: StaticAcLookups =
    standard_ac_lookups(&STD_LUMA_AC_BITS, &STD_LUMA_AC_VALUES);
static CHROMA_AC_LOOKUPS: StaticAcLookups =
    standard_ac_lookups(&STD_CHROMA_AC_BITS, &STD_CHROMA_AC_VALUES);

static LUMA_DC_COMPILED: StaticCompiledHuffmanTable = StaticCompiledHuffmanTable {
    fast: &LUMA_DC_LOOKUPS.fast,
    metadata: standard_metadata(&STD_LUMA_DC_BITS, &STD_LUMA_DC_VALUES, HuffmanTableRole::Dc),
    primary: &LUMA_DC_LOOKUPS.primary,
    primary8: &LUMA_DC_LOOKUPS.primary8,
    packed: None,
};
static CHROMA_DC_COMPILED: StaticCompiledHuffmanTable = StaticCompiledHuffmanTable {
    fast: &CHROMA_DC_LOOKUPS.fast,
    metadata: standard_metadata(
        &STD_CHROMA_DC_BITS,
        &STD_CHROMA_DC_VALUES,
        HuffmanTableRole::Dc,
    ),
    primary: &CHROMA_DC_LOOKUPS.primary,
    primary8: &CHROMA_DC_LOOKUPS.primary8,
    packed: None,
};
static LUMA_AC_COMPILED: StaticCompiledHuffmanTable = StaticCompiledHuffmanTable {
    fast: &LUMA_AC_LOOKUPS.fast,
    metadata: standard_metadata(&STD_LUMA_AC_BITS, &STD_LUMA_AC_VALUES, HuffmanTableRole::Ac),
    primary: &LUMA_AC_LOOKUPS.primary,
    primary8: &LUMA_AC_LOOKUPS.primary8,
    packed: Some(&LUMA_AC_LOOKUPS.packed),
};
static CHROMA_AC_COMPILED: StaticCompiledHuffmanTable = StaticCompiledHuffmanTable {
    fast: &CHROMA_AC_LOOKUPS.fast,
    metadata: standard_metadata(
        &STD_CHROMA_AC_BITS,
        &STD_CHROMA_AC_VALUES,
        HuffmanTableRole::Ac,
    ),
    primary: &CHROMA_AC_LOOKUPS.primary,
    primary8: &CHROMA_AC_LOOKUPS.primary8,
    packed: Some(&CHROMA_AC_LOOKUPS.packed),
};

fn standard_compiled_table(
    raw: &RawHuffmanTable,
    role: HuffmanTableRole,
) -> Option<&'static StaticCompiledHuffmanTable> {
    let values = raw.values.as_slice();
    match role {
        HuffmanTableRole::Dc if raw.bits == STD_LUMA_DC_BITS && values == STD_LUMA_DC_VALUES => {
            Some(&LUMA_DC_COMPILED)
        }
        HuffmanTableRole::Dc
            if raw.bits == STD_CHROMA_DC_BITS && values == STD_CHROMA_DC_VALUES =>
        {
            Some(&CHROMA_DC_COMPILED)
        }
        HuffmanTableRole::Ac if raw.bits == STD_LUMA_AC_BITS && values == STD_LUMA_AC_VALUES => {
            Some(&LUMA_AC_COMPILED)
        }
        HuffmanTableRole::Ac
            if raw.bits == STD_CHROMA_AC_BITS && values == STD_CHROMA_AC_VALUES =>
        {
            Some(&CHROMA_AC_COMPILED)
        }
        _ => None,
    }
}

fn dynamic_raw_lookup_bits(canonical: &CanonicalHuffmanDerivation) -> u8 {
    canonical.huffsize[..canonical.huffsize_len]
        .iter()
        .copied()
        .max()
        .unwrap_or(MIN_DYNAMIC_RAW_BITS)
        .clamp(MIN_DYNAMIC_RAW_BITS, FAST_BITS)
}

impl HuffmanTable {
    const COMPILED_OWNER_BYTES: usize = size_of::<CompiledHuffmanTable>() + 2 * size_of::<usize>();

    pub(crate) const fn maximum_retained_allocation_bytes() -> usize {
        Self::COMPILED_OWNER_BYTES
            + FAST_ENTRIES * size_of::<(u8, u8)>()
            + FULL_PRIMARY_ENTRIES * size_of::<u32>()
            + NARROW_PRIMARY_ENTRIES * size_of::<u32>()
            + FAST_ENTRIES * size_of::<u32>()
    }

    pub(crate) fn retained_allocation_bytes(&self) -> usize {
        match self {
            Self::Static(_) => 0,
            Self::Dynamic(compiled) => compiled.retained_allocation_bytes,
        }
    }

    #[cfg(any(test, feature = "bench-internals"))]
    pub(crate) fn from_raw(
        raw: &RawHuffmanTable,
        role: HuffmanTableRole,
    ) -> Result<Self, JpegError> {
        let mut live_bytes = 0;
        Self::from_raw_with_live_budget(
            raw,
            role,
            &mut live_bytes,
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )
    }

    #[cfg(any(test, feature = "bench-internals"))]
    pub(crate) fn from_raw_with_live_budget(
        raw: &RawHuffmanTable,
        role: HuffmanTableRole,
        live_bytes: &mut usize,
        cap: usize,
    ) -> Result<Self, JpegError> {
        Self::from_raw_with_requirement_and_live_budget(
            raw,
            role,
            HuffmanLookupRequirement::FusedFull,
            live_bytes,
            cap,
        )
    }

    pub(crate) fn from_raw_with_requirement_and_live_budget(
        raw: &RawHuffmanTable,
        role: HuffmanTableRole,
        requirement: HuffmanLookupRequirement,
        live_bytes: &mut usize,
        cap: usize,
    ) -> Result<Self, JpegError> {
        if let Some(compiled) = standard_compiled_table(raw, role) {
            return Ok(Self::Static(compiled));
        }

        if requirement == HuffmanLookupRequirement::Raw {
            return Self::from_raw_only_with_live_budget(raw, role, live_bytes, cap);
        }

        // Validation remains authoritative for every untrusted/custom DHT.
        let canonical = derive_canonical_huffman(raw)?;
        let initial_live_bytes = *live_bytes;
        let mut table_live_bytes = initial_live_bytes;
        charge_live_bytes(&mut table_live_bytes, Self::COMPILED_OWNER_BYTES, cap)?;
        let raw_lookup_bits = match (requirement, role) {
            (HuffmanLookupRequirement::Raw, _) => dynamic_raw_lookup_bits(&canonical),
            (HuffmanLookupRequirement::FusedNarrow, _) => NARROW_RAW_BITS,
            (HuffmanLookupRequirement::FusedFull, _) => FAST_BITS,
        };
        let raw_lookup_entries = 1usize << raw_lookup_bits;
        let mut fast = Vec::new();
        try_reserve_for_len_with_live_budget(
            &mut fast,
            raw_lookup_entries,
            &mut table_live_bytes,
            cap,
        )?;
        fast.resize(raw_lookup_entries, (0u8, 0u8));

        let mut packed = Vec::new();
        if requirement == HuffmanLookupRequirement::FusedFull && role == HuffmanTableRole::Ac {
            try_reserve_for_len_with_live_budget(
                &mut packed,
                FAST_ENTRIES,
                &mut table_live_bytes,
                cap,
            )?;
            packed.resize(FAST_ENTRIES, 0u32);
        }

        let mut primary = Vec::new();
        let primary_entries = match requirement {
            HuffmanLookupRequirement::Raw => 0,
            HuffmanLookupRequirement::FusedNarrow => NARROW_PRIMARY_ENTRIES,
            HuffmanLookupRequirement::FusedFull => FULL_PRIMARY_ENTRIES,
        };
        if primary_entries != 0 {
            try_reserve_for_len_with_live_budget(
                &mut primary,
                primary_entries,
                &mut table_live_bytes,
                cap,
            )?;
            primary.resize(primary_entries, 0u32);
        }
        let mut primary8 = Vec::new();
        if requirement == HuffmanLookupRequirement::FusedFull {
            try_reserve_for_len_with_live_budget(
                &mut primary8,
                NARROW_PRIMARY_ENTRIES,
                &mut table_live_bytes,
                cap,
            )?;
            primary8.resize(NARROW_PRIMARY_ENTRIES, 0u32);
        }
        fill_lookup_tables(
            &mut fast,
            raw_lookup_bits,
            &mut primary,
            match requirement {
                HuffmanLookupRequirement::FusedNarrow => NARROW_PRIMARY_BITS,
                _ => FULL_PRIMARY_BITS,
            },
            &mut primary8,
            &mut packed,
            raw.values.as_slice(),
            &canonical,
            role == HuffmanTableRole::Ac,
        );

        let retained_allocation_bytes = table_live_bytes.checked_sub(initial_live_bytes).ok_or(
            JpegError::InternalInvariant {
                reason: "compiled Huffman table allocation accounting underflowed",
            },
        )?;
        let compiled = CompiledHuffmanTable {
            fast,
            metadata: HuffmanMetadata {
                bits: raw.bits,
                min_code: canonical.min_code,
                max_code: canonical.max_code,
                val_offset: canonical.val_offset,
                values: raw.values.clone(),
                role,
                raw_lookup_shift: FAST_BITS - raw_lookup_bits,
            },
            primary,
            primary8,
            packed,
            retained_allocation_bytes,
        };
        *live_bytes = table_live_bytes;
        Ok(Self::Dynamic(Arc::new(compiled)))
    }

    #[inline(never)]
    fn from_raw_only_with_live_budget(
        raw: &RawHuffmanTable,
        role: HuffmanTableRole,
        live_bytes: &mut usize,
        cap: usize,
    ) -> Result<Self, JpegError> {
        // Complete canonical validation precedes every fallible allocation, as
        // it does for the full-table path above.
        let canonical = derive_compact_raw_canonical_huffman(raw)?;
        let initial_live_bytes = *live_bytes;
        let mut table_live_bytes = initial_live_bytes;
        charge_live_bytes(&mut table_live_bytes, Self::COMPILED_OWNER_BYTES, cap)?;
        let raw_lookup_bits = canonical.lookup_bits;
        let raw_lookup_entries = 1usize << raw_lookup_bits;
        let mut fast = Vec::new();
        try_reserve_for_len_with_live_budget(
            &mut fast,
            raw_lookup_entries,
            &mut table_live_bytes,
            cap,
        )?;
        fast.resize(raw_lookup_entries, (0u8, 0u8));
        fill_raw_lookup_from_bits(&mut fast, raw_lookup_bits, raw, &canonical)?;

        let retained_allocation_bytes = table_live_bytes.checked_sub(initial_live_bytes).ok_or(
            JpegError::InternalInvariant {
                reason: "compiled Huffman table allocation accounting underflowed",
            },
        )?;
        let compiled = CompiledHuffmanTable {
            fast,
            metadata: HuffmanMetadata {
                bits: raw.bits,
                min_code: canonical.min_code,
                max_code: canonical.max_code,
                val_offset: canonical.val_offset,
                values: raw.values.clone(),
                role,
                raw_lookup_shift: FAST_BITS - raw_lookup_bits,
            },
            primary: Vec::new(),
            primary8: Vec::new(),
            packed: Vec::new(),
            retained_allocation_bytes,
        };
        *live_bytes = table_live_bytes;
        Ok(Self::Dynamic(Arc::new(compiled)))
    }

    fn metadata(&self) -> &HuffmanMetadata {
        match self {
            Self::Static(compiled) => &compiled.metadata,
            Self::Dynamic(compiled) => &compiled.metadata,
        }
    }

    pub(crate) fn matches_raw_identity(
        &self,
        raw: &RawHuffmanTable,
        role: HuffmanTableRole,
    ) -> bool {
        let metadata = self.metadata();
        metadata.role == role
            && metadata.bits == raw.bits
            && metadata.values.as_slice() == raw.values.as_slice()
    }

    #[cfg(any(test, feature = "bench-internals"))]
    fn fast(&self) -> Result<&[(u8, u8); FAST_ENTRIES], JpegError> {
        match self {
            Self::Static(compiled) => Ok(compiled.fast),
            Self::Dynamic(compiled) => {
                compiled
                    .fast
                    .as_slice()
                    .first_chunk()
                    .ok_or(JpegError::InternalInvariant {
                        reason: "compiled Huffman raw lookup has the wrong length",
                    })
            }
        }
    }

    fn raw_fast(&self) -> RawSymbolLookup<'_> {
        let entries = match self {
            Self::Static(compiled) => compiled.fast.as_slice(),
            Self::Dynamic(compiled) => compiled.fast.as_slice(),
        };
        debug_assert_eq!(
            entries.len(),
            1usize << (FAST_BITS - self.metadata().raw_lookup_shift)
        );
        RawSymbolLookup {
            ptr: entries.as_ptr(),
            marker: core::marker::PhantomData,
        }
    }

    fn lookup_capability(&self) -> HuffmanLookupRequirement {
        match self {
            Self::Static(_) => HuffmanLookupRequirement::FusedFull,
            Self::Dynamic(compiled) => {
                if compiled.primary.len() == FULL_PRIMARY_ENTRIES
                    && compiled.primary8.len() == NARROW_PRIMARY_ENTRIES
                    && compiled.fast.len() == FAST_ENTRIES
                    && (compiled.metadata.role == HuffmanTableRole::Dc
                        || compiled.packed.len() == FAST_ENTRIES)
                {
                    HuffmanLookupRequirement::FusedFull
                } else if compiled.primary.len() == NARROW_PRIMARY_ENTRIES
                    && compiled.primary8.is_empty()
                    && compiled.fast.len() == NARROW_RAW_ENTRIES
                    && compiled.packed.is_empty()
                {
                    HuffmanLookupRequirement::FusedNarrow
                } else {
                    HuffmanLookupRequirement::Raw
                }
            }
        }
    }

    pub(crate) fn satisfies_lookup_requirement(
        &self,
        requirement: HuffmanLookupRequirement,
    ) -> bool {
        requirement == HuffmanLookupRequirement::Raw || self.lookup_capability() >= requirement
    }

    pub(crate) fn supports_fused_lookup(&self) -> bool {
        self.satisfies_lookup_requirement(HuffmanLookupRequirement::FusedNarrow)
    }

    fn primary<const N: usize>(&self) -> Result<&[u32; N], JpegError> {
        let entries = match self {
            Self::Static(compiled) if N == NARROW_PRIMARY_ENTRIES => compiled.primary8.as_slice(),
            Self::Static(compiled) if N == FULL_PRIMARY_ENTRIES => compiled.primary.as_slice(),
            Self::Dynamic(compiled)
                if N == NARROW_PRIMARY_ENTRIES
                    && compiled.primary.len() == FULL_PRIMARY_ENTRIES =>
            {
                compiled.primary8.as_slice()
            }
            Self::Dynamic(compiled) if N == NARROW_PRIMARY_ENTRIES => compiled.primary.as_slice(),
            Self::Dynamic(compiled) if N == FULL_PRIMARY_ENTRIES => compiled.primary.as_slice(),
            Self::Static(_) | Self::Dynamic(_) => &[],
        };
        entries.first_chunk().ok_or(JpegError::InternalInvariant {
            reason: "compiled Huffman primary lookup has the wrong length",
        })
    }

    fn packed(&self) -> Result<(&[u32; FAST_ENTRIES], bool), JpegError> {
        match self {
            Self::Static(compiled) => {
                compiled
                    .packed
                    .map(|packed| (packed, true))
                    .ok_or(JpegError::InternalInvariant {
                        reason: "prepared DC Huffman table has no AC lookup",
                    })
            }
            Self::Dynamic(compiled) if compiled.packed.is_empty() => {
                Ok((&EMPTY_PACKED_LOOKUP, false))
            }
            Self::Dynamic(compiled) => compiled
                .packed
                .as_slice()
                .first_chunk()
                .map(|packed| (packed, true))
                .ok_or(JpegError::InternalInvariant {
                    reason: "compiled Huffman AC lookup has the wrong length",
                }),
        }
    }

    #[cfg(test)]
    fn packed_lookup(&self) -> &[u32] {
        match self {
            Self::Static(compiled) => compiled.packed.map_or(&[], |packed| packed.as_slice()),
            Self::Dynamic(compiled) => &compiled.packed,
        }
    }

    pub(crate) fn dc_with_entries<const N: usize>(
        &self,
    ) -> Result<DcHuffmanTable<'_, N>, JpegError> {
        if self.metadata().role != HuffmanTableRole::Dc {
            return Err(JpegError::InternalInvariant {
                reason: "prepared AC Huffman table was resolved for a DC decode",
            });
        }
        if !self.supports_fused_lookup() {
            return Err(JpegError::InternalInvariant {
                reason: "raw-only Huffman table was resolved for a fused DC decode",
            });
        }
        Ok(DcHuffmanTable {
            fast: self.raw_fast(),
            metadata: self.metadata(),
            primary: self.primary::<N>()?,
        })
    }

    #[cfg(test)]
    pub(crate) fn dc(&self) -> Result<DcHuffmanTable<'_>, JpegError> {
        self.dc_with_entries::<FULL_PRIMARY_ENTRIES>()
    }

    pub(crate) fn ac_with_entries<const N: usize>(
        &self,
    ) -> Result<AcHuffmanTable<'_, N>, JpegError> {
        if self.metadata().role != HuffmanTableRole::Ac {
            return Err(JpegError::InternalInvariant {
                reason: "prepared DC Huffman table was resolved for an AC decode",
            });
        }
        if !self.supports_fused_lookup() {
            return Err(JpegError::InternalInvariant {
                reason: "raw-only Huffman table was resolved for a fused AC decode",
            });
        }
        let (packed, wide_available) = self.packed()?;
        Ok(AcHuffmanTable {
            fast: self.raw_fast(),
            metadata: self.metadata(),
            primary: self.primary::<N>()?,
            packed,
            wide: false,
            wide_available,
        })
    }

    #[cfg(test)]
    pub(crate) fn ac(&self) -> Result<AcHuffmanTable<'_>, JpegError> {
        self.ac_with_entries::<FULL_PRIMARY_ENTRIES>()
    }

    pub(crate) fn raw(
        &self,
        expected_role: HuffmanTableRole,
    ) -> Result<RawHuffmanDecoder<'_>, JpegError> {
        if self.metadata().role != expected_role {
            return Err(JpegError::InternalInvariant {
                reason: "prepared Huffman table has the wrong role for a raw decode",
            });
        }
        Ok(RawHuffmanDecoder {
            fast: self.raw_fast(),
            metadata: self.metadata(),
        })
    }

    #[cfg(any(test, feature = "bench-internals"))]
    pub(crate) fn decode(&self, br: &mut BitReader<'_>) -> Result<u8, JpegError> {
        decode_symbol(self.fast()?, self.metadata(), br)
    }
}

#[cfg(any(test, feature = "bench-internals"))]
#[expect(
    clippy::inline_always,
    reason = "measured Huffman lookup hot path requires cross-helper inlining"
)]
#[inline(always)]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "the Huffman slow path converts only validated 16-bit codes and non-negative table offsets"
)]
fn decode_symbol(
    fast: &[(u8, u8); FAST_ENTRIES],
    metadata: &HuffmanMetadata,
    br: &mut BitReader<'_>,
) -> Result<u8, JpegError> {
    br.ensure_bits_padded(FAST_BITS)?;
    let peek = br.peek_bits(FAST_BITS) as usize;
    let (symbol, length) = fast[peek];
    if length != 0 {
        br.consume_bits(length);
        return Ok(symbol);
    }
    br.ensure_bits_padded(16)?;
    let code16 = br.peek_bits(16) as i32;
    for len in (FAST_BITS as usize + 1)..=16 {
        let length = len as u8;
        let code = code16 >> (16 - length);
        if code <= metadata.max_code[len] {
            br.consume_bits(length);
            let index = (code + metadata.val_offset[len]) as usize;
            return metadata.values.get(index).ok_or(JpegError::HuffmanDecode {
                mcu: 0,
                reason: HuffmanFailure::InvalidSymbol,
            });
        }
    }
    Err(JpegError::HuffmanDecode {
        mcu: 0,
        reason: HuffmanFailure::CodeOverflow,
    })
}

#[expect(
    clippy::inline_always,
    reason = "progressive raw Huffman lookup is in every symbol loop"
)]
#[inline(always)]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "the Huffman slow path converts only validated 16-bit codes and non-negative table offsets"
)]
// `N == 0` denotes a raw-only handle. Successful Full typed construction
// guarantees a 15-bit raw lookup with shift zero, while N256 may use either a
// Narrow table or a Full table's companion and therefore retains its metadata width.
fn decode_raw_symbol<const N: usize>(
    fast: RawSymbolLookup<'_>,
    metadata: &HuffmanMetadata,
    br: &mut BitReader<'_>,
) -> Result<u8, JpegError> {
    br.ensure_bits_padded(FAST_BITS)?;
    let peek15 = br.peek_bits(FAST_BITS) as i32;
    let raw_lookup_shift = if N == FULL_PRIMARY_ENTRIES {
        debug_assert_eq!(metadata.raw_lookup_shift, 0);
        0
    } else {
        metadata.raw_lookup_shift
    };
    let peek = (peek15 >> raw_lookup_shift) as usize;
    // SAFETY: static tables have fixed 15-bit arrays; dynamic construction
    // allocates exactly `1 << (15 - shift)` entries from this same metadata.
    // A shifted 15-bit peek is therefore in bounds.
    let (symbol, length) = unsafe { fast.get_unchecked(peek) };
    if length != 0 {
        br.consume_bits(length);
        return Ok(symbol);
    }

    let lookup_bits = FAST_BITS - raw_lookup_shift;
    for len in (lookup_bits as usize + 1)..=FAST_BITS as usize {
        let length = len as u8;
        let code = peek15 >> (FAST_BITS - length);
        if code <= metadata.max_code[len] {
            br.consume_bits(length);
            let index = (code + metadata.val_offset[len]) as usize;
            return metadata.values.get(index).ok_or(JpegError::HuffmanDecode {
                mcu: 0,
                reason: HuffmanFailure::InvalidSymbol,
            });
        }
    }

    br.ensure_bits_padded(16)?;
    let code16 = br.peek_bits(16) as i32;
    if code16 <= metadata.max_code[16] {
        br.consume_bits(16);
        let index = (code16 + metadata.val_offset[16]) as usize;
        return metadata.values.get(index).ok_or(JpegError::HuffmanDecode {
            mcu: 0,
            reason: HuffmanFailure::InvalidSymbol,
        });
    }
    Err(JpegError::HuffmanDecode {
        mcu: 0,
        reason: HuffmanFailure::CodeOverflow,
    })
}

/// Resolve the symbol that the old primary12 table would have covered without
/// consuming bits. Narrow handles call this only after a primary8 miss; the
/// reader has already performed the same 12-bit padded lookahead as the Full
/// path. Primary8 AC huffman-only entries carry their symbol directly.
#[expect(
    clippy::inline_always,
    reason = "primary8 misses remain inside the measured Huffman hot path"
)]
#[inline(always)]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "the lookup examines only 12-bit canonical codes and validated non-negative offsets"
)]
fn symbol_from_primary12_peek(
    metadata: &HuffmanMetadata,
    peek12: usize,
) -> Result<Option<(u8, u8)>, JpegError> {
    for len in 1..=FULL_PRIMARY_BITS as usize {
        let length = len as u8;
        let code = (peek12 >> (FULL_PRIMARY_BITS - length)) as i32;
        if code >= metadata.min_code[len] && code <= metadata.max_code[len] {
            let index = (code + metadata.val_offset[len]) as usize;
            return metadata
                .values
                .get(index)
                .map(|symbol| Some((symbol, length)))
                .ok_or(JpegError::HuffmanDecode {
                    mcu: 0,
                    reason: HuffmanFailure::InvalidSymbol,
                });
        }
    }
    Ok(None)
}

#[expect(
    clippy::inline_always,
    reason = "the const entry count folds primary indexing into each symbol decoder"
)]
#[inline(always)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "supported primary tables contain 256 or 4096 entries, so their bit counts fit u8"
)]
fn primary_index<const N: usize>(peek12: usize) -> usize {
    debug_assert!(N == NARROW_PRIMARY_ENTRIES || N == FULL_PRIMARY_ENTRIES);
    let primary_bits = N.trailing_zeros() as u8;
    peek12 >> (FULL_PRIMARY_BITS - primary_bits)
}

impl<const N: usize> DcHuffmanTable<'_, N> {
    #[expect(
        clippy::inline_always,
        reason = "measured Huffman lookup hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    #[expect(
        clippy::cast_possible_wrap,
        reason = "packed DC values intentionally reinterpret a validated 16-bit two's-complement field"
    )]
    pub(crate) fn decode_fast_dc(self, br: &mut BitReader<'_>) -> Result<i32, JpegError> {
        br.ensure_bits_padded(FULL_PRIMARY_BITS)?;
        let peek12 = br.peek_bits(FULL_PRIMARY_BITS) as usize;
        let packed = self.primary[primary_index::<N>(peek12)];
        if packed != 0 {
            br.consume_bits((packed & DC_FAST_LEN_MASK) as u8);
            return Ok(i32::from(
                ((packed >> DC_FAST_VALUE_SHIFT) & 0xFFFF) as u16 as i16,
            ));
        }
        if N == NARROW_PRIMARY_ENTRIES {
            if let Some((symbol, length)) = symbol_from_primary12_peek(self.metadata, peek12)? {
                if symbol <= 15 && length + symbol <= FULL_PRIMARY_BITS {
                    br.consume_bits(length);
                    return br.receive_extend(symbol);
                }
            }
        }
        let symbol = decode_raw_symbol::<N>(self.fast, self.metadata, br)?;
        if symbol > 15 {
            return Err(JpegError::HuffmanDecode {
                mcu: 0,
                reason: HuffmanFailure::InvalidSymbol,
            });
        }
        br.receive_extend(symbol)
    }

    #[expect(
        clippy::inline_always,
        reason = "measured Huffman lookup hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "lossless differences are defined modulo 2^16, so wrapping into u16 is exact"
    )]
    pub(crate) fn decode_lossless_diff(self, br: &mut BitReader<'_>) -> Result<u16, JpegError> {
        br.ensure_bits_padded(FULL_PRIMARY_BITS)?;
        let peek12 = br.peek_bits(FULL_PRIMARY_BITS) as usize;
        let packed = self.primary[primary_index::<N>(peek12)];
        if packed != 0 {
            br.consume_bits((packed & DC_FAST_LEN_MASK) as u8);
            return Ok((packed >> DC_FAST_VALUE_SHIFT) as u16);
        }
        if N == NARROW_PRIMARY_ENTRIES {
            if let Some((symbol, length)) = symbol_from_primary12_peek(self.metadata, peek12)? {
                if symbol <= 15 && length + symbol <= FULL_PRIMARY_BITS {
                    br.consume_bits(length);
                    return Ok(br.receive_extend(symbol)? as u16);
                }
            }
        }
        match decode_raw_symbol::<N>(self.fast, self.metadata, br)? {
            16 => Ok(0x8000),
            ssss @ 0..=15 => Ok(br.receive_extend(ssss)? as u16),
            _ => Err(JpegError::HuffmanDecode {
                mcu: 0,
                reason: HuffmanFailure::InvalidSymbol,
            }),
        }
    }
}

impl RawHuffmanDecoder<'_> {
    #[expect(
        clippy::inline_always,
        reason = "progressive scans decode every Huffman symbol through this handle"
    )]
    #[inline(always)]
    pub(crate) fn decode(self, br: &mut BitReader<'_>) -> Result<u8, JpegError> {
        decode_raw_symbol::<0>(self.fast, self.metadata, br)
    }
}

impl<const N: usize> AcHuffmanTable<'_, N> {
    pub(crate) fn with_wide_lookup(mut self, wide: bool) -> Self {
        self.wide = wide && self.wide_available;
        self
    }

    #[inline]
    pub(crate) fn uses_wide_lookup(self) -> bool {
        self.wide
    }

    #[expect(
        clippy::inline_always,
        reason = "measured Huffman lookup hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "JPEG receive-extend values are bounded to the signed 16-bit packed AC field"
    )]
    pub(crate) fn decode_fast_ac<const WIDE: bool>(
        self,
        br: &mut BitReader<'_>,
    ) -> Result<u32, JpegError> {
        let bits = if WIDE { FAST_BITS } else { FULL_PRIMARY_BITS };
        br.ensure_bits_padded(bits)?;
        let peek = br.peek_bits(bits) as usize;
        let packed = if WIDE {
            self.packed[peek]
        } else {
            self.primary[primary_index::<N>(peek)]
        };
        if packed & AC_FAST_KIND_MASK != 0 {
            br.consume_bits((packed & AC_FAST_LEN_MASK) as u8);
            return Ok(packed);
        }
        let [length, symbol, _, _] = packed.to_le_bytes();
        let (symbol, primary_length) = if length != 0 {
            br.consume_bits(length);
            (symbol, length)
        } else if !WIDE && N == NARROW_PRIMARY_ENTRIES {
            if let Some((symbol, length)) = symbol_from_primary12_peek(self.metadata, peek)? {
                br.consume_bits(length);
                (symbol, length)
            } else {
                (decode_raw_symbol::<N>(self.fast, self.metadata, br)?, 0)
            }
        } else {
            (decode_raw_symbol::<N>(self.fast, self.metadata, br)?, 0)
        };
        let run = symbol >> 4;
        let magnitude_bits = symbol & 0x0f;
        if magnitude_bits == 0 {
            if !WIDE && N == NARROW_PRIMARY_ENTRIES {
                return Ok(match run {
                    0 => pack_ac_eob(primary_length),
                    15 => pack_ac_zrl(primary_length),
                    _ => pack_ac_eob(0),
                });
            }
            return Ok(if run == 15 {
                pack_ac_zrl(0)
            } else {
                pack_ac_eob(0)
            });
        }
        if !WIDE && N == NARROW_PRIMARY_ENTRIES {
            let primary_total_len = primary_length + magnitude_bits;
            if primary_length != 0 && primary_total_len <= FULL_PRIMARY_BITS {
                let value = br.receive_extend(magnitude_bits)?;
                return Ok(pack_ac_value(primary_total_len, run, value as i16));
            }
        }
        let value = br.receive_extend(magnitude_bits)?;
        Ok(pack_ac_value(0, run, value as i16))
    }

    #[expect(
        clippy::inline_always,
        reason = "measured Huffman lookup hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "JPEG receive-extend values are bounded to the signed 16-bit packed AC field"
    )]
    pub(crate) fn skip_fast_ac<const WIDE: bool>(
        self,
        br: &mut BitReader<'_>,
    ) -> Result<u32, JpegError> {
        let bits = if WIDE { FAST_BITS } else { FULL_PRIMARY_BITS };
        br.ensure_bits_padded(bits)?;
        let peek = br.peek_bits(bits) as usize;
        let packed = if WIDE {
            self.packed[peek]
        } else {
            self.primary[primary_index::<N>(peek)]
        };
        if packed & AC_FAST_KIND_MASK != 0 {
            br.consume_bits((packed & AC_FAST_LEN_MASK) as u8);
            return Ok(packed);
        }
        let [length, symbol, _, _] = packed.to_le_bytes();
        let (symbol, primary_length) = if length != 0 {
            br.consume_bits(length);
            (symbol, length)
        } else if !WIDE && N == NARROW_PRIMARY_ENTRIES {
            if let Some((symbol, length)) = symbol_from_primary12_peek(self.metadata, peek)? {
                br.consume_bits(length);
                (symbol, length)
            } else {
                (decode_raw_symbol::<N>(self.fast, self.metadata, br)?, 0)
            }
        } else {
            (decode_raw_symbol::<N>(self.fast, self.metadata, br)?, 0)
        };
        let run = symbol >> 4;
        let magnitude_bits = symbol & 0x0f;
        if magnitude_bits == 0 {
            if !WIDE && N == NARROW_PRIMARY_ENTRIES {
                return Ok(match run {
                    0 => pack_ac_eob(primary_length),
                    15 => pack_ac_zrl(primary_length),
                    _ => pack_ac_eob(0),
                });
            }
            return Ok(if run == 15 {
                pack_ac_zrl(0)
            } else {
                pack_ac_eob(0)
            });
        }
        if !WIDE && N == NARROW_PRIMARY_ENTRIES {
            let primary_total_len = primary_length + magnitude_bits;
            if primary_length != 0 && primary_total_len <= FULL_PRIMARY_BITS {
                let value = br.receive_extend(magnitude_bits)?;
                return Ok(pack_ac_value(primary_total_len, run, value as i16));
            }
        }
        br.ensure_bits(magnitude_bits)?;
        br.consume_bits(magnitude_bits);
        Ok(pack_ac_value(0, run, 0))
    }
}

impl PreparedHuffmanTableId {
    fn for_next_index(index: usize) -> Result<Self, JpegError> {
        let one_based = index
            .checked_add(1)
            .and_then(|value| u32::try_from(value).ok())
            .and_then(NonZeroU32::new)
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })?;
        Ok(Self(one_based))
    }

    fn index(self) -> Option<usize> {
        usize::try_from(self.0.get())
            .ok()
            .and_then(|value| value.checked_sub(1))
    }
}

impl PreparedHuffmanTables {
    const OWNER_BYTES: usize =
        core::mem::size_of::<Vec<HuffmanTable>>() + 2 * core::mem::size_of::<usize>();

    pub(crate) fn allocation_bytes_for_capacity(capacity: usize) -> Result<usize, JpegError> {
        crate::allocation::checked_add_allocation_bytes(
            Self::OWNER_BYTES,
            crate::allocation::checked_allocation_bytes::<HuffmanTable>(capacity)?,
        )
    }

    /// Conservative preflight when table roles are not available yet. Actual
    /// retained accounting uses each role-specific shared payload instead.
    pub(crate) fn maximum_allocation_bytes_for_capacity(
        capacity: usize,
    ) -> Result<usize, JpegError> {
        let payload_bytes = capacity
            .checked_mul(HuffmanTable::maximum_retained_allocation_bytes())
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })?;
        checked_add_allocation_bytes(
            Self::allocation_bytes_for_capacity(capacity)?,
            payload_bytes,
        )
    }

    #[cfg(test)]
    pub(crate) fn try_with_capacity(capacity: usize) -> Result<Self, JpegError> {
        let mut live_bytes = 0;
        Self::try_with_capacity_and_live_budget(
            capacity,
            &mut live_bytes,
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )
    }

    pub(crate) fn try_with_capacity_and_live_budget(
        capacity: usize,
        live_bytes: &mut usize,
        cap: usize,
    ) -> Result<Self, JpegError> {
        let requested = live_bytes.saturating_add(Self::OWNER_BYTES);
        if requested > cap {
            return Err(JpegError::MemoryCapExceeded { requested, cap });
        }
        *live_bytes = requested;
        let mut entries = Vec::new();
        crate::allocation::try_reserve_for_len_with_live_budget(
            &mut entries,
            capacity,
            live_bytes,
            cap,
        )?;
        Ok(Self {
            entries: Arc::new(entries),
            retained_table_bytes: 0,
        })
    }

    pub(crate) fn push(
        &mut self,
        table: HuffmanTable,
    ) -> Result<PreparedHuffmanTableId, JpegError> {
        let id = PreparedHuffmanTableId::for_next_index(self.entries.len())?;
        if self.entries.len() == self.entries.capacity() {
            return Err(JpegError::InternalInvariant {
                reason: "prepared Huffman arena exceeded its reserved capacity",
            });
        }
        let retained_table_bytes = checked_add_allocation_bytes(
            self.retained_table_bytes,
            table.retained_allocation_bytes(),
        )?;
        Arc::get_mut(&mut self.entries)
            .ok_or(JpegError::InternalInvariant {
                reason: "prepared Huffman arena was shared before construction finished",
            })?
            .push(table);
        self.retained_table_bytes = retained_table_bytes;
        Ok(id)
    }

    pub(crate) fn push_with_live_budget(
        &mut self,
        table: HuffmanTable,
        live_bytes: &mut usize,
        cap: usize,
    ) -> Result<PreparedHuffmanTableId, JpegError> {
        let mut projected_live_bytes = *live_bytes;
        charge_live_bytes(
            &mut projected_live_bytes,
            table.retained_allocation_bytes(),
            cap,
        )?;
        let id = self.push(table)?;
        *live_bytes = projected_live_bytes;
        Ok(id)
    }

    pub(crate) fn get(&self, id: PreparedHuffmanTableId) -> Option<&HuffmanTable> {
        id.index().and_then(|index| self.entries.get(index))
    }

    pub(crate) fn all_satisfy_lookup_requirement(
        &self,
        requirement: HuffmanLookupRequirement,
    ) -> bool {
        self.entries
            .iter()
            .all(|table| table.satisfies_lookup_requirement(requirement))
    }

    #[cfg(test)]
    pub(crate) fn get_dc(
        &self,
        id: PreparedHuffmanTableId,
    ) -> Result<DcHuffmanTable<'_>, JpegError> {
        self.get_dc_with_entries::<FULL_PRIMARY_ENTRIES>(id)
    }

    pub(crate) fn get_dc_with_entries<const N: usize>(
        &self,
        id: PreparedHuffmanTableId,
    ) -> Result<DcHuffmanTable<'_, N>, JpegError> {
        self.get(id)
            .ok_or(JpegError::InternalInvariant {
                reason: "prepared component references a missing Huffman table",
            })?
            .dc_with_entries::<N>()
    }

    #[cfg(test)]
    pub(crate) fn get_ac(
        &self,
        id: PreparedHuffmanTableId,
    ) -> Result<AcHuffmanTable<'_>, JpegError> {
        self.get_ac_with_entries::<FULL_PRIMARY_ENTRIES>(id)
    }

    pub(crate) fn get_ac_with_entries<const N: usize>(
        &self,
        id: PreparedHuffmanTableId,
    ) -> Result<AcHuffmanTable<'_, N>, JpegError> {
        self.get(id)
            .ok_or(JpegError::InternalInvariant {
                reason: "prepared component references a missing Huffman table",
            })?
            .ac_with_entries::<N>()
    }

    pub(crate) fn get_raw(
        &self,
        id: PreparedHuffmanTableId,
        role: HuffmanTableRole,
    ) -> Result<RawHuffmanDecoder<'_>, JpegError> {
        self.get(id)
            .ok_or(JpegError::InternalInvariant {
                reason: "prepared component references a missing Huffman table",
            })?
            .raw(role)
    }

    pub(crate) fn id_at(&self, index: usize) -> Option<PreparedHuffmanTableId> {
        if index >= self.entries.len() {
            return None;
        }
        PreparedHuffmanTableId::for_next_index(index).ok()
    }

    pub(crate) fn retained_allocation_bytes(&self) -> Result<usize, JpegError> {
        checked_add_allocation_bytes(
            Self::allocation_bytes_for_capacity(self.entries.capacity())?,
            self.retained_table_bytes,
        )
    }

    pub(crate) fn try_clone_with_live_budget(
        &self,
        live_bytes: &mut usize,
        cap: usize,
    ) -> Result<Self, JpegError> {
        // Charge the full retained arena to each owner conservatively, even
        // though cloning only increments the reference count.
        let requested = live_bytes.saturating_add(self.retained_allocation_bytes()?);
        if requested > cap {
            return Err(JpegError::MemoryCapExceeded { requested, cap });
        }
        *live_bytes = requested;
        Ok(Self {
            entries: Arc::clone(&self.entries),
            retained_table_bytes: self.retained_table_bytes,
        })
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn capacity(&self) -> usize {
        self.entries.capacity()
    }
}

#[expect(
    clippy::inline_always,
    reason = "measured Huffman lookup hot path requires cross-helper inlining"
)]
#[inline(always)]
pub(crate) fn ac_decoded_run(packed: u32) -> usize {
    ((packed & AC_FAST_RUN_MASK) >> 4) as usize
}

#[expect(
    clippy::inline_always,
    reason = "measured Huffman lookup hot path requires cross-helper inlining"
)]
#[inline(always)]
#[expect(
    clippy::cast_possible_wrap,
    reason = "packed AC values intentionally reinterpret a 16-bit two's-complement field"
)]
pub(crate) fn ac_decoded_value(packed: u32) -> i32 {
    i32::from(((packed >> AC_FAST_VALUE_SHIFT) & 0xFFFF) as u16 as i16)
}

#[inline]
#[expect(
    clippy::cast_sign_loss,
    reason = "signed AC coefficients are intentionally stored as a 16-bit two's-complement bit field"
)]
const fn pack_ac_value(total_len: u8, run: u8, value: i16) -> u32 {
    AC_FAST_VALUE
        | (((value as u16) as u32) << AC_FAST_VALUE_SHIFT)
        | ((run as u32) << 4)
        | total_len as u32
}

#[inline]
const fn pack_ac_eob(total_len: u8) -> u32 {
    AC_FAST_EOB | total_len as u32
}

#[inline]
const fn pack_ac_zrl(total_len: u8) -> u32 {
    AC_FAST_ZRL | (15 << 4) | total_len as u32
}

#[inline]
#[expect(
    clippy::cast_sign_loss,
    reason = "signed DC coefficients are intentionally stored as a 16-bit two's-complement bit field"
)]
const fn pack_dc_value(total_len: u8, value: i16) -> u32 {
    (((value as u16) as u32) << DC_FAST_VALUE_SHIFT) | total_len as u32
}

const fn huff_extend(v: i32, ssss: u8) -> i32 {
    let threshold = 1i32 << (ssss - 1);
    if v < threshold {
        v + ((-1i32) << ssss) + 1
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct LegacyLookups {
        fast: Vec<(u8, u8)>,
        min_code: [i32; 17],
        max_code: [i32; 17],
        val_offset: [i32; 17],
        fast_dc: Vec<u32>,
        fast_ac: Vec<u32>,
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the legacy oracle reproduces conversions bounded by the fast lookup and JPEG symbol fields"
    )]
    fn legacy_lookups(raw: &RawHuffmanTable, lookup_bits: u8) -> Result<LegacyLookups, JpegError> {
        let canonical = derive_canonical_huffman(raw)?;
        let mut fast = alloc::vec![(0u8, 0u8); 1usize << lookup_bits];
        let mut fast_dc = alloc::vec![0u32; 1usize << lookup_bits];
        let mut fast_ac = alloc::vec![0u32; 1usize << lookup_bits];
        let mut k = 0;
        for len_minus_1 in 0..lookup_bits as usize {
            let len = (len_minus_1 + 1) as u8;
            for _ in 0..raw.bits[len_minus_1] {
                let code = canonical.huffcode[k];
                let base = usize::from(code) << (lookup_bits - len);
                let count = 1 << (lookup_bits - len);
                fast[base..base + count].fill((raw.values.as_slice()[k], len));
                k += 1;
            }
        }
        for (idx, &(sym, len)) in fast.iter().enumerate() {
            if len == 0 {
                continue;
            }
            if sym <= 15 {
                let total_len = len + sym;
                if total_len <= lookup_bits {
                    let diff = if sym == 0 {
                        0
                    } else {
                        let mag_shift = lookup_bits - total_len;
                        let mag_mask = (1u16 << sym) - 1;
                        let mag_bits = ((idx as u16) >> mag_shift) & mag_mask;
                        huff_extend(i32::from(mag_bits), sym)
                    };
                    if (i32::from(i16::MIN)..=i32::from(i16::MAX)).contains(&diff) {
                        fast_dc[idx] = pack_dc_value(total_len, diff as i16);
                    }
                }
            }
            let run = usize::from((sym >> 4) & 0x0F);
            let ssss = sym & 0x0F;
            if ssss == 0 {
                fast_ac[idx] = match run {
                    0 => pack_ac_eob(len),
                    15 => pack_ac_zrl(len),
                    _ => 0,
                };
                continue;
            }
            let total_len = len + ssss;
            if total_len > lookup_bits {
                continue;
            }
            let mag_shift = lookup_bits - total_len;
            let mag_mask = (1u16 << ssss) - 1;
            let mag_bits = ((idx as u16) >> mag_shift) & mag_mask;
            let value = huff_extend(i32::from(mag_bits), ssss);
            if (i32::from(i16::MIN)..=i32::from(i16::MAX)).contains(&value) {
                fast_ac[idx] = pack_ac_value(total_len, run as u8, value as i16);
            }
        }
        Ok(LegacyLookups {
            fast,
            min_code: canonical.min_code,
            max_code: canonical.max_code,
            val_offset: canonical.val_offset,
            fast_dc,
            fast_ac,
        })
    }

    #[test]
    fn compiled_table_retains_only_one_packed_specialization() {
        let raw = nonstandard_dc_shape_raw();
        let dc = HuffmanTable::from_raw(&raw, HuffmanTableRole::Dc).unwrap();
        let ac = HuffmanTable::from_raw(&raw, HuffmanTableRole::Ac).unwrap();
        assert_eq!(dc.packed_lookup(), [] as [u32; 0]);
        assert_eq!(ac.packed_lookup().len(), FAST_ENTRIES);
        assert_eq!(
            ac.retained_allocation_bytes() - dc.retained_allocation_bytes(),
            FAST_ENTRIES * core::mem::size_of::<u32>()
        );
        assert!(core::mem::size_of::<HuffmanTable>() <= 2 * core::mem::size_of::<usize>());
    }

    #[test]
    fn raw_only_table_omits_fused_payload_and_rejects_fused_handles() {
        let raw = nonstandard_dc_shape_raw();
        for role in [HuffmanTableRole::Dc, HuffmanTableRole::Ac] {
            let mut raw_live_bytes = 0;
            let raw_only = HuffmanTable::from_raw_with_requirement_and_live_budget(
                &raw,
                role,
                HuffmanLookupRequirement::Raw,
                &mut raw_live_bytes,
                j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            )
            .unwrap();
            let fused = HuffmanTable::from_raw(&raw, role).unwrap();

            assert!(!raw_only.supports_fused_lookup());
            assert!(raw_only.raw(role).is_ok());
            assert!(raw_only.retained_allocation_bytes() < fused.retained_allocation_bytes());
            match role {
                HuffmanTableRole::Dc => assert!(raw_only.dc().is_err()),
                HuffmanTableRole::Ac => assert!(raw_only.ac().is_err()),
            }
        }
    }

    #[test]
    fn raw_only_short_lookups_match_full_decode_and_error_state() {
        for code_length in 0usize..=7 {
            let mut bits = [0; 16];
            let values = if code_length == 0 {
                HuffmanValues::default()
            } else {
                bits[code_length - 1] = 1;
                HuffmanValues::from_slice(&[7])
            };
            let raw = RawHuffmanTable { bits, values };
            let mut live_bytes = 0;
            let raw_only = HuffmanTable::from_raw_with_requirement_and_live_budget(
                &raw,
                HuffmanTableRole::Dc,
                HuffmanLookupRequirement::Raw,
                &mut live_bytes,
                j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            )
            .unwrap();
            let full = HuffmanTable::from_raw(&raw, HuffmanTableRole::Dc).unwrap();
            let HuffmanTable::Dynamic(compiled) = &raw_only else {
                panic!("nonstandard short table must use dynamic storage");
            };
            assert_eq!(compiled.fast.len(), 1usize << code_length.max(1));

            if code_length != 0 {
                let mut compact_reader = BitReader::new(&[0; 3]);
                let mut full_reader = BitReader::new(&[0; 3]);
                let compact_result = raw_only
                    .raw(HuffmanTableRole::Dc)
                    .unwrap()
                    .decode(&mut compact_reader);
                let full_result = full
                    .raw(HuffmanTableRole::Dc)
                    .unwrap()
                    .decode(&mut full_reader);
                assert_eq!(compact_result, Ok(7));
                assert_eq!(compact_result, full_result);
                assert_eq!(compact_reader.snapshot(), full_reader.snapshot());
            }

            let mut compact_reader = BitReader::new(&[0x80, 0, 0]);
            let mut full_reader = BitReader::new(&[0x80, 0, 0]);
            assert_eq!(
                raw_only
                    .raw(HuffmanTableRole::Dc)
                    .unwrap()
                    .decode(&mut compact_reader),
                full.raw(HuffmanTableRole::Dc)
                    .unwrap()
                    .decode(&mut full_reader)
            );
            assert_eq!(compact_reader.snapshot(), full_reader.snapshot());
        }
    }

    #[test]
    fn raw_only_compact_canonical_bounds_match_full_derivation() {
        let cases = [
            RawHuffmanTable {
                bits: [0; 16],
                values: HuffmanValues::default(),
            },
            RawHuffmanTable {
                bits: [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                values: HuffmanValues::from_slice(&[17, 42]),
            },
            nonstandard_dc_shape_raw(),
            RawHuffmanTable {
                bits: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                values: HuffmanValues::from_slice(&[7]),
            },
        ];

        for raw in cases {
            let full_canonical = derive_canonical_huffman(&raw)
                .expect("valid table should derive full canonical metadata");
            let expected_lookup_bits = dynamic_raw_lookup_bits(&full_canonical);
            let mut live_bytes = 0;
            let raw_only = HuffmanTable::from_raw_with_requirement_and_live_budget(
                &raw,
                HuffmanTableRole::Dc,
                HuffmanLookupRequirement::Raw,
                &mut live_bytes,
                j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            )
            .expect("valid Raw-only table should compile");
            let full = HuffmanTable::from_raw(&raw, HuffmanTableRole::Dc)
                .expect("valid full table should compile");
            let legacy = legacy_lookups(&raw, expected_lookup_bits)
                .expect("valid table should build the legacy lookup");
            let HuffmanTable::Dynamic(raw_compiled) = &raw_only else {
                panic!("nonstandard Raw-only table must use dynamic storage");
            };

            assert_eq!(raw_only.metadata().min_code, full.metadata().min_code);
            assert_eq!(raw_only.metadata().max_code, full.metadata().max_code);
            assert_eq!(raw_only.metadata().val_offset, full.metadata().val_offset);
            assert_eq!(
                raw_only.metadata().raw_lookup_shift,
                FAST_BITS - expected_lookup_bits
            );
            assert_eq!(raw_compiled.fast, legacy.fast);
            assert!(raw_only.matches_raw_identity(&raw, HuffmanTableRole::Dc));
            assert!(!raw_only.matches_raw_identity(&raw, HuffmanTableRole::Ac));
            assert_eq!(live_bytes, raw_only.retained_allocation_bytes());
            assert_eq!(
                raw_only.retained_allocation_bytes(),
                HuffmanTable::COMPILED_OWNER_BYTES
                    + (1usize << expected_lookup_bits) * core::mem::size_of::<(u8, u8)>()
            );
        }
    }

    #[test]
    fn raw_only_invalid_tables_report_canonical_error_before_budget_error() {
        let cases = [
            (
                "oversubscribed",
                RawHuffmanTable {
                    bits: [1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    values: HuffmanValues::from_slice(&[0, 1, 2, 3, 4]),
                },
            ),
            (
                "BITS exceed HUFFVAL",
                RawHuffmanTable {
                    bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    values: HuffmanValues::default(),
                },
            ),
            (
                "BITS shorter than HUFFVAL",
                RawHuffmanTable {
                    bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    values: HuffmanValues::from_slice(&[0, 1]),
                },
            ),
            (
                "code after complete prefix",
                RawHuffmanTable {
                    bits: [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    values: HuffmanValues::from_slice(&[0, 1, 2]),
                },
            ),
        ];

        for (case, raw) in cases {
            let expected = derive_canonical_huffman(&raw)
                .expect_err("invalid table must fail full canonical validation");
            let initial_live_bytes = 7;
            let mut live_bytes = initial_live_bytes;
            let error = HuffmanTable::from_raw_with_requirement_and_live_budget(
                &raw,
                HuffmanTableRole::Dc,
                HuffmanLookupRequirement::Raw,
                &mut live_bytes,
                initial_live_bytes,
            )
            .expect_err("canonical validation must precede the insufficient budget");

            assert_eq!(error, expected, "{case}");
            assert_eq!(live_bytes, initial_live_bytes, "{case}");
        }
    }

    #[test]
    fn narrow_fused_ac_omits_the_full_raw_and_wide_lookups() {
        let raw = nonstandard_dc_shape_raw();
        let mut narrow_live_bytes = 0;
        let narrow = HuffmanTable::from_raw_with_requirement_and_live_budget(
            &raw,
            HuffmanTableRole::Ac,
            HuffmanLookupRequirement::FusedNarrow,
            &mut narrow_live_bytes,
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )
        .unwrap();
        let full = HuffmanTable::from_raw(&raw, HuffmanTableRole::Ac).unwrap();

        assert!(narrow.satisfies_lookup_requirement(HuffmanLookupRequirement::Raw));
        assert!(narrow.satisfies_lookup_requirement(HuffmanLookupRequirement::FusedNarrow));
        assert!(!narrow.satisfies_lookup_requirement(HuffmanLookupRequirement::FusedFull));
        let HuffmanTable::Dynamic(narrow_compiled) = &narrow else {
            panic!("nonstandard table must use dynamic storage");
        };
        assert_eq!(narrow_compiled.fast.len(), NARROW_RAW_ENTRIES);
        assert_eq!(narrow_compiled.primary.len(), NARROW_PRIMARY_ENTRIES);
        assert_eq!(narrow_compiled.primary8.as_slice(), [] as [u32; 0]);
        assert_eq!(narrow_compiled.packed.as_slice(), [] as [u32; 0]);
        assert_eq!(narrow_live_bytes, narrow.retained_allocation_bytes());

        let HuffmanTable::Dynamic(full_compiled) = &full else {
            panic!("nonstandard table must use dynamic storage");
        };
        assert_eq!(full_compiled.fast.len(), FAST_ENTRIES);
        assert_eq!(full_compiled.primary.len(), FULL_PRIMARY_ENTRIES);
        assert_eq!(full_compiled.primary8, narrow_compiled.primary);
        assert_eq!(full_compiled.packed.len(), FAST_ENTRIES);
        assert_eq!(
            full.retained_allocation_bytes() - narrow.retained_allocation_bytes(),
            (FAST_ENTRIES - NARROW_RAW_ENTRIES) * core::mem::size_of::<(u8, u8)>()
                + FULL_PRIMARY_ENTRIES * core::mem::size_of::<u32>()
                + FAST_ENTRIES * core::mem::size_of::<u32>()
        );

        let narrow_ac = narrow.ac_with_entries::<NARROW_PRIMARY_ENTRIES>().unwrap();
        assert!(!narrow_ac.uses_wide_lookup());
        assert!(!narrow_ac.with_wide_lookup(true).uses_wide_lookup());
    }

    #[test]
    fn narrow_dc_omits_the_full_raw_lookup_and_reports_narrow_capability() {
        let raw = nonstandard_dc_shape_raw();
        let mut narrow_live_bytes = 0;
        let narrow = HuffmanTable::from_raw_with_requirement_and_live_budget(
            &raw,
            HuffmanTableRole::Dc,
            HuffmanLookupRequirement::FusedNarrow,
            &mut narrow_live_bytes,
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )
        .unwrap();
        let full = HuffmanTable::from_raw(&raw, HuffmanTableRole::Dc).unwrap();

        assert!(narrow.satisfies_lookup_requirement(HuffmanLookupRequirement::Raw));
        assert!(narrow.satisfies_lookup_requirement(HuffmanLookupRequirement::FusedNarrow));
        assert!(!narrow.satisfies_lookup_requirement(HuffmanLookupRequirement::FusedFull));
        let HuffmanTable::Dynamic(narrow_compiled) = &narrow else {
            panic!("nonstandard table must use dynamic storage");
        };
        let HuffmanTable::Dynamic(full_compiled) = &full else {
            panic!("nonstandard table must use dynamic storage");
        };
        assert_eq!(narrow_compiled.fast.len(), NARROW_RAW_ENTRIES);
        assert_eq!(narrow_compiled.primary.len(), NARROW_PRIMARY_ENTRIES);
        assert_eq!(narrow_compiled.primary8.as_slice(), [] as [u32; 0]);
        assert_eq!(full_compiled.primary.len(), FULL_PRIMARY_ENTRIES);
        assert_eq!(narrow_compiled.primary, full_compiled.primary8);
        assert_eq!(narrow_compiled.packed.as_slice(), [] as [u32; 0]);
        assert_eq!(full_compiled.fast.len(), FAST_ENTRIES);
        assert_eq!(
            full.retained_allocation_bytes() - narrow.retained_allocation_bytes(),
            (FAST_ENTRIES - NARROW_RAW_ENTRIES) * core::mem::size_of::<(u8, u8)>()
                + FULL_PRIMARY_ENTRIES * core::mem::size_of::<u32>()
        );
        assert_eq!(narrow_live_bytes, narrow.retained_allocation_bytes());
    }

    #[test]
    fn narrow_dc_codes_beyond_primary8_match_full12_result_and_state() {
        for code_length in 9..=16 {
            let mut bits = [0; 16];
            bits[code_length - 1] = 1;
            let raw = RawHuffmanTable {
                bits,
                values: HuffmanValues::from_slice(&[0]),
            };
            let mut live_bytes = 0;
            let narrow = HuffmanTable::from_raw_with_requirement_and_live_budget(
                &raw,
                HuffmanTableRole::Dc,
                HuffmanLookupRequirement::FusedNarrow,
                &mut live_bytes,
                j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            )
            .unwrap();
            let full = HuffmanTable::from_raw(&raw, HuffmanTableRole::Dc).unwrap();
            let bytes = [0u8; 8];

            let mut narrow_reader = BitReader::new(&bytes);
            let mut full_reader = BitReader::new(&bytes);
            assert_eq!(
                narrow
                    .dc_with_entries::<NARROW_PRIMARY_ENTRIES>()
                    .unwrap()
                    .decode_fast_dc(&mut narrow_reader),
                full.dc().unwrap().decode_fast_dc(&mut full_reader),
                "sequential code length {code_length}"
            );
            assert_eq!(
                narrow_reader.snapshot(),
                full_reader.snapshot(),
                "sequential code length {code_length}"
            );

            let mut narrow_reader = BitReader::new(&bytes);
            let mut full_reader = BitReader::new(&bytes);
            assert_eq!(
                narrow
                    .dc_with_entries::<NARROW_PRIMARY_ENTRIES>()
                    .unwrap()
                    .decode_lossless_diff(&mut narrow_reader),
                full.dc().unwrap().decode_lossless_diff(&mut full_reader),
                "lossless code length {code_length}"
            );
            assert_eq!(
                narrow_reader.snapshot(),
                full_reader.snapshot(),
                "lossless code length {code_length}"
            );
        }
    }

    #[test]
    fn primary8_magnitude_boundary_matches_full12_reader_state() {
        for (role, symbols) in [
            (HuffmanTableRole::Dc, [7u8, 8, 11, 12]),
            (HuffmanTableRole::Ac, [0x07u8, 0x08, 0x0b, 0x0c]),
        ] {
            for symbol in symbols {
                let raw = RawHuffmanTable {
                    bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    values: HuffmanValues::from_slice(&[symbol]),
                };
                let mut live_bytes = 0;
                let narrow = HuffmanTable::from_raw_with_requirement_and_live_budget(
                    &raw,
                    role,
                    HuffmanLookupRequirement::FusedNarrow,
                    &mut live_bytes,
                    j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                )
                .unwrap();
                let full = HuffmanTable::from_raw(&raw, role).unwrap();
                let total_bits = usize::from((symbol & 0x0f) + 1);
                let mut bytes = [0u8; 8];
                bytes[total_bits / 8] |= 1 << (7 - total_bits % 8);
                let mut narrow_reader = BitReader::new(&bytes);
                let mut full_reader = BitReader::new(&bytes);

                match role {
                    HuffmanTableRole::Dc => assert_eq!(
                        narrow
                            .dc_with_entries::<NARROW_PRIMARY_ENTRIES>()
                            .unwrap()
                            .decode_fast_dc(&mut narrow_reader),
                        full.dc().unwrap().decode_fast_dc(&mut full_reader)
                    ),
                    HuffmanTableRole::Ac => assert_eq!(
                        narrow
                            .ac_with_entries::<NARROW_PRIMARY_ENTRIES>()
                            .unwrap()
                            .decode_fast_ac::<false>(&mut narrow_reader),
                        full.ac().unwrap().decode_fast_ac::<false>(&mut full_reader)
                    ),
                }
                assert_eq!(narrow_reader.snapshot(), full_reader.snapshot());
                assert_eq!(narrow_reader.read_bits(1).unwrap(), 1);
                assert_eq!(full_reader.read_bits(1).unwrap(), 1);

                if role == HuffmanTableRole::Ac {
                    let mut narrow_reader = BitReader::new(&bytes);
                    let mut full_reader = BitReader::new(&bytes);
                    assert_eq!(
                        narrow
                            .ac_with_entries::<NARROW_PRIMARY_ENTRIES>()
                            .unwrap()
                            .skip_fast_ac::<false>(&mut narrow_reader),
                        full.ac().unwrap().skip_fast_ac::<false>(&mut full_reader)
                    );
                    assert_eq!(narrow_reader.snapshot(), full_reader.snapshot());
                    assert_eq!(narrow_reader.read_bits(1).unwrap(), 1);
                    assert_eq!(full_reader.read_bits(1).unwrap(), 1);
                }
            }
        }
    }

    fn reader_at_final_byte(bytes: &[u8], allow_eof_padding: bool) -> BitReader<'_> {
        let mut reader = BitReader::new_with_eof_padding(bytes, allow_eof_padding);
        reader.ensure_bits_padded(FULL_PRIMARY_BITS).unwrap();
        reader.consume_bits(48);
        reader
    }

    fn decode_final_symbol<const N: usize>(
        table: &HuffmanTable,
        role: HuffmanTableRole,
        reader: &mut BitReader<'_>,
    ) -> Result<(), JpegError> {
        match role {
            HuffmanTableRole::Dc => table
                .dc_with_entries::<N>()?
                .decode_fast_dc(reader)
                .map(|_| ()),
            HuffmanTableRole::Ac => table
                .ac_with_entries::<N>()?
                .skip_fast_ac::<false>(reader)
                .map(|_| ()),
        }
    }

    #[test]
    fn primary8_final_symbol_matches_full12_marker_and_eof_contract() {
        for role in [HuffmanTableRole::Dc, HuffmanTableRole::Ac] {
            let raw = RawHuffmanTable {
                bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                values: HuffmanValues::from_slice(&[0]),
            };
            let mut live_bytes = 0;
            let narrow = HuffmanTable::from_raw_with_requirement_and_live_budget(
                &raw,
                role,
                HuffmanLookupRequirement::FusedNarrow,
                &mut live_bytes,
                j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            )
            .unwrap();
            let full = HuffmanTable::from_raw(&raw, role).unwrap();

            // Seven buffered bytes leave one final code byte unread in the
            // reservoir and the following EOI unobserved. The next symbol
            // must retain primary12's 12-bit probe and expose that marker.
            let marker_bytes = [0, 0, 0, 0, 0, 0, 0x7f, 0xff, 0xd9];
            let mut narrow_reader = reader_at_final_byte(&marker_bytes, false);
            let mut full_reader = reader_at_final_byte(&marker_bytes, false);
            assert_eq!(
                decode_final_symbol::<NARROW_PRIMARY_ENTRIES>(&narrow, role, &mut narrow_reader,),
                decode_final_symbol::<FULL_PRIMARY_ENTRIES>(&full, role, &mut full_reader)
            );
            assert_eq!(narrow_reader.snapshot(), full_reader.snapshot());
            let narrow_finish =
                crate::entropy::sequential::finish_scan(&mut narrow_reader, true).unwrap();
            let full_finish =
                crate::entropy::sequential::finish_scan(&mut full_reader, true).unwrap();
            assert_eq!(narrow_finish, full_finish);
            assert_eq!(narrow_finish, []);

            // Physical EOF without the parser's explicit padding permission
            // must fail at the same 12-bit lookahead and preserve reader state.
            let eof_bytes = [0, 0, 0, 0, 0, 0, 0x7f];
            let mut narrow_reader = reader_at_final_byte(&eof_bytes, false);
            let mut full_reader = reader_at_final_byte(&eof_bytes, false);
            let narrow_result =
                decode_final_symbol::<NARROW_PRIMARY_ENTRIES>(&narrow, role, &mut narrow_reader);
            let full_result =
                decode_final_symbol::<FULL_PRIMARY_ENTRIES>(&full, role, &mut full_reader);
            assert_eq!(narrow_result, full_result);
            assert!(matches!(
                narrow_result,
                Err(JpegError::HuffmanDecode {
                    reason: HuffmanFailure::TableExhausted,
                    ..
                })
            ));
            assert_eq!(narrow_reader.snapshot(), full_reader.snapshot());

            // Explicit EOF padding decodes the same final symbol, then both
            // finish paths report the same MissingEoi warning.
            let mut narrow_reader = reader_at_final_byte(&eof_bytes, true);
            let mut full_reader = reader_at_final_byte(&eof_bytes, true);
            assert_eq!(
                decode_final_symbol::<NARROW_PRIMARY_ENTRIES>(&narrow, role, &mut narrow_reader,),
                decode_final_symbol::<FULL_PRIMARY_ENTRIES>(&full, role, &mut full_reader)
            );
            assert_eq!(narrow_reader.snapshot(), full_reader.snapshot());
            let narrow_finish =
                crate::entropy::sequential::finish_scan(&mut narrow_reader, true).unwrap();
            let full_finish =
                crate::entropy::sequential::finish_scan(&mut full_reader, true).unwrap();
            assert_eq!(narrow_finish, full_finish);
            assert_eq!(
                narrow_finish,
                alloc::vec![crate::error::Warning::MissingEoi]
            );
        }
    }

    #[test]
    fn narrow_dc_marker_bounded_codes_match_full_error_and_state() {
        let marker_bounded = [0x80, 0x00, 0xff, 0xd9];
        for code_length in 9usize..=16 {
            let mut bits = [0; 16];
            bits[code_length - 1] = 1;
            let raw = RawHuffmanTable {
                bits,
                values: HuffmanValues::from_slice(&[0]),
            };
            let mut live_bytes = 0;
            let narrow = HuffmanTable::from_raw_with_requirement_and_live_budget(
                &raw,
                HuffmanTableRole::Dc,
                HuffmanLookupRequirement::FusedNarrow,
                &mut live_bytes,
                j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            )
            .unwrap();
            let full = HuffmanTable::from_raw(&raw, HuffmanTableRole::Dc).unwrap();
            let mut narrow_reader = BitReader::new(&marker_bounded);
            let mut full_reader = BitReader::new(&marker_bounded);
            assert_eq!(narrow_reader.read_bits(1).unwrap(), 1);
            assert_eq!(full_reader.read_bits(1).unwrap(), 1);

            let narrow_result = narrow
                .dc_with_entries::<NARROW_PRIMARY_ENTRIES>()
                .unwrap()
                .decode_fast_dc(&mut narrow_reader);
            let full_result = full.dc().unwrap().decode_fast_dc(&mut full_reader);
            assert_eq!(narrow_result, full_result, "code length {code_length}");
            assert_eq!(
                narrow_reader.snapshot(),
                full_reader.snapshot(),
                "code length {code_length}"
            );
            if code_length <= 15 {
                assert_eq!(narrow_result.unwrap(), 0);
            } else {
                // After the leading bit is consumed, the marker leaves fifteen
                // real zero bits. Padded lookahead appends one, so code 1 is
                // greater than this table's sole 16-bit canonical code 0. The
                // retained full-table slow path reports CodeOverflow as well.
                assert!(matches!(
                    narrow_result,
                    Err(JpegError::HuffmanDecode {
                        reason: HuffmanFailure::CodeOverflow,
                        ..
                    })
                ));
            }
        }
    }

    #[test]
    fn narrow_ac_codes_beyond_primary8_match_full12_decode_and_skip() {
        for code_length in 9..=16 {
            for symbol in [0, 0xf0, 0x01] {
                let mut bits = [0; 16];
                bits[code_length - 1] = 1;
                let raw = RawHuffmanTable {
                    bits,
                    values: HuffmanValues::from_slice(&[symbol]),
                };
                let mut live_bytes = 0;
                let narrow = HuffmanTable::from_raw_with_requirement_and_live_budget(
                    &raw,
                    HuffmanTableRole::Ac,
                    HuffmanLookupRequirement::FusedNarrow,
                    &mut live_bytes,
                    j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                )
                .unwrap();
                let full = HuffmanTable::from_raw(&raw, HuffmanTableRole::Ac).unwrap();
                let bytes = [0u8; 8];
                let mut narrow_reader = BitReader::new(&bytes);
                let mut full_reader = BitReader::new(&bytes);

                assert_eq!(
                    narrow
                        .ac_with_entries::<NARROW_PRIMARY_ENTRIES>()
                        .unwrap()
                        .decode_fast_ac::<false>(&mut narrow_reader)
                        .unwrap(),
                    full.ac()
                        .unwrap()
                        .decode_fast_ac::<false>(&mut full_reader)
                        .unwrap(),
                    "symbol {symbol:#04x}, code length {code_length}"
                );
                assert_eq!(
                    narrow_reader.snapshot(),
                    full_reader.snapshot(),
                    "symbol {symbol:#04x}, code length {code_length}"
                );

                let mut narrow_reader = BitReader::new(&bytes);
                let mut full_reader = BitReader::new(&bytes);
                assert_eq!(
                    narrow
                        .ac_with_entries::<NARROW_PRIMARY_ENTRIES>()
                        .unwrap()
                        .skip_fast_ac::<false>(&mut narrow_reader),
                    full.ac().unwrap().skip_fast_ac::<false>(&mut full_reader),
                    "skip symbol {symbol:#04x}, code length {code_length}"
                );
                assert_eq!(
                    narrow_reader.snapshot(),
                    full_reader.snapshot(),
                    "skip symbol {symbol:#04x}, code length {code_length}"
                );
            }
        }
    }

    #[test]
    fn compiled_table_budget_is_transactional_and_uses_actual_role_payload() {
        let raw = nonstandard_dc_shape_raw();
        for role in [HuffmanTableRole::Dc, HuffmanTableRole::Ac] {
            let expected_bytes = HuffmanTable::from_raw(&raw, role)
                .unwrap()
                .retained_allocation_bytes();
            let initial_live_bytes = 7;
            let exact_cap = initial_live_bytes + expected_bytes;

            let mut live_bytes = initial_live_bytes;
            let error =
                HuffmanTable::from_raw_with_live_budget(&raw, role, &mut live_bytes, exact_cap - 1)
                    .expect_err("one byte below the actual payload must fail");
            assert!(matches!(
                error,
                JpegError::MemoryCapExceeded { requested, cap }
                    if requested == exact_cap && cap == exact_cap - 1
            ));
            assert_eq!(live_bytes, initial_live_bytes);

            let table =
                HuffmanTable::from_raw_with_live_budget(&raw, role, &mut live_bytes, exact_cap)
                    .expect("exact role-specific payload must fit");
            assert_eq!(live_bytes, exact_cap);
            assert_eq!(table.retained_allocation_bytes(), expected_bytes);
        }
    }

    #[test]
    fn prepared_arena_shared_payload_charge_is_transactional() {
        let table =
            HuffmanTable::from_raw(&nonstandard_dc_shape_raw(), HuffmanTableRole::Dc).unwrap();
        let table_bytes = table.retained_allocation_bytes();
        let mut live_bytes = 0;
        let mut arena = PreparedHuffmanTables::try_with_capacity_and_live_budget(
            1,
            &mut live_bytes,
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )
        .unwrap();
        let arena_bytes = live_bytes;

        let error = arena
            .push_with_live_budget(
                table.clone(),
                &mut live_bytes,
                arena_bytes + table_bytes - 1,
            )
            .expect_err("one byte below the conservative owner charge must fail");
        assert!(matches!(error, JpegError::MemoryCapExceeded { .. }));
        assert_eq!(live_bytes, arena_bytes);
        assert_eq!(arena.len(), 0);

        arena
            .push_with_live_budget(table, &mut live_bytes, arena_bytes + table_bytes)
            .expect("exact conservative owner charge must fit");
        assert_eq!(live_bytes, arena_bytes + table_bytes);
        assert_eq!(arena.len(), 1);
    }

    #[test]
    fn role_specializations_match_legacy_tables_for_all_fast_indices_and_code_lengths() {
        let standard_dc = luma_dc_raw();
        // Annex K luminance AC symbol order.
        let standard_ac_values = [
            0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51,
            0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1,
            0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18,
            0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
            0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57,
            0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
            0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92,
            0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
            0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
            0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8,
            0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2,
            0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
        ];
        let standard_ac = RawHuffmanTable {
            bits: [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D],
            values: HuffmanValues::from_slice(&standard_ac_values),
        };
        let length_values: Vec<u8> = (0..16).collect();
        let length_coverage = RawHuffmanTable {
            bits: [1; 16],
            values: HuffmanValues::from_slice(&length_values),
        };
        let empty = RawHuffmanTable {
            bits: [0; 16],
            values: HuffmanValues::default(),
        };
        let complete = RawHuffmanTable {
            bits: [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0, 0xF0]),
        };

        for raw in [
            &standard_dc,
            &standard_ac,
            &length_coverage,
            &empty,
            &complete,
        ] {
            assert_matches_legacy(raw);
        }

        // Short codes exercise packed magnitudes and every run/EOB/ZRL symbol;
        // longer codes also verify that both specializations keep the slow path.
        for code_length in 1..=16 {
            for symbol in 0..=u8::MAX {
                let mut bits = [0; 16];
                bits[code_length - 1] = 1;
                assert_matches_legacy(&RawHuffmanTable {
                    bits,
                    values: HuffmanValues::from_slice(&[symbol]),
                });
            }
        }

        let oversubscribed = RawHuffmanTable {
            bits: [1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0, 1, 2, 3, 4]),
        };
        let Err(legacy_error) = legacy_lookups(&oversubscribed, FAST_BITS) else {
            panic!("legacy builder must reject an oversubscribed tree");
        };
        for role in [HuffmanTableRole::Dc, HuffmanTableRole::Ac] {
            assert_eq!(
                HuffmanTable::from_raw(&oversubscribed, role).unwrap_err(),
                legacy_error
            );
        }
    }

    fn assert_matches_legacy(raw: &RawHuffmanTable) {
        for role in [HuffmanTableRole::Dc, HuffmanTableRole::Ac] {
            let table = HuffmanTable::from_raw(raw, role).unwrap();
            for (bits, actual) in [
                (FAST_BITS, table.packed_lookup()),
                (
                    FULL_PRIMARY_BITS,
                    table.primary::<FULL_PRIMARY_ENTRIES>().unwrap().as_slice(),
                ),
                (
                    NARROW_PRIMARY_BITS,
                    table
                        .primary::<NARROW_PRIMARY_ENTRIES>()
                        .unwrap()
                        .as_slice(),
                ),
            ] {
                let legacy = legacy_lookups(raw, bits).unwrap();
                assert_eq!(table.metadata().min_code, legacy.min_code);
                assert_eq!(table.metadata().max_code, legacy.max_code);
                assert_eq!(table.metadata().val_offset, legacy.val_offset);
                if bits == FAST_BITS {
                    assert_eq!(table.fast().unwrap().as_slice(), legacy.fast.as_slice());
                }
                if role == HuffmanTableRole::Dc && bits == FAST_BITS {
                    assert_eq!(table.packed_lookup(), [] as [u32; 0]);
                    continue;
                }
                let expected = match role {
                    HuffmanTableRole::Dc => legacy.fast_dc,
                    HuffmanTableRole::Ac => (0..(1usize << bits))
                        .map(|index| {
                            let fused = legacy.fast_ac[index];
                            if fused != 0 {
                                fused
                            } else {
                                let (symbol, length) = legacy.fast[index];
                                u32::from(length) | (u32::from(symbol) << 8)
                            }
                        })
                        .collect(),
                };
                assert_eq!(actual, expected.as_slice(), "lookup bits {bits}");
            }
        }
    }

    /// Standard JPEG luminance DC table from Annex K.3 — well-known fixture.
    /// `bits[0..16]` counts per length; `values` lists the symbols in order.
    fn luma_dc_raw() -> RawHuffmanTable {
        RawHuffmanTable {
            bits: [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        }
    }

    fn nonstandard_dc_shape_raw() -> RawHuffmanTable {
        RawHuffmanTable {
            bits: STD_LUMA_DC_BITS,
            values: HuffmanValues::from_slice(&[11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
        }
    }

    #[test]
    fn annex_k_tables_reuse_static_storage_without_charging_live_heap() {
        let cases = [
            (
                RawHuffmanTable {
                    bits: STD_LUMA_DC_BITS,
                    values: HuffmanValues::from_slice(&STD_LUMA_DC_VALUES),
                },
                HuffmanTableRole::Dc,
            ),
            (
                RawHuffmanTable {
                    bits: STD_CHROMA_DC_BITS,
                    values: HuffmanValues::from_slice(&STD_CHROMA_DC_VALUES),
                },
                HuffmanTableRole::Dc,
            ),
            (
                RawHuffmanTable {
                    bits: STD_LUMA_AC_BITS,
                    values: HuffmanValues::from_slice(&STD_LUMA_AC_VALUES),
                },
                HuffmanTableRole::Ac,
            ),
            (
                RawHuffmanTable {
                    bits: STD_CHROMA_AC_BITS,
                    values: HuffmanValues::from_slice(&STD_CHROMA_AC_VALUES),
                },
                HuffmanTableRole::Ac,
            ),
        ];
        for (raw, role) in cases {
            let mut live_bytes = 19;
            let table = HuffmanTable::from_raw_with_live_budget(&raw, role, &mut live_bytes, 19)
                .expect("standard tables require no retained heap payload");
            assert!(matches!(&table, HuffmanTable::Static(_)));
            assert_eq!(table.retained_allocation_bytes(), 0);
            assert_eq!(live_bytes, 19);
            assert_matches_legacy(&raw);
        }
    }

    #[test]
    fn builds_fast_table_from_standard_luma_dc() {
        let table = HuffmanTable::from_raw(&luma_dc_raw(), HuffmanTableRole::Dc).unwrap();
        let (sym, len) = table.fast().unwrap()[0b0000_0000_0000 << (FAST_BITS - 12)];
        assert_eq!((sym, len), (0, 2));
        let (sym, len) = table.fast().unwrap()[0b0011_1111_1111 << (FAST_BITS - 12)];
        assert_eq!((sym, len), (0, 2));
        let (sym, len) = table.fast().unwrap()[0b0100_0000_0000 << (FAST_BITS - 12)];
        assert_eq!((sym, len), (1, 3));
    }

    #[test]
    fn widened_fast_table_covers_9_bit_luma_dc_code() {
        let table = HuffmanTable::from_raw(&luma_dc_raw(), HuffmanTableRole::Dc).unwrap();
        let idx = 0b1_1111_1110usize << usize::from(FAST_BITS - 9);
        let (sym, len) = table.fast().unwrap().get(idx).copied().unwrap_or((0, 0));
        assert_eq!((sym, len), (11, 9));
    }

    #[test]
    fn prepared_arena_ids_are_checked_and_capacity_bounded() {
        let table =
            HuffmanTable::from_raw(&luma_dc_raw(), HuffmanTableRole::Dc).expect("valid fixture");
        let mut arena = PreparedHuffmanTables::try_with_capacity(1).expect("bounded arena");
        let id = arena.push(table.clone()).expect("reserved slot");

        assert_eq!(arena.get(id), Some(&table));
        assert!(matches!(
            arena.push(table),
            Err(JpegError::InternalInvariant {
                reason: "prepared Huffman arena exceeded its reserved capacity"
            })
        ));
    }

    #[test]
    fn prepared_arena_validates_roles_and_accounts_actual_capacity() {
        let raw = luma_dc_raw();
        let mut arena = PreparedHuffmanTables::try_with_capacity(2).expect("bounded arena");
        let dc = arena
            .push(HuffmanTable::from_raw(&raw, HuffmanTableRole::Dc).unwrap())
            .unwrap();
        let ac = arena
            .push(HuffmanTable::from_raw(&raw, HuffmanTableRole::Ac).unwrap())
            .unwrap();

        assert!(arena.get_dc(dc).is_ok());
        assert!(arena.get_ac(dc).is_err());
        assert!(arena.get_ac(ac).is_ok());
        assert!(arena.get_dc(ac).is_err());
        let retained_table_bytes = arena.get(dc).unwrap().retained_allocation_bytes()
            + arena.get(ac).unwrap().retained_allocation_bytes();
        assert_eq!(
            arena.retained_allocation_bytes().unwrap(),
            PreparedHuffmanTables::allocation_bytes_for_capacity(arena.capacity()).unwrap()
                + retained_table_bytes
        );
    }

    #[test]
    fn rejects_oversubscribed_code_table() {
        let raw = RawHuffmanTable {
            bits: [1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0, 1, 2, 3, 4]),
        };
        let err = HuffmanTable::from_raw(&raw, HuffmanTableRole::Dc).unwrap_err();
        assert!(matches!(
            err,
            JpegError::HuffmanDecode {
                reason: HuffmanFailure::CodeOverflow,
                ..
            }
        ));
    }

    #[test]
    fn accepts_complete_prefix_table_for_decoder_compatibility() {
        let raw = RawHuffmanTable {
            bits: [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0, 1]),
        };
        HuffmanTable::from_raw(&raw, HuffmanTableRole::Dc)
            .expect("CPU decoder accepts a complete prefix table");
    }

    #[test]
    fn handles_empty_table_without_panic() {
        let raw = RawHuffmanTable {
            bits: [0; 16],
            values: HuffmanValues::default(),
        };
        let table = HuffmanTable::from_raw(&raw, HuffmanTableRole::Dc).unwrap();
        assert!(table.fast().unwrap().iter().all(|&(_, len)| len == 0));
    }

    /// Exercises every standard JPEG luma DC code — Annex K.3.
    fn luma_dc_code_cases() -> &'static [(u32, u8, u8)] {
        &[
            (0b00, 2, 0),
            (0b010, 3, 1),
            (0b011, 3, 2),
            (0b100, 3, 3),
            (0b101, 3, 4),
            (0b110, 3, 5),
            (0b1110, 4, 6),
            (0b1_1110, 5, 7),
            (0b11_1110, 6, 8),
            (0b111_1110, 7, 9),
            (0b1111_1110, 8, 10),
            (0b1_1111_1110, 9, 11),
        ]
    }

    #[test]
    fn decodes_all_standard_luma_dc_codes() {
        let table = HuffmanTable::from_raw(&luma_dc_raw(), HuffmanTableRole::Dc).unwrap();
        for &(code, len, expected) in luma_dc_code_cases() {
            let shift = 32 - len;
            let aligned = code << shift;
            let bytes = aligned.to_be_bytes();
            let mut br = BitReader::new(&bytes);
            let sym = table.decode(&mut br).unwrap();
            assert_eq!(sym, expected, "code={code:b} len={len}");
        }
    }

    #[test]
    fn fast_dc_decodes_symbol_and_magnitude_in_one_lookup() {
        let table = HuffmanTable::from_raw(&luma_dc_raw(), HuffmanTableRole::Dc).unwrap();
        // Standard luma DC code `011` => category 2, followed by magnitude
        // bits `10` => diff +2. The fast DC path should consume all 5 bits.
        let bytes = [0b0111_0000u8, 0, 0, 0, 0, 0, 0, 0];
        let mut br = BitReader::new(&bytes);

        let diff = table.dc().unwrap().decode_fast_dc(&mut br).unwrap();

        assert_eq!(diff, 2);
        assert_eq!(br.snapshot().bits, 51);
        assert_eq!(br.peek_bits(3), 0);
    }

    #[test]
    fn decodes_single_bit_table_before_marker_padding() {
        let raw = RawHuffmanTable {
            bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0]),
        };
        let table = HuffmanTable::from_raw(&raw, HuffmanTableRole::Dc).unwrap();
        let mut br = BitReader::new(&[0x7f, 0xff, 0xc4]);

        let symbol = table.decode(&mut br).unwrap();

        assert_eq!(symbol, 0);
    }

    #[test]
    fn decodes_9_plus_bit_codes_via_slow_path() {
        let table = HuffmanTable::from_raw(&luma_dc_raw(), HuffmanTableRole::Dc).unwrap();
        // Code `111111110` (9 bits) → symbol 11. A literal 0xFF in a JPEG
        // entropy stream must be byte-stuffed as `FF 00` (T.81 §F.1.2.3) so
        // the BitReader does not mistake it for a marker prefix.
        let bytes = [0xFFu8, 0x00, 0b0100_0000];
        let mut br = BitReader::new(&bytes);
        let sym = table.decode(&mut br).unwrap();
        assert_eq!(sym, 11);
    }

    #[test]
    fn reports_huffman_failure_on_truncated_bit_stream() {
        let table = HuffmanTable::from_raw(&luma_dc_raw(), HuffmanTableRole::Dc).unwrap();
        let bytes = [];
        let mut br = BitReader::new(&bytes);
        let err = table.decode(&mut br).unwrap_err();
        assert!(matches!(
            err,
            JpegError::HuffmanDecode {
                reason: HuffmanFailure::TableExhausted,
                ..
            }
        ));
    }
}
