// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared decode context for tile-oriented workloads.

use crate::allocation::{
    checked_add_allocation_bytes, checked_allocation_bytes, try_reserve_for_len_with_live_budget,
};
use crate::entropy::huffman::{HuffmanLookupRequirement, HuffmanTable, NARROW_PRIMARY_ENTRIES};
use crate::entropy::sequential::{PreparedComponentPlan, PreparedDecodePlan};
use crate::error::{JpegError, Warning};
use crate::info::Info;
use crate::parse::header::count_scan_markers;
use crate::parse::tables::{HuffmanTableRole, RawHuffmanTable};
use alloc::vec::Vec;
use core::mem::size_of;
use j2k_core::{CacheStats, CodecContext};

const QUANT_CACHE_SLOTS: usize = 8;
const HUFFMAN_CACHE_SLOTS: usize = 20;
const PLAN_CACHE_SLOTS: usize = 8;
const MAX_DECODE_PLAN_CACHE_BYTES: usize = 16 * 1024 * 1024;
const PRIMARY8_COMPANION_ALLOCATION_BYTES: usize =
    NARROW_PRIMARY_ENTRIES * size_of::<u32>() + size_of::<Vec<u32>>();
// Preserve the established common working set (2 DC + 8 AC tables) and its
// one in-flight construction slot after every Full table gains primary8.
const TABLE_CACHE_ALLOCATION_RESERVE_BYTES: usize =
    2 * 1024 * 1024 + 11 * PRIMARY8_COMPANION_ALLOCATION_BYTES;

/// Conservative retained-byte reservation used by decode workspace planning.
/// Decode plan keys, plans, and exact-header templates are hard-capped at
/// 16 MiB; the remaining table-cache reserve covers fallibly reserved Huffman
/// slots, compiled payloads, one construction peak, and bookkeeping.
pub(crate) const MAX_DECODER_CONTEXT_ALLOCATION_BYTES: usize =
    MAX_DECODE_PLAN_CACHE_BYTES + TABLE_CACHE_ALLOCATION_RESERVE_BYTES;

#[derive(Debug, Clone)]
struct CachedQuantTable {
    digest: u64,
    table: [u16; 64],
}

#[derive(Debug)]
struct CachedHuffmanTable {
    digest: u64,
    table: HuffmanTable,
}

#[derive(Debug)]
struct CachedDecodePlan {
    digest: u64,
    header_prefix: Vec<u8>,
    plan: PreparedDecodePlan,
    template: Option<CachedSequentialTemplate>,
    allocation_bytes: usize,
}

#[derive(Debug)]
struct CachedSequentialTemplate {
    info: Info,
    warnings: Vec<Warning>,
    parsed_bytes: usize,
    parsed_nonwarning_bytes: usize,
}

pub(crate) struct CachedDecodePlanLease<'a> {
    // The vacant slot's plan lives in the temporary decoder and its key/header
    // template in this guard; all stay charged until restoration or Drop.
    slot: &'a mut Option<CachedDecodePlan>,
    decode_plan_cache_bytes: &'a mut usize,
    cache_evictions: &'a mut u64,
    digest: u64,
    header_prefix: Vec<u8>,
    template: Option<CachedSequentialTemplate>,
    allocation_bytes: usize,
    restored: bool,
}

impl CachedDecodePlanLease<'_> {
    pub(crate) fn restore(mut self, plan: PreparedDecodePlan) {
        let header_prefix = core::mem::take(&mut self.header_prefix);
        *self.slot = Some(CachedDecodePlan {
            digest: self.digest,
            header_prefix,
            plan,
            template: self.template.take(),
            allocation_bytes: self.allocation_bytes,
        });
        self.restored = true;
    }
}

impl Drop for CachedDecodePlanLease<'_> {
    fn drop(&mut self) {
        if self.restored {
            return;
        }
        *self.decode_plan_cache_bytes = self
            .decode_plan_cache_bytes
            .saturating_sub(self.allocation_bytes);
        *self.cache_evictions = self.cache_evictions.saturating_add(1);
    }
}

/// Shared decode context for WSI tile batches.
///
/// Reuse one context across related JPEG tiles to avoid rebuilding Huffman
/// tables and decode plans when table definitions or header prefixes repeat.
/// Prepared sequential batches can also clone exact validated header metadata
/// without reparsing a matching prefix.
#[derive(Debug, Default)]
pub struct DecoderContext {
    quant_tables: [Option<CachedQuantTable>; QUANT_CACHE_SLOTS],
    huffman_tables: Vec<Option<CachedHuffmanTable>>,
    huffman_payload_bytes: usize,
    huffman_eviction_cursor: usize,
    huffman_cache_has_tombstones: bool,
    decode_plans: [Option<CachedDecodePlan>; PLAN_CACHE_SLOTS],
    decode_plan_cache_bytes: usize,
    cache_hits: u64,
    cache_misses: u64,
    cache_evictions: u64,
    #[cfg(test)]
    decode_plan_clones: u64,
}

impl DecoderContext {
    /// Create an empty decode context.
    #[must_use]
    pub fn new() -> Self {
        Self {
            quant_tables: core::array::from_fn(|_| None),
            huffman_tables: Vec::new(),
            huffman_payload_bytes: 0,
            huffman_eviction_cursor: 0,
            huffman_cache_has_tombstones: false,
            decode_plans: core::array::from_fn(|_| None),
            decode_plan_cache_bytes: 0,
            cache_hits: 0,
            cache_misses: 0,
            cache_evictions: 0,
            #[cfg(test)]
            decode_plan_clones: 0,
        }
    }

    pub(crate) fn resolve_quant_table(&mut self, table: [u16; 64]) -> [u16; 64] {
        let digest = digest_quant_table(&table);
        self.resolve_quant_table_with_digest(table, digest)
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the digest cast only selects a cache shard and intentionally uses the native word"
    )]
    pub(crate) fn cached_decode_plan(
        &self,
        header_prefix: &[u8],
    ) -> Option<(usize, &PreparedDecodePlan)> {
        let digest = digest_bytes(header_prefix);
        let start = (digest as usize) % self.decode_plans.len();
        (0..self.decode_plans.len())
            .map(|probe| (start + probe) % self.decode_plans.len())
            .find_map(|slot| {
                self.decode_plans[slot].as_ref().and_then(|cached| {
                    (cached.digest == digest && cached.header_prefix.as_slice() == header_prefix)
                        .then_some((slot, &cached.plan))
                })
            })
    }

    /// Lease a slot returned by the immediately preceding exact-key lookup.
    /// The caller validates the live budget without mutating this context
    /// between lookup and lease, so hashing and comparing the key once suffice.
    pub(crate) fn lease_cached_decode_plan(
        &mut self,
        slot: usize,
    ) -> Result<(CachedDecodePlanLease<'_>, PreparedDecodePlan), JpegError> {
        let cached = self
            .decode_plans
            .get_mut(slot)
            .and_then(Option::take)
            .ok_or(JpegError::InternalInvariant {
                reason: "matched cached decode plan disappeared before leasing",
            })?;
        self.cache_hits = self.cache_hits.saturating_add(1);
        let CachedDecodePlan {
            digest,
            header_prefix,
            plan,
            template,
            allocation_bytes,
        } = cached;
        Ok((
            CachedDecodePlanLease {
                slot: &mut self.decode_plans[slot],
                decode_plan_cache_bytes: &mut self.decode_plan_cache_bytes,
                cache_evictions: &mut self.cache_evictions,
                digest,
                header_prefix,
                template,
                allocation_bytes,
                restored: false,
            },
            plan,
        ))
    }

    /// Clone a prepared sequential decoder template whose complete validated
    /// header prefix matches `input`. Bytes after the first entropy offset stay
    /// input-specific and are checked by the normal entropy decoder.
    pub(crate) fn clone_cached_sequential_template(
        &mut self,
        input: &[u8],
        retained_external_bytes: usize,
    ) -> Result<Option<(Info, Vec<Warning>, PreparedDecodePlan)>, JpegError> {
        let Some(slot) = self.decode_plans.iter().position(|entry| {
            entry.as_ref().is_some_and(|cached| {
                cached.template.is_some()
                    && cached.plan.scan_offset == cached.header_prefix.len()
                    && input.starts_with(&cached.header_prefix)
            })
        }) else {
            return Ok(None);
        };
        let retained_context_bytes = self.retained_allocation_bytes();
        let cached = self.decode_plans[slot]
            .as_ref()
            .ok_or(JpegError::InternalInvariant {
                reason: "matched cached sequential template disappeared before cloning",
            })?;
        let template = cached
            .template
            .as_ref()
            .ok_or(JpegError::InternalInvariant {
                reason: "matched cached sequential plan is missing its header template",
            })?;
        let mut live_bytes =
            checked_add_allocation_bytes(retained_external_bytes, retained_context_bytes)?;
        let _parsed_live_bytes = checked_add_allocation_bytes(live_bytes, template.parsed_bytes)?;
        live_bytes = checked_add_allocation_bytes(live_bytes, template.parsed_nonwarning_bytes)?;
        let scan_count = count_scan_markers(input, cached.plan.scan_offset);
        if scan_count != 1 {
            return Err(JpegError::InvalidSequentialScanCount {
                sof: template.info.sof_kind,
                count: scan_count,
            });
        }

        self.cache_hits = self.cache_hits.saturating_add(1);
        let plan = try_clone_decode_plan(&cached.plan, &mut live_bytes, None)?.ok_or(
            JpegError::InternalInvariant {
                reason: "cached sequential plan unexpectedly bypassed cloning",
            },
        )?;
        let warnings = try_clone_warnings(&template.warnings, &mut live_bytes)?;
        ensure_cached_template_construction_fits(template.parsed_bytes, &plan)?;
        let info = template.info.clone();
        #[cfg(test)]
        {
            self.decode_plan_clones = self.decode_plan_clones.saturating_add(1);
        }
        Ok(Some((info, warnings, plan)))
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the digest cast only selects a cache shard and intentionally uses the native word"
    )]
    fn resolve_quant_table_with_digest(&mut self, table: [u16; 64], digest: u64) -> [u16; 64] {
        let start = (digest as usize) % self.quant_tables.len();
        for probe in 0..self.quant_tables.len() {
            let slot = (start + probe) % self.quant_tables.len();
            match &self.quant_tables[slot] {
                Some(cached) if cached.digest == digest && cached.table == table => {
                    self.cache_hits = self.cache_hits.saturating_add(1);
                    return cached.table;
                }
                None => {
                    self.quant_tables[slot] = Some(CachedQuantTable { digest, table });
                    self.cache_misses = self.cache_misses.saturating_add(1);
                    return table;
                }
                Some(_) => {}
            }
        }

        let slot = start;
        self.quant_tables[slot] = Some(CachedQuantTable { digest, table });
        self.cache_misses = self.cache_misses.saturating_add(1);
        self.cache_evictions = self.cache_evictions.saturating_add(1);
        table
    }

    /// The compiled table for `raw`, from the cache or compiled into it.
    /// Returned by reference so callers clone only the immutable Arc handle.
    #[cfg(test)]
    pub(crate) fn resolve_huffman_table_with_live_budget(
        &mut self,
        raw: &RawHuffmanTable,
        role: HuffmanTableRole,
        live_bytes: &mut usize,
        cap: usize,
    ) -> Result<&HuffmanTable, JpegError> {
        self.resolve_huffman_table_with_requirement_and_live_budget(
            raw,
            role,
            HuffmanLookupRequirement::FusedFull,
            live_bytes,
            cap,
        )
    }

    pub(crate) fn resolve_huffman_table_with_requirement_and_live_budget(
        &mut self,
        raw: &RawHuffmanTable,
        role: HuffmanTableRole,
        requirement: HuffmanLookupRequirement,
        live_bytes: &mut usize,
        cap: usize,
    ) -> Result<&HuffmanTable, JpegError> {
        let digest = digest_huffman_table(raw, role);
        self.resolve_huffman_table_with_digest_and_requirement_and_live_budget(
            raw,
            role,
            digest,
            requirement,
            live_bytes,
            cap,
        )
    }

    #[cfg(test)]
    fn resolve_huffman_table_with_digest_and_live_budget(
        &mut self,
        raw: &RawHuffmanTable,
        role: HuffmanTableRole,
        digest: u64,
        live_bytes: &mut usize,
        cap: usize,
    ) -> Result<&HuffmanTable, JpegError> {
        self.resolve_huffman_table_with_digest_and_requirement_and_live_budget(
            raw,
            role,
            digest,
            HuffmanLookupRequirement::FusedFull,
            live_bytes,
            cap,
        )
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the digest cast only selects a cache shard and intentionally uses the native word"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "cache probing, eviction, and live-byte accounting form one transactional update"
    )]
    fn resolve_huffman_table_with_digest_and_requirement_and_live_budget(
        &mut self,
        raw: &RawHuffmanTable,
        role: HuffmanTableRole,
        digest: u64,
        requirement: HuffmanLookupRequirement,
        live_bytes: &mut usize,
        cap: usize,
    ) -> Result<&HuffmanTable, JpegError> {
        self.ensure_huffman_cache_slots(live_bytes, cap)?;
        let slots = self.huffman_tables.len();
        let start = (digest as usize) % slots;
        let mut exact = None;
        let mut empty = None;
        // In a cache with no deletion holes, insertion into the first vacant
        // slot guarantees that an exact entry cannot appear after a vacancy.
        // Byte-cap eviction makes the tombstone flag sticky and restores the
        // full scan required to find survivors beyond those holes.
        for probe in 0..slots {
            let slot = (start + probe) % slots;
            let Some(cached) = self.huffman_tables[slot].as_ref() else {
                empty.get_or_insert(slot);
                if !self.huffman_cache_has_tombstones {
                    break;
                }
                continue;
            };
            if cached.digest == digest && cached.table.matches_raw_identity(raw, role) {
                exact = Some(slot);
                break;
            }
        }
        let hit = exact.filter(|&slot| {
            requirement == HuffmanLookupRequirement::Raw
                || self.huffman_tables[slot]
                    .as_ref()
                    .is_some_and(|cached| cached.table.satisfies_lookup_requirement(requirement))
        });
        if let Some(slot) = hit {
            self.cache_hits = self.cache_hits.saturating_add(1);
            return self.huffman_tables[slot]
                .as_ref()
                .map(|cached| &cached.table)
                .ok_or(JpegError::InternalInvariant {
                    reason: "matched Huffman cache slot disappeared",
                });
        }

        let slot_bytes = self.huffman_slot_allocation_bytes()?;
        let retained_bytes = self.huffman_payload_bytes;
        let retained_payload_cap = TABLE_CACHE_ALLOCATION_RESERVE_BYTES
            .checked_sub(slot_bytes)
            .and_then(|bytes| bytes.checked_sub(HuffmanTable::maximum_retained_allocation_bytes()))
            .ok_or(JpegError::InternalInvariant {
                reason: "Huffman cache metadata and construction reserve exceed their budget",
            })?;
        let construction_headroom = TABLE_CACHE_ALLOCATION_RESERVE_BYTES
            .checked_sub(slot_bytes)
            .and_then(|bytes| bytes.checked_sub(retained_bytes))
            .ok_or(JpegError::InternalInvariant {
                reason: "retained Huffman cache exceeds its construction-peak budget",
            })?;
        let context_peak_cap =
            live_bytes
                .checked_add(construction_headroom)
                .ok_or(JpegError::MemoryCapExceeded {
                    requested: usize::MAX,
                    cap,
                })?;
        let build_cap = cap.min(context_peak_cap);
        let mut projected_live_bytes = *live_bytes;
        let table = HuffmanTable::from_raw_with_requirement_and_live_budget(
            raw,
            role,
            requirement,
            &mut projected_live_bytes,
            build_cap,
        )?;
        let table_bytes = table.retained_allocation_bytes();
        if table_bytes > HuffmanTable::maximum_retained_allocation_bytes() {
            return Err(JpegError::InternalInvariant {
                reason: "compiled Huffman table exceeded its reserved maximum allocation",
            });
        }

        let mut victims = [false; HUFFMAN_CACHE_SLOTS];
        let mut victim_count = 0u64;
        let mut evicted_bytes = 0usize;
        let mut retained_bytes = retained_bytes;
        let mut next_cursor = self.huffman_eviction_cursor;
        // Any exact entry reaching this miss path has a lower lookup
        // capability than the requested rank, so upgrade it transactionally.
        let replacement = exact;
        let slot = if let Some(slot) = replacement {
            victims[slot] = true;
            victim_count += 1;
            let bytes = self.huffman_tables[slot]
                .as_ref()
                .map_or(0, |cached| cached.table.retained_allocation_bytes());
            evicted_bytes = evicted_bytes.saturating_add(bytes);
            retained_bytes =
                retained_bytes
                    .checked_sub(bytes)
                    .ok_or(JpegError::InternalInvariant {
                        reason: "cached Huffman payload accounting underflowed during replacement",
                    })?;
            slot
        } else if let Some(slot) = empty {
            slot
        } else {
            let (slot, following) = self.next_huffman_fifo_victim(&victims, next_cursor)?;
            next_cursor = following;
            victims[slot] = true;
            victim_count += 1;
            let bytes = self.huffman_tables[slot]
                .as_ref()
                .map_or(0, |cached| cached.table.retained_allocation_bytes());
            evicted_bytes = evicted_bytes.saturating_add(bytes);
            retained_bytes =
                retained_bytes
                    .checked_sub(bytes)
                    .ok_or(JpegError::InternalInvariant {
                        reason: "cached Huffman payload accounting underflowed during eviction",
                    })?;
            slot
        };
        while retained_bytes.saturating_add(table_bytes) > retained_payload_cap {
            let (victim, following) = self.next_huffman_fifo_victim(&victims, next_cursor)?;
            next_cursor = following;
            victims[victim] = true;
            victim_count += 1;
            let bytes = self.huffman_tables[victim]
                .as_ref()
                .map_or(0, |cached| cached.table.retained_allocation_bytes());
            evicted_bytes = evicted_bytes.saturating_add(bytes);
            retained_bytes = retained_bytes.checked_sub(bytes).ok_or(
                JpegError::InternalInvariant {
                    reason:
                        "cached Huffman payload accounting underflowed during byte-cap eviction",
                },
            )?;
        }
        let committed_payload_bytes =
            retained_bytes
                .checked_add(table_bytes)
                .ok_or(JpegError::InternalInvariant {
                    reason: "cached Huffman payload accounting overflowed during insertion",
                })?;
        projected_live_bytes = projected_live_bytes.checked_sub(evicted_bytes).ok_or(
            JpegError::InternalInvariant {
                reason: "Huffman cache eviction allocation accounting underflowed",
            },
        )?;
        // The selected slot is refilled below. Every other selected victim
        // remains vacant and invalidates early termination for later probes.
        let mut left_tombstone = false;
        for (victim, selected) in victims.into_iter().enumerate() {
            if selected {
                self.huffman_tables[victim] = None;
                left_tombstone |= victim != slot;
            }
        }
        self.huffman_tables[slot] = Some(CachedHuffmanTable { digest, table });
        self.huffman_cache_has_tombstones |= left_tombstone;
        self.huffman_payload_bytes = committed_payload_bytes;
        debug_assert_eq!(
            self.huffman_payload_bytes,
            self.huffman_tables
                .iter()
                .flatten()
                .map(|cached| cached.table.retained_allocation_bytes())
                .sum::<usize>(),
        );
        self.huffman_eviction_cursor = next_cursor;
        *live_bytes = projected_live_bytes;
        self.cache_misses = self.cache_misses.saturating_add(1);
        self.cache_evictions = self.cache_evictions.saturating_add(victim_count);
        self.huffman_tables[slot]
            .as_ref()
            .map(|cached| &cached.table)
            .ok_or(JpegError::InternalInvariant {
                reason: "Huffman cache slot emptied while resolving a table",
            })
    }

    fn next_huffman_fifo_victim(
        &self,
        excluded: &[bool; HUFFMAN_CACHE_SLOTS],
        cursor: usize,
    ) -> Result<(usize, usize), JpegError> {
        for probe in 0..self.huffman_tables.len() {
            let slot = (cursor + probe) % self.huffman_tables.len();
            if self.huffman_tables[slot].is_some() && !excluded[slot] {
                return Ok((slot, (slot + 1) % self.huffman_tables.len()));
            }
        }
        Err(JpegError::InternalInvariant {
            reason: "Huffman cache byte cap could not select a FIFO victim",
        })
    }

    fn ensure_huffman_cache_slots(
        &mut self,
        live_bytes: &mut usize,
        cap: usize,
    ) -> Result<(), JpegError> {
        if self.huffman_tables.len() == HUFFMAN_CACHE_SLOTS {
            return Ok(());
        }
        if !self.huffman_tables.is_empty() || self.huffman_payload_bytes != 0 {
            return Err(JpegError::InternalInvariant {
                reason: "Huffman cache slots can only be allocated for an empty cache",
            });
        }
        let mut slots = Vec::new();
        let mut projected_live_bytes = *live_bytes;
        try_reserve_for_len_with_live_budget(
            &mut slots,
            HUFFMAN_CACHE_SLOTS,
            &mut projected_live_bytes,
            cap,
        )?;
        let slot_bytes = slots
            .capacity()
            .checked_mul(size_of::<Option<CachedHuffmanTable>>())
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap,
            })?;
        if slot_bytes
            .checked_add(HuffmanTable::maximum_retained_allocation_bytes())
            .is_none_or(|peak| peak > TABLE_CACHE_ALLOCATION_RESERVE_BYTES)
        {
            return Err(JpegError::MemoryCapExceeded {
                requested: slot_bytes
                    .saturating_add(HuffmanTable::maximum_retained_allocation_bytes()),
                cap: TABLE_CACHE_ALLOCATION_RESERVE_BYTES,
            });
        }
        slots.resize_with(HUFFMAN_CACHE_SLOTS, || None);
        self.huffman_tables = slots;
        *live_bytes = projected_live_bytes;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn resolve_decode_plan<F>(
        &mut self,
        header_prefix: &[u8],
        retained_external_bytes: usize,
        build: F,
    ) -> Result<PreparedDecodePlan, JpegError>
    where
        F: FnOnce(&mut Self) -> Result<PreparedDecodePlan, JpegError>,
    {
        let digest = digest_bytes(header_prefix);
        self.resolve_decode_plan_with_digest(
            header_prefix,
            digest,
            None,
            retained_external_bytes,
            build,
        )
    }

    pub(crate) fn resolve_decode_plan_with_template<F>(
        &mut self,
        header_prefix: &[u8],
        template: (&Info, &[Warning], usize, usize),
        retained_external_bytes: usize,
        build: F,
    ) -> Result<PreparedDecodePlan, JpegError>
    where
        F: FnOnce(&mut Self) -> Result<PreparedDecodePlan, JpegError>,
    {
        let digest = digest_bytes(header_prefix);
        self.resolve_decode_plan_with_digest(
            header_prefix,
            digest,
            Some(template),
            retained_external_bytes,
            build,
        )
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the digest cast only selects a cache shard and intentionally uses the native word"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "cache lookup, fallible admission, and exact retained-byte accounting form one transaction"
    )]
    fn resolve_decode_plan_with_digest<F>(
        &mut self,
        header_prefix: &[u8],
        digest: u64,
        template: Option<(&Info, &[Warning], usize, usize)>,
        retained_external_bytes: usize,
        build: F,
    ) -> Result<PreparedDecodePlan, JpegError>
    where
        F: FnOnce(&mut Self) -> Result<PreparedDecodePlan, JpegError>,
    {
        let start = (digest as usize) % self.decode_plans.len();
        let retained_context_bytes = self.retained_allocation_bytes();
        let initial_live_bytes =
            checked_add_allocation_bytes(retained_external_bytes, retained_context_bytes)?;
        for probe in 0..self.decode_plans.len() {
            let slot = (start + probe) % self.decode_plans.len();
            match &self.decode_plans[slot] {
                Some(cached)
                    if cached.digest == digest
                        && cached.header_prefix.as_slice() == header_prefix =>
                {
                    self.cache_hits = self.cache_hits.saturating_add(1);
                    let mut live_bytes = initial_live_bytes;
                    let cloned = try_clone_decode_plan(&cached.plan, &mut live_bytes, None)?
                        .ok_or(JpegError::InternalInvariant {
                            reason: "cached decode plan unexpectedly bypassed cloning",
                        })?;
                    #[cfg(test)]
                    {
                        self.decode_plan_clones = self.decode_plan_clones.saturating_add(1);
                    }
                    return Ok(cloned);
                }
                Some(_) | None => {}
            }
        }
        let built = build(self)?;
        self.cache_misses = self.cache_misses.saturating_add(1);
        let predicted_template_bytes = template.map_or(Ok(0), |(_, warnings, _, _)| {
            sequential_template_bytes(warnings.len())
        })?;
        let predicted_bytes =
            decode_plan_entry_bytes(header_prefix.len(), &built, predicted_template_bytes)?;
        if predicted_bytes > MAX_DECODE_PLAN_CACHE_BYTES {
            // The cache is an optimization. A key that cannot fit its entire
            // exact entry under the aggregate cache budget is decoded but not
            // retained.
            return Ok(built);
        }
        self.evict_decode_plans_until_fits(start, predicted_bytes);
        let slot = self.first_empty_decode_plan_slot(start).unwrap_or_else(|| {
            self.evict_decode_plan_slot(start);
            start
        });
        let mut live_bytes = checked_add_allocation_bytes(
            retained_external_bytes,
            self.retained_allocation_bytes(),
        )?;
        live_bytes = checked_add_allocation_bytes(live_bytes, built.retained_allocation_bytes()?)?;
        let mut owned_prefix = Vec::new();
        try_reserve_for_len_with_live_budget(
            &mut owned_prefix,
            header_prefix.len(),
            &mut live_bytes,
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )?;
        owned_prefix.extend_from_slice(header_prefix);
        let prefix_bytes = owned_prefix.capacity();
        let Some(cached_plan) = try_clone_decode_plan(&built, &mut live_bytes, Some(prefix_bytes))?
        else {
            return Ok(built);
        };
        let cached_template = match template {
            Some((info, warnings, parsed_warning_capacity, parsed_bytes)) => {
                if parsed_warning_capacity < warnings.len() {
                    return Err(JpegError::InternalInvariant {
                        reason: "parsed warning capacity is smaller than its length",
                    });
                }
                let parsed_warning_bytes =
                    checked_allocation_bytes::<Warning>(parsed_warning_capacity)?;
                let parsed_nonwarning_bytes = parsed_bytes
                    .checked_sub(parsed_warning_bytes)
                    .ok_or(JpegError::InternalInvariant {
                        reason: "parsed warning capacity exceeds parsed header ownership",
                    })?;
                live_bytes = checked_add_allocation_bytes(
                    live_bytes,
                    size_of::<CachedSequentialTemplate>(),
                )?;
                Some(CachedSequentialTemplate {
                    info: info.clone(),
                    warnings: try_clone_warnings(warnings, &mut live_bytes)?,
                    parsed_bytes,
                    parsed_nonwarning_bytes,
                })
            }
            None => None,
        };
        #[cfg(test)]
        {
            self.decode_plan_clones = self.decode_plan_clones.saturating_add(1);
        }
        let template_bytes = cached_template.as_ref().map_or(Ok(0), |template| {
            sequential_template_bytes(template.warnings.capacity())
        })?;
        let allocation_bytes =
            decode_plan_entry_bytes(owned_prefix.capacity(), &cached_plan, template_bytes)?;
        if allocation_bytes > MAX_DECODE_PLAN_CACHE_BYTES {
            return Ok(built);
        }
        self.evict_decode_plans_until_fits(start, allocation_bytes);
        if self.decode_plans[slot].is_some() {
            return Err(JpegError::InternalInvariant {
                reason: "decode-plan cache selected an occupied insertion slot",
            });
        }
        let new_cache_bytes = self
            .decode_plan_cache_bytes
            .checked_add(allocation_bytes)
            .ok_or(JpegError::InternalInvariant {
                reason: "decode-plan cache byte accounting overflowed",
            })?;
        let huffman_bytes = self.retained_huffman_allocation_bytes();
        let retained_after_insert =
            new_cache_bytes
                .checked_add(huffman_bytes)
                .ok_or(JpegError::InternalInvariant {
                    reason: "decoder context byte accounting overflowed",
                })?;
        if retained_after_insert > MAX_DECODER_CONTEXT_ALLOCATION_BYTES {
            // The cache is optional. Allocator capacity rounding can make an
            // otherwise valid entry exceed the context-only retention cap;
            // decode with `built` and release the attempted cache copy.
            return Ok(built);
        }
        self.decode_plans[slot] = Some(CachedDecodePlan {
            digest,
            header_prefix: owned_prefix,
            plan: cached_plan,
            template: cached_template,
            allocation_bytes,
        });
        self.decode_plan_cache_bytes = new_cache_bytes;
        Ok(built)
    }

    fn first_empty_decode_plan_slot(&self, start: usize) -> Option<usize> {
        (0..self.decode_plans.len())
            .map(|probe| (start + probe) % self.decode_plans.len())
            .find(|&slot| self.decode_plans[slot].is_none())
    }

    fn evict_decode_plans_until_fits(&mut self, start: usize, incoming_bytes: usize) {
        for probe in 0..self.decode_plans.len() {
            if self.decode_plan_cache_bytes.saturating_add(incoming_bytes)
                <= MAX_DECODE_PLAN_CACHE_BYTES
            {
                break;
            }
            let slot = (start + probe) % self.decode_plans.len();
            self.evict_decode_plan_slot(slot);
        }
    }

    fn evict_decode_plan_slot(&mut self, slot: usize) {
        if let Some(cached) = self.decode_plans[slot].take() {
            self.decode_plan_cache_bytes = self
                .decode_plan_cache_bytes
                .saturating_sub(cached.allocation_bytes);
            self.cache_evictions = self.cache_evictions.saturating_add(1);
        }
    }

    pub(crate) fn retained_allocation_bytes(&self) -> usize {
        self.decode_plan_cache_bytes
            .saturating_add(self.retained_huffman_allocation_bytes())
    }

    fn retained_huffman_allocation_bytes(&self) -> usize {
        self.huffman_tables
            .capacity()
            .saturating_mul(size_of::<Option<CachedHuffmanTable>>())
            .saturating_add(self.huffman_payload_bytes)
    }

    fn huffman_slot_allocation_bytes(&self) -> Result<usize, JpegError> {
        self.huffman_tables
            .capacity()
            .checked_mul(size_of::<Option<CachedHuffmanTable>>())
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap: TABLE_CACHE_ALLOCATION_RESERVE_BYTES,
            })
    }

    #[cfg(test)]
    pub(crate) const fn decode_plan_clone_count(&self) -> u64 {
        self.decode_plan_clones
    }

    fn occupied_cache_slots(&self) -> u64 {
        let occupied = self
            .quant_tables
            .iter()
            .filter(|slot| slot.is_some())
            .count()
            + self
                .huffman_tables
                .iter()
                .filter(|slot| slot.is_some())
                .count()
            + self
                .decode_plans
                .iter()
                .filter(|slot| slot.is_some())
                .count();
        occupied as u64
    }
}

#[doc(hidden)]
impl CodecContext for DecoderContext {
    fn clear(&mut self) {
        *self = Self::new();
    }

    fn cache_stats(&self) -> CacheStats {
        CacheStats::with_slots(
            self.cache_hits,
            self.cache_misses,
            self.occupied_cache_slots(),
            self.cache_evictions,
        )
    }
}

fn digest_bytes(bytes: &[u8]) -> u64 {
    j2k_core::__j2k_fnv1a64_bytes!(bytes)
}

fn digest_quant_table(table: &[u16; 64]) -> u64 {
    let mut hash = j2k_core::__j2k_fnv1a64_init!();
    for &entry in table {
        for byte in entry.to_le_bytes() {
            j2k_core::__j2k_fnv1a64_update!(hash, byte);
        }
    }
    hash
}

fn digest_huffman_table(raw: &RawHuffmanTable, role: HuffmanTableRole) -> u64 {
    let mut hash = digest_bytes(&raw.bits);
    for &byte in raw.values.as_slice() {
        j2k_core::__j2k_fnv1a64_update!(hash, byte);
    }
    j2k_core::__j2k_fnv1a64_update!(
        hash,
        match role {
            HuffmanTableRole::Dc => 0u8,
            HuffmanTableRole::Ac => 1u8,
        }
    );
    hash
}

fn decode_plan_entry_bytes(
    header_prefix_len: usize,
    plan: &PreparedDecodePlan,
    template_bytes: usize,
) -> Result<usize, JpegError> {
    header_prefix_len
        .checked_add(plan.retained_allocation_bytes()?)
        .and_then(|bytes| bytes.checked_add(template_bytes))
        .ok_or_else(context_cap_error)
}

fn sequential_template_bytes(warning_capacity: usize) -> Result<usize, JpegError> {
    let warning_bytes = checked_allocation_bytes::<Warning>(warning_capacity)?;
    size_of::<CachedSequentialTemplate>()
        .checked_add(warning_bytes)
        .ok_or_else(context_cap_error)
}

fn ensure_cached_template_construction_fits(
    parsed_bytes: usize,
    plan: &PreparedDecodePlan,
) -> Result<(), JpegError> {
    let metadata_bytes =
        checked_add_allocation_bytes(parsed_bytes, plan.retained_allocation_bytes()?)?;
    checked_add_allocation_bytes(metadata_bytes, MAX_DECODER_CONTEXT_ALLOCATION_BYTES)?;
    Ok(())
}

fn context_cap_error() -> JpegError {
    JpegError::MemoryCapExceeded {
        requested: usize::MAX,
        cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    }
}

fn try_clone_decode_plan(
    plan: &PreparedDecodePlan,
    live_bytes: &mut usize,
    cache_prefix_bytes: Option<usize>,
) -> Result<Option<PreparedDecodePlan>, JpegError> {
    let mut components = Vec::new();
    try_reserve_for_len_with_live_budget(
        &mut components,
        plan.components.len(),
        live_bytes,
        j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    )?;
    components.extend(plan.components.iter().cloned());
    if let Some(prefix_bytes) = cache_prefix_bytes {
        let component_bytes =
            checked_allocation_bytes::<PreparedComponentPlan>(components.capacity())?;
        let huffman_bytes = plan.huffman_tables.retained_allocation_bytes()?;
        let projected = prefix_bytes
            .checked_add(component_bytes)
            .and_then(|bytes| bytes.checked_add(huffman_bytes))
            .ok_or_else(context_cap_error)?;
        if projected > MAX_DECODE_PLAN_CACHE_BYTES {
            return Ok(None);
        }
    }
    let huffman_tables = plan
        .huffman_tables
        .try_clone_with_live_budget(live_bytes, j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES)?;
    Ok(Some(PreparedDecodePlan {
        components,
        huffman_tables,
        huffman_primary_width: plan.huffman_primary_width,
        sampling: plan.sampling,
        color_space: plan.color_space,
        restart_interval: plan.restart_interval,
        dimensions: plan.dimensions,
        scan_offset: plan.scan_offset,
        scratch_bytes: plan.scratch_bytes,
    }))
}

fn try_clone_warnings(
    warnings: &[Warning],
    live_bytes: &mut usize,
) -> Result<Vec<Warning>, JpegError> {
    let mut cloned = Vec::new();
    try_reserve_for_len_with_live_budget(
        &mut cloned,
        warnings.len(),
        live_bytes,
        j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    )?;
    cloned.extend_from_slice(warnings);
    Ok(cloned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::sequential::PreparedComponentPlan;
    use crate::info::{ColorSpace, McuGeometry, SamplingFactors, SofKind};
    use alloc::vec;

    fn empty_plan(scan_offset: usize) -> PreparedDecodePlan {
        PreparedDecodePlan {
            components: vec![],
            huffman_tables: crate::entropy::huffman::PreparedHuffmanTables::try_with_capacity(0)
                .expect("empty arena"),
            huffman_primary_width: crate::entropy::sequential::HuffmanPrimaryWidth::Full12,
            sampling: SamplingFactors::from_validated_components(&[(1, 1)]),
            color_space: ColorSpace::Grayscale,
            restart_interval: None,
            dimensions: (16, 16),
            scan_offset,
            scratch_bytes: 0,
        }
    }

    fn sequential_info() -> Info {
        let sampling = SamplingFactors::from_validated_components(&[(1, 1)]);
        Info {
            dimensions: (16, 16),
            color_space: ColorSpace::Grayscale,
            sampling,
            sof_kind: SofKind::Baseline8,
            bit_depth: 8,
            restart_interval: None,
            mcu_geometry: McuGeometry::from_sampling((16, 16), sampling),
            scan_count: 1,
        }
    }

    fn cache_sequential_template(ctx: &mut DecoderContext, prefix: &[u8], warnings: &[Warning]) {
        let warnings = warnings.to_vec();
        let parsed_bytes =
            checked_allocation_bytes::<Warning>(warnings.capacity()).expect("warning bytes");
        ctx.resolve_decode_plan_with_template(
            prefix,
            (
                &sequential_info(),
                &warnings,
                warnings.capacity(),
                parsed_bytes,
            ),
            0,
            |_| Ok(empty_plan(prefix.len())),
        )
        .expect("cache sequential template");
    }

    #[test]
    fn sequential_template_requires_an_exact_header_and_one_current_scan() {
        let prefix = b"validated sequential header";
        let mut ctx = DecoderContext::new();
        cache_sequential_template(&mut ctx, prefix, &[]);
        let before = ctx.cache_stats();
        let mut different_payload = prefix.to_vec();
        different_payload.extend_from_slice(b"different entropy payload");

        let (info, warnings, plan) = ctx
            .clone_cached_sequential_template(&different_payload, 0)
            .expect("bounded clone")
            .expect("exact header template");
        assert_eq!(info, sequential_info());
        assert_eq!(warnings, []);
        assert_eq!(plan.scan_offset, prefix.len());
        assert_eq!(ctx.cache_stats().hits, before.hits + 1);

        assert!(ctx
            .clone_cached_sequential_template(&prefix[..prefix.len() - 1], 0)
            .expect("truncated input falls through")
            .is_none());
        assert!(ctx
            .clone_cached_sequential_template(prefix, 0)
            .expect("an empty entropy tail is still a header match")
            .is_some());

        let mut extra_scan = prefix.to_vec();
        extra_scan.extend_from_slice(&[0xff, 0xda, 0x00, 0x02]);
        assert!(matches!(
            ctx.clone_cached_sequential_template(&extra_scan, 0),
            Err(JpegError::InvalidSequentialScanCount {
                sof: SofKind::Baseline8,
                count: 2,
            })
        ));
    }

    #[test]
    fn sequential_template_warnings_stay_charged_while_plan_is_leased() {
        let prefix = b"validated sequential header";
        let warnings = [Warning::UnknownAppMarker {
            marker: 0xe3,
            size: 2,
        }];
        let mut ctx = DecoderContext::new();
        cache_sequential_template(&mut ctx, prefix, &warnings);
        let retained = ctx.retained_allocation_bytes();
        let (slot, _) = ctx.cached_decode_plan(prefix).expect("cached plan");
        let cached = ctx.decode_plans[slot].as_ref().expect("occupied slot");
        let template = cached.template.as_ref().expect("cached template");
        let expected = decode_plan_entry_bytes(
            cached.header_prefix.capacity(),
            &cached.plan,
            sequential_template_bytes(template.warnings.capacity()).unwrap(),
        )
        .unwrap();
        assert_eq!(cached.allocation_bytes, expected);
        assert_eq!(retained, expected);

        let (lease, plan) = ctx.lease_cached_decode_plan(slot).expect("leased plan");
        assert_eq!(*lease.decode_plan_cache_bytes, retained);
        assert_eq!(
            lease.template.as_ref().expect("leased template").warnings,
            warnings
        );
        lease.restore(plan);
        assert_eq!(ctx.retained_allocation_bytes(), retained);
    }

    #[test]
    fn sequential_template_warning_clone_obeys_the_shared_live_budget() {
        let prefix = b"validated sequential header";
        let warnings = [Warning::UnknownAppMarker {
            marker: 0xe3,
            size: 2,
        }];
        let mut ctx = DecoderContext::new();
        cache_sequential_template(&mut ctx, prefix, &warnings);
        let mut input = prefix.to_vec();
        input.push(0);
        let (_, cloned_warnings, cloned_plan) = ctx
            .clone_cached_sequential_template(&input, 0)
            .expect("measure warning clone")
            .expect("cached template");
        let clone_bytes = checked_allocation_bytes::<Warning>(cloned_warnings.capacity()).unwrap();
        let cloned_plan_bytes = cloned_plan.retained_allocation_bytes().unwrap();
        drop(cloned_warnings);
        drop(cloned_plan);
        let parsed_bytes = ctx
            .decode_plans
            .iter()
            .flatten()
            .find_map(|cached| cached.template.as_ref())
            .expect("cached template")
            .parsed_bytes;
        let exact_external = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES
            .checked_sub(ctx.retained_allocation_bytes())
            .and_then(|remaining| {
                remaining.checked_sub(parsed_bytes.max(cloned_plan_bytes.checked_add(clone_bytes)?))
            })
            .expect("fixture leaves an external budget");

        ctx.clone_cached_sequential_template(&input, exact_external)
            .expect("exact warning clone boundary")
            .expect("cached template");
        assert!(matches!(
            ctx.clone_cached_sequential_template(&input, exact_external + 1),
            Err(JpegError::MemoryCapExceeded { .. })
        ));
    }

    fn resolve_huffman_table(
        ctx: &mut DecoderContext,
        raw: &RawHuffmanTable,
    ) -> Result<HuffmanTable, JpegError> {
        let mut live_bytes = ctx.retained_allocation_bytes();
        ctx.resolve_huffman_table_with_live_budget(
            raw,
            HuffmanTableRole::Dc,
            &mut live_bytes,
            MAX_DECODER_CONTEXT_ALLOCATION_BYTES,
        )
        .cloned()
    }

    fn distinct_dc_raw(index: u8) -> RawHuffmanTable {
        let mut bits = [0; 16];
        bits[usize::from(index / 12)] = 1;
        RawHuffmanTable {
            bits,
            values: crate::parse::tables::HuffmanValues::from_slice(&[index % 12]),
        }
    }

    fn resolve_huffman_table_with_digest(
        ctx: &mut DecoderContext,
        raw: &RawHuffmanTable,
        digest: u64,
    ) -> Result<HuffmanTable, JpegError> {
        let mut live_bytes = ctx.retained_allocation_bytes();
        ctx.resolve_huffman_table_with_digest_and_live_budget(
            raw,
            HuffmanTableRole::Dc,
            digest,
            &mut live_bytes,
            MAX_DECODER_CONTEXT_ALLOCATION_BYTES,
        )
        .cloned()
    }

    #[test]
    fn quant_table_cache_hits_return_same_value() {
        let mut ctx = DecoderContext::new();
        let first = ctx.resolve_quant_table([7; 64]);
        let second = ctx.resolve_quant_table([7; 64]);
        assert_eq!(first, second);

        let stats = ctx.cache_stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.occupied_slots, 1);
        assert_eq!(stats.evictions, 0);
    }

    #[test]
    fn huffman_table_cache_hits_return_same_value() {
        let raw = RawHuffmanTable {
            bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: crate::parse::tables::HuffmanValues::from_slice(&[0]),
        };
        let mut ctx = DecoderContext::new();
        let first = resolve_huffman_table(&mut ctx, &raw).unwrap();
        let second = resolve_huffman_table(&mut ctx, &raw).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn fused_resolve_transactionally_upgrades_matching_raw_only_entry() {
        let raw = distinct_dc_raw(17);
        let mut ctx = DecoderContext::new();
        let mut live_bytes = 0;
        {
            let raw_only = ctx
                .resolve_huffman_table_with_requirement_and_live_budget(
                    &raw,
                    HuffmanTableRole::Dc,
                    HuffmanLookupRequirement::Raw,
                    &mut live_bytes,
                    j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                )
                .unwrap();
            assert!(!raw_only.supports_fused_lookup());
        }
        assert_eq!(ctx.huffman_tables.iter().flatten().count(), 1);
        let raw_context_bytes = ctx.retained_allocation_bytes();
        assert_eq!(live_bytes, raw_context_bytes);

        {
            let fused = ctx
                .resolve_huffman_table_with_live_budget(
                    &raw,
                    HuffmanTableRole::Dc,
                    &mut live_bytes,
                    j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                )
                .unwrap();
            assert!(fused.supports_fused_lookup());
        }
        assert_eq!(ctx.huffman_tables.iter().flatten().count(), 1);
        assert!(ctx.retained_allocation_bytes() > raw_context_bytes);
        assert_eq!(live_bytes, ctx.retained_allocation_bytes());

        let misses = ctx.cache_misses;
        let _ = ctx
            .resolve_huffman_table_with_requirement_and_live_budget(
                &raw,
                HuffmanTableRole::Dc,
                HuffmanLookupRequirement::Raw,
                &mut live_bytes,
                j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            )
            .unwrap();
        assert_eq!(ctx.cache_misses, misses);
    }

    #[test]
    fn full_resolve_transactionally_upgrades_matching_narrow_entry() {
        let raw = distinct_dc_raw(17);
        let mut ctx = DecoderContext::new();
        let mut live_bytes = 0;
        let narrow_payload_bytes = ctx
            .resolve_huffman_table_with_requirement_and_live_budget(
                &raw,
                HuffmanTableRole::Ac,
                HuffmanLookupRequirement::FusedNarrow,
                &mut live_bytes,
                MAX_DECODER_CONTEXT_ALLOCATION_BYTES,
            )
            .unwrap()
            .retained_allocation_bytes();
        let narrow_bytes = ctx.retained_allocation_bytes();
        let stats_before = ctx.cache_stats();
        let mut full_live_bytes = 0;
        let full = HuffmanTable::from_raw_with_requirement_and_live_budget(
            &raw,
            HuffmanTableRole::Ac,
            HuffmanLookupRequirement::FusedFull,
            &mut full_live_bytes,
            MAX_DECODER_CONTEXT_ALLOCATION_BYTES,
        )
        .unwrap();
        let full_bytes = full.retained_allocation_bytes();
        drop(full);

        let peak = live_bytes + full_bytes;
        let error = ctx
            .resolve_huffman_table_with_requirement_and_live_budget(
                &raw,
                HuffmanTableRole::Ac,
                HuffmanLookupRequirement::FusedFull,
                &mut live_bytes,
                peak - 1,
            )
            .expect_err("upgrade must retain the narrow payload until full construction succeeds");
        assert!(matches!(error, JpegError::MemoryCapExceeded { .. }));
        assert_eq!(live_bytes, narrow_bytes);
        assert_eq!(ctx.retained_allocation_bytes(), narrow_bytes);
        assert_eq!(ctx.cache_stats(), stats_before);

        let upgraded = ctx
            .resolve_huffman_table_with_requirement_and_live_budget(
                &raw,
                HuffmanTableRole::Ac,
                HuffmanLookupRequirement::FusedFull,
                &mut live_bytes,
                peak,
            )
            .unwrap();
        assert!(upgraded.satisfies_lookup_requirement(HuffmanLookupRequirement::FusedFull));
        assert_eq!(ctx.huffman_tables.iter().flatten().count(), 1);
        assert_eq!(live_bytes, narrow_bytes - narrow_payload_bytes + full_bytes);
        assert_eq!(live_bytes, ctx.retained_allocation_bytes());
        let stats = ctx.cache_stats();
        assert_eq!(stats.misses, stats_before.misses + 1);
        assert_eq!(stats.evictions, stats_before.evictions + 1);

        let reused = ctx
            .resolve_huffman_table_with_requirement_and_live_budget(
                &raw,
                HuffmanTableRole::Ac,
                HuffmanLookupRequirement::FusedNarrow,
                &mut live_bytes,
                peak,
            )
            .expect("a full entry satisfies later narrow requests");
        reused
            .ac_with_entries::<{ crate::entropy::huffman::NARROW_PRIMARY_ENTRIES }>()
            .expect("the retained full entry exposes its primary8 companion");
        assert_eq!(ctx.cache_stats().hits, stats.hits + 1);
    }

    #[test]
    fn quant_table_digest_collision_compares_full_table_contents() {
        let mut ctx = DecoderContext::new();
        let first = ctx.resolve_quant_table_with_digest([7; 64], 0);
        let second = ctx.resolve_quant_table_with_digest([8; 64], 0);

        assert_ne!(first, second);
        assert_eq!(first, [7; 64]);
        assert_eq!(second, [8; 64]);
        assert_eq!(ctx.cache_stats().misses, 2);
    }

    #[test]
    fn huffman_table_digest_collision_compares_code_lengths_and_values() {
        let first_raw = RawHuffmanTable {
            bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: crate::parse::tables::HuffmanValues::from_slice(&[0]),
        };
        let second_raw = RawHuffmanTable {
            bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: crate::parse::tables::HuffmanValues::from_slice(&[1]),
        };
        let third_raw = RawHuffmanTable {
            bits: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: crate::parse::tables::HuffmanValues::from_slice(&[0]),
        };
        let mut ctx = DecoderContext::new();

        let first = resolve_huffman_table_with_digest(&mut ctx, &first_raw, 0).unwrap();
        let second = resolve_huffman_table_with_digest(&mut ctx, &second_raw, 0).unwrap();
        let third = resolve_huffman_table_with_digest(&mut ctx, &third_raw, 0).unwrap();

        assert_ne!(first, second);
        assert_ne!(first, third);
        assert_eq!(ctx.cache_stats().misses, 3);
    }

    #[test]
    fn prepared_plan_cache_hits_skip_rebuild() {
        let mut ctx = DecoderContext::new();
        let prefix = [0xFF, 0xD8, 0xFF, 0xDA];
        let mut builds = 0usize;

        let first = ctx
            .resolve_decode_plan(&prefix, 0, |_| {
                builds += 1;
                Ok(empty_plan(42))
            })
            .unwrap();

        let second = ctx
            .resolve_decode_plan(&prefix, 0, |_| {
                builds += 1;
                unreachable!("cache hit should bypass rebuild")
            })
            .unwrap();

        assert_eq!(builds, 1);
        assert_eq!(first.scan_offset, second.scan_offset);
    }

    #[test]
    fn cache_hit_clone_shares_one_exact_external_live_budget() {
        let raw = RawHuffmanTable {
            bits: [0; 16],
            values: crate::parse::tables::HuffmanValues::default(),
        };
        let mut huffman_tables =
            crate::entropy::huffman::PreparedHuffmanTables::try_with_capacity(1)
                .expect("bounded arena");
        let table = huffman_tables
            .push(HuffmanTable::from_raw(&raw, HuffmanTableRole::Dc).expect("empty table"))
            .expect("reserved arena");
        let mut plan = empty_plan(7);
        plan.huffman_tables = huffman_tables;
        plan.components.push(PreparedComponentPlan {
            h: 1,
            v: 1,
            output_index: 0,
            quant: [1; 64],
            dc_table: Some(table),
            ac_table: Some(table),
        });

        let mut ctx = DecoderContext::new();
        let prefix = [0xFF, 0xD8, 0xFF, 0xDA];
        ctx.resolve_decode_plan(&prefix, 0, |_| Ok(plan))
            .expect("initial cache insertion");
        let cached_plan_bytes = ctx
            .decode_plans
            .iter()
            .flatten()
            .next()
            .expect("cached plan")
            .plan
            .retained_allocation_bytes()
            .expect("cached plan bytes");
        let exact_external = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES
            .checked_sub(ctx.retained_allocation_bytes())
            .and_then(|remaining| remaining.checked_sub(cached_plan_bytes))
            .expect("fixture leaves an external budget");

        ctx.resolve_decode_plan(&prefix, exact_external, |_| {
            unreachable!("cache hit must bypass rebuild")
        })
        .expect("exact live boundary");
        assert!(matches!(
            ctx.resolve_decode_plan(&prefix, exact_external + 1, |_| {
                unreachable!("cache hit must bypass rebuild")
            }),
            Err(JpegError::MemoryCapExceeded { .. })
        ));
    }

    #[test]
    fn prepared_plan_digest_collision_compares_full_header_prefix() {
        let mut ctx = DecoderContext::new();
        let first = ctx
            .resolve_decode_plan_with_digest(b"first", 0, None, 0, |_| Ok(empty_plan(1)))
            .unwrap();
        let second = ctx
            .resolve_decode_plan_with_digest(b"second", 0, None, 0, |_| Ok(empty_plan(2)))
            .unwrap();
        let first_hit = ctx
            .resolve_decode_plan_with_digest(b"first", 0, None, 0, |_| {
                unreachable!("full-key cache hit must bypass rebuild")
            })
            .unwrap();

        assert_eq!(first.scan_offset, 1);
        assert_eq!(second.scan_offset, 2);
        assert_eq!(first_hit.scan_offset, 1);
        assert_eq!(ctx.cache_stats().hits, 1);
    }

    #[test]
    fn prepared_plan_lease_compares_full_header_prefix_after_digest_collision() {
        let requested = b"requested";
        let digest = digest_bytes(requested);
        let slot_count = u64::try_from(PLAN_CACHE_SLOTS).unwrap();
        let start = usize::try_from(digest % slot_count).unwrap();
        let wrong_slot = start;
        let matching_slot = (start + 1) % PLAN_CACHE_SLOTS;
        let wrong_prefix = b"collision".to_vec();
        let matching_prefix = requested.to_vec();
        let wrong_plan = empty_plan(11);
        let matching_plan = empty_plan(29);
        let wrong_bytes = decode_plan_entry_bytes(wrong_prefix.capacity(), &wrong_plan, 0).unwrap();
        let matching_bytes =
            decode_plan_entry_bytes(matching_prefix.capacity(), &matching_plan, 0).unwrap();
        let mut ctx = DecoderContext::new();
        ctx.decode_plans[wrong_slot] = Some(CachedDecodePlan {
            digest,
            header_prefix: wrong_prefix,
            plan: wrong_plan,
            template: None,
            allocation_bytes: wrong_bytes,
        });
        ctx.decode_plans[matching_slot] = Some(CachedDecodePlan {
            digest,
            header_prefix: matching_prefix,
            plan: matching_plan,
            template: None,
            allocation_bytes: matching_bytes,
        });
        ctx.decode_plan_cache_bytes = wrong_bytes + matching_bytes;
        let retained = ctx.retained_allocation_bytes();
        let occupied = ctx.cache_stats().occupied_slots;

        assert!(ctx.cached_decode_plan(b"absent").is_none());
        let (slot, _) = ctx
            .cached_decode_plan(requested)
            .expect("matching full prefix must resolve");
        let (lease, plan) = ctx
            .lease_cached_decode_plan(slot)
            .expect("matching full prefix must lease");
        assert_eq!(plan.scan_offset, 29);
        lease.restore(plan);

        assert_eq!(
            ctx.cached_decode_plan(requested)
                .expect("leased plan restored under exact key")
                .1
                .scan_offset,
            29
        );
        assert_eq!(ctx.retained_allocation_bytes(), retained);
        assert_eq!(ctx.cache_stats().occupied_slots, occupied);
    }

    #[test]
    fn prepared_plan_cache_full_eviction_is_deterministic() {
        let mut ctx = DecoderContext::new();
        let cache_slots = u8::try_from(PLAN_CACHE_SLOTS).expect("plan cache slot count fits u8");
        for key in 0..cache_slots {
            ctx.resolve_decode_plan_with_digest(&[key], 0, None, 0, |_| {
                Ok(empty_plan(usize::from(key)))
            })
            .unwrap();
        }
        ctx.resolve_decode_plan_with_digest(&[cache_slots], 0, None, 0, |_| {
            Ok(empty_plan(PLAN_CACHE_SLOTS))
        })
        .unwrap();

        assert_eq!(ctx.cache_stats().evictions, 1);
        let mut rebuilt = false;
        let first = ctx
            .resolve_decode_plan_with_digest(&[0], 0, None, 0, |_| {
                rebuilt = true;
                Ok(empty_plan(99))
            })
            .unwrap();
        assert!(rebuilt, "the start slot must be the deterministic victim");
        assert_eq!(first.scan_offset, 99);
    }

    #[test]
    fn decode_plan_cache_entry_boundary_bypasses_oversized_keys() {
        let plan = empty_plan(0);
        let max_key_bytes = MAX_DECODE_PLAN_CACHE_BYTES - plan.retained_allocation_bytes().unwrap();
        assert_eq!(
            decode_plan_entry_bytes(max_key_bytes, &plan, 0).unwrap(),
            MAX_DECODE_PLAN_CACHE_BYTES
        );
        assert!(
            decode_plan_entry_bytes(max_key_bytes + 1, &plan, 0).unwrap()
                > MAX_DECODE_PLAN_CACHE_BYTES
        );
    }

    #[test]
    fn decode_plan_cache_entry_counts_tables_retained_after_table_cache_eviction() {
        let raw = RawHuffmanTable {
            bits: [0; 16],
            values: crate::parse::tables::HuffmanValues::default(),
        };
        let mut huffman_tables =
            crate::entropy::huffman::PreparedHuffmanTables::try_with_capacity(1)
                .expect("bounded arena");
        let table = huffman_tables
            .push(HuffmanTable::from_raw(&raw, HuffmanTableRole::Dc).expect("empty table"))
            .expect("reserved arena");
        let mut plan = empty_plan(0);
        plan.huffman_tables = huffman_tables;
        plan.components.push(PreparedComponentPlan {
            h: 1,
            v: 1,
            output_index: 0,
            quant: [1; 64],
            dc_table: Some(table),
            ac_table: Some(table),
        });

        let entry_bytes = decode_plan_entry_bytes(0, &plan, 0).expect("bounded plan");
        assert_eq!(entry_bytes, plan.retained_allocation_bytes().unwrap());
        let logical_bytes = PreparedDecodePlan::allocation_bytes_for_counts(
            plan.components.len(),
            plan.huffman_tables.len(),
        )
        .expect("logical plan bytes");
        let exact_without_component_spare = plan.components.len()
            * size_of::<PreparedComponentPlan>()
            + plan.huffman_tables.retained_allocation_bytes().unwrap();
        // `push` from an empty vector commonly retains spare component slots.
        // Cache entry accounting must use that allocator-returned capacity,
        // not the one logical component requested by this fixture.
        let component_spare_bytes = (plan.components.capacity() - plan.components.len())
            * size_of::<PreparedComponentPlan>();
        assert_eq!(
            entry_bytes - exact_without_component_spare,
            component_spare_bytes
        );
        assert!(logical_bytes >= entry_bytes);
        assert!(entry_bytes > size_of::<PreparedComponentPlan>());
    }

    #[test]
    fn oversized_decode_plan_key_is_not_retained() {
        let prefix = vec![0u8; MAX_DECODE_PLAN_CACHE_BYTES + 1];
        let mut ctx = DecoderContext::new();
        let mut builds = 0usize;
        for _ in 0..2 {
            ctx.resolve_decode_plan(&prefix, 0, |_| {
                builds += 1;
                Ok(empty_plan(builds))
            })
            .unwrap();
        }

        assert_eq!(builds, 2, "oversized keys must bypass the cache");
        assert_eq!(ctx.decode_plan_cache_bytes, 0);
        assert!(ctx.decode_plans.iter().all(Option::is_none));
    }

    #[test]
    fn identical_raw_huffman_tables_keep_roles_distinct_even_on_digest_collision() {
        let raw = RawHuffmanTable {
            bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: crate::parse::tables::HuffmanValues::from_slice(&[0]),
        };
        for force_collision in [false, true] {
            let mut ctx = DecoderContext::new();
            for role in [
                HuffmanTableRole::Dc,
                HuffmanTableRole::Ac,
                HuffmanTableRole::Dc,
                HuffmanTableRole::Ac,
            ] {
                let mut live_bytes = ctx.retained_allocation_bytes();
                let digest = if force_collision {
                    0
                } else {
                    digest_huffman_table(&raw, role)
                };
                let table = ctx
                    .resolve_huffman_table_with_digest_and_live_budget(
                        &raw,
                        role,
                        digest,
                        &mut live_bytes,
                        MAX_DECODER_CONTEXT_ALLOCATION_BYTES,
                    )
                    .expect("role-specific cached table");
                assert_eq!(table.dc().is_ok(), role == HuffmanTableRole::Dc);
                assert_eq!(table.ac().is_ok(), role == HuffmanTableRole::Ac);
                assert_eq!(live_bytes, ctx.retained_allocation_bytes());
            }
            let stats = ctx.cache_stats();
            assert_eq!(stats.misses, 2);
            assert_eq!(stats.hits, 2);
            assert_eq!(stats.occupied_slots, 2);
        }
    }

    #[test]
    fn role_specific_cache_slots_obey_exact_live_byte_boundary() {
        let raw = RawHuffmanTable {
            bits: [0; 16],
            values: crate::parse::tables::HuffmanValues::default(),
        };
        let external_bytes = 7;
        let slot_bytes = HUFFMAN_CACHE_SLOTS * size_of::<Option<CachedHuffmanTable>>();
        let dc_bytes = HuffmanTable::from_raw(&raw, HuffmanTableRole::Dc)
            .unwrap()
            .retained_allocation_bytes();
        let ac_bytes = HuffmanTable::from_raw(&raw, HuffmanTableRole::Ac)
            .unwrap()
            .retained_allocation_bytes();
        let exact_cap = external_bytes + slot_bytes + dc_bytes + ac_bytes;
        let mut ctx = DecoderContext::new();
        let mut live_bytes = external_bytes;
        let error = ctx
            .resolve_huffman_table_with_live_budget(
                &raw,
                HuffmanTableRole::Dc,
                &mut live_bytes,
                external_bytes + slot_bytes - 1,
            )
            .expect_err("one byte below the slot allocation");
        assert!(
            matches!(error, JpegError::MemoryCapExceeded { requested, cap }
            if requested == external_bytes + slot_bytes && cap == external_bytes + slot_bytes - 1)
        );
        assert_eq!(ctx.retained_allocation_bytes(), 0);
        assert_eq!(live_bytes, external_bytes);
        assert_eq!(ctx.cache_stats().occupied_slots, 0);

        for role in [HuffmanTableRole::Dc, HuffmanTableRole::Ac] {
            ctx.resolve_huffman_table_with_live_budget(&raw, role, &mut live_bytes, exact_cap)
                .expect("both role-specific payloads fit at the exact boundary");
        }
        assert_eq!(live_bytes, exact_cap);
        assert_eq!(
            ctx.retained_allocation_bytes(),
            slot_bytes + dc_bytes + ac_bytes
        );
        assert_eq!(ctx.cache_stats().occupied_slots, 2);
    }

    #[test]
    fn huffman_cache_replacement_counts_old_and_new_payloads_at_peak() {
        let mut ctx = DecoderContext::new();
        let mut live_bytes = 0;
        let cache_slots = u8::try_from(HUFFMAN_CACHE_SLOTS).expect("cache slot count fits u8");
        for value in 0..cache_slots {
            ctx.resolve_huffman_table_with_digest_and_live_budget(
                &distinct_dc_raw(value),
                HuffmanTableRole::Dc,
                0,
                &mut live_bytes,
                MAX_DECODER_CONTEXT_ALLOCATION_BYTES,
            )
            .unwrap();
        }
        let retained_before = ctx.retained_allocation_bytes();
        let stats_before = ctx.cache_stats();
        let cursor_before = ctx.huffman_eviction_cursor;
        let replacement = distinct_dc_raw(cache_slots);
        let replacement_bytes = HuffmanTable::from_raw(&replacement, HuffmanTableRole::Dc)
            .unwrap()
            .retained_allocation_bytes();
        let peak = retained_before + replacement_bytes;

        let error = ctx
            .resolve_huffman_table_with_digest_and_live_budget(
                &replacement,
                HuffmanTableRole::Dc,
                0,
                &mut live_bytes,
                peak - 1,
            )
            .expect_err("replacement must retain the old payload until the new one is built");
        assert!(matches!(error, JpegError::MemoryCapExceeded { .. }));
        assert_eq!(live_bytes, retained_before);
        assert_eq!(ctx.retained_allocation_bytes(), retained_before);
        assert_eq!(ctx.cache_stats(), stats_before);
        assert_eq!(ctx.huffman_eviction_cursor, cursor_before);

        ctx.resolve_huffman_table_with_digest_and_live_budget(
            &replacement,
            HuffmanTableRole::Dc,
            0,
            &mut live_bytes,
            peak,
        )
        .expect("exact old-plus-new peak must fit");
        assert_eq!(live_bytes, retained_before);
        assert_eq!(ctx.retained_allocation_bytes(), retained_before);
        assert_eq!(ctx.cache_stats().evictions, stats_before.evictions + 1);
    }

    #[test]
    fn huffman_eviction_stabilizes_a_colliding_twenty_table_working_set() {
        const WORKING_SET_TABLES: u8 = 20;

        let mut ctx = DecoderContext::new();
        let mut live_bytes = 0;
        let cache_slots = u8::try_from(HUFFMAN_CACHE_SLOTS).expect("cache slot count fits u8");
        for value in 0..cache_slots {
            ctx.resolve_huffman_table_with_digest_and_live_budget(
                &distinct_dc_raw(value),
                HuffmanTableRole::Dc,
                0,
                &mut live_bytes,
                MAX_DECODER_CONTEXT_ALLOCATION_BYTES,
            )
            .unwrap();
        }

        let working = (cache_slots..cache_slots + WORKING_SET_TABLES)
            .map(distinct_dc_raw)
            .collect::<Vec<_>>();
        let before = ctx.cache_stats();
        for raw in &working {
            ctx.resolve_huffman_table_with_digest_and_live_budget(
                raw,
                HuffmanTableRole::Dc,
                0,
                &mut live_bytes,
                MAX_DECODER_CONTEXT_ALLOCATION_BYTES,
            )
            .unwrap();
        }
        let warmed = ctx.cache_stats();
        assert_eq!(warmed.misses, before.misses + u64::from(WORKING_SET_TABLES));
        assert_eq!(
            warmed.evictions,
            before.evictions + u64::from(WORKING_SET_TABLES)
        );
        for raw in &working {
            ctx.resolve_huffman_table_with_digest_and_live_budget(
                raw,
                HuffmanTableRole::Dc,
                0,
                &mut live_bytes,
                MAX_DECODER_CONTEXT_ALLOCATION_BYTES,
            )
            .unwrap();
        }
        let hot = ctx.cache_stats();
        assert_eq!(hot.hits, warmed.hits + u64::from(WORKING_SET_TABLES));
        assert_eq!(hot.misses, warmed.misses);
        assert_eq!(hot.evictions, warmed.evictions);
    }

    #[test]
    fn huffman_cache_retains_two_dc_and_eight_ac_tables() {
        let raw_for = |value| RawHuffmanTable {
            bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: crate::parse::tables::HuffmanValues::from_slice(&[value]),
        };
        let entries = [
            (HuffmanTableRole::Dc, raw_for(0)),
            (HuffmanTableRole::Dc, raw_for(1)),
            (HuffmanTableRole::Ac, raw_for(0x00)),
            (HuffmanTableRole::Ac, raw_for(0x01)),
            (HuffmanTableRole::Ac, raw_for(0x02)),
            (HuffmanTableRole::Ac, raw_for(0x03)),
            (HuffmanTableRole::Ac, raw_for(0x04)),
            (HuffmanTableRole::Ac, raw_for(0x05)),
            (HuffmanTableRole::Ac, raw_for(0x06)),
            (HuffmanTableRole::Ac, raw_for(0x07)),
        ];
        let mut ctx = DecoderContext::new();
        let mut live_bytes = 0;
        for (role, raw) in &entries {
            ctx.resolve_huffman_table_with_live_budget(
                raw,
                *role,
                &mut live_bytes,
                MAX_DECODER_CONTEXT_ALLOCATION_BYTES,
            )
            .unwrap();
        }
        let warmed = ctx.cache_stats();
        assert_eq!(warmed.occupied_slots, 10);
        assert_eq!(warmed.evictions, 0);
        assert!(ctx.retained_huffman_allocation_bytes() <= TABLE_CACHE_ALLOCATION_RESERVE_BYTES);
        assert!(
            ctx.retained_huffman_allocation_bytes()
                + HuffmanTable::maximum_retained_allocation_bytes()
                <= TABLE_CACHE_ALLOCATION_RESERVE_BYTES
        );
        for (role, raw) in &entries {
            ctx.resolve_huffman_table_with_live_budget(
                raw,
                *role,
                &mut live_bytes,
                MAX_DECODER_CONTEXT_ALLOCATION_BYTES,
            )
            .unwrap();
        }
        let hot = ctx.cache_stats();
        assert_eq!(hot.hits, warmed.hits + 10);
        assert_eq!(hot.misses, warmed.misses);
        assert_eq!(hot.evictions, warmed.evictions);
    }

    #[test]
    fn huffman_byte_cap_evicts_pathological_ac_working_sets() {
        let raw_for = |value| RawHuffmanTable {
            bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: crate::parse::tables::HuffmanValues::from_slice(&[value]),
        };
        let raws = (0..12).map(raw_for).collect::<Vec<_>>();
        let mut ctx = DecoderContext::new();
        let mut live_bytes = 0;
        for raw in &raws {
            ctx.resolve_huffman_table_with_digest_and_live_budget(
                raw,
                HuffmanTableRole::Ac,
                0,
                &mut live_bytes,
                MAX_DECODER_CONTEXT_ALLOCATION_BYTES,
            )
            .unwrap();
            assert_eq!(live_bytes, ctx.retained_allocation_bytes());
            assert!(
                ctx.retained_huffman_allocation_bytes()
                    + HuffmanTable::maximum_retained_allocation_bytes()
                    <= TABLE_CACHE_ALLOCATION_RESERVE_BYTES
            );
        }
        assert!(ctx.cache_stats().evictions > 0);
        assert!(ctx.cache_stats().occupied_slots < HUFFMAN_CACHE_SLOTS as u64);

        let survivor_raw = {
            let hole = ctx
                .huffman_tables
                .iter()
                .position(Option::is_none)
                .expect("byte-cap eviction leaves a cache hole");
            let survivor = ctx.huffman_tables[hole + 1..]
                .iter()
                .flatten()
                .next()
                .expect("a cached table survives after the first hole");
            raws.iter()
                .find(|raw| {
                    survivor
                        .table
                        .matches_raw_identity(raw, HuffmanTableRole::Ac)
                })
                .expect("surviving cached table came from the inserted working set")
                .clone()
        };
        let before = ctx.cache_stats();
        ctx.resolve_huffman_table_with_digest_and_live_budget(
            &survivor_raw,
            HuffmanTableRole::Ac,
            0,
            &mut live_bytes,
            MAX_DECODER_CONTEXT_ALLOCATION_BYTES,
        )
        .expect("an exact entry remains reachable after a cache hole");
        let after = ctx.cache_stats();
        assert_eq!(after.hits, before.hits + 1);
        assert_eq!(after.misses, before.misses);
        assert_eq!(after.evictions, before.evictions);
        assert_eq!(after.occupied_slots, before.occupied_slots);
    }

    #[test]
    fn context_reserve_covers_all_fixed_table_cache_allocations() {
        let slot_bytes =
            HUFFMAN_CACHE_SLOTS.saturating_mul(size_of::<Option<CachedHuffmanTable>>());
        assert!(
            slot_bytes.saturating_add(HuffmanTable::maximum_retained_allocation_bytes())
                < TABLE_CACHE_ALLOCATION_RESERVE_BYTES
        );

        let ctx = DecoderContext::new();
        assert_eq!(ctx.retained_allocation_bytes(), 0);
        assert!(ctx.decode_plan_cache_bytes <= MAX_DECODE_PLAN_CACHE_BYTES);
    }
}
