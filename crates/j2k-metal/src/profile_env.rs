// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use std::cell::Cell;
use std::sync::OnceLock;

use metal::{CommandBufferRef, ComputeCommandEncoderRef};

const CLASSIC_SELECTIVE_BYPASS_ENV: &str = "J2K_METAL_CLASSIC_SELECTIVE_BYPASS";
const METAL_PROFILE_STAGES_ENV: &str = "J2K_METAL_PROFILE_STAGES";
const METAL_PROFILE_SIGNPOSTS_ENV: &str = "J2K_METAL_PROFILE_SIGNPOSTS";
const METAL_PROFILE_DECODE_LABEL_ENV: &str = "J2K_METAL_PROFILE_DECODE_LABEL";
const METAL_PROFILE_DECODE_SPLIT_COMMANDS_ENV: &str = "J2K_METAL_PROFILE_DECODE_SPLIT_COMMANDS";
const METAL_PROFILE_COEFFICIENT_PREP_SPLIT_COMMANDS_ENV: &str =
    "J2K_METAL_PROFILE_COEFFICIENT_PREP_SPLIT_COMMANDS";
const METAL_PROFILE_CLASSIC_TIER1_DENSITY_ENV: &str = "J2K_METAL_PROFILE_CLASSIC_TIER1_DENSITY";
const METAL_PROFILE_CLASSIC_TIER1_RAW_PACK_ENV: &str = "J2K_METAL_PROFILE_CLASSIC_TIER1_RAW_PACK";
const METAL_PROFILE_CLASSIC_TIER1_ARITHMETIC_PACK_ENV: &str =
    "J2K_METAL_PROFILE_CLASSIC_TIER1_ARITHMETIC_PACK";
const METAL_PROFILE_CLASSIC_TIER1_SYMBOL_PLAN_ENV: &str =
    "J2K_METAL_PROFILE_CLASSIC_TIER1_SYMBOL_PLAN";
const METAL_PROFILE_CLASSIC_TIER1_PASS_PLAN_ENV: &str = "J2K_METAL_PROFILE_CLASSIC_TIER1_PASS_PLAN";
const METAL_PROFILE_CLASSIC_TIER1_TOKEN_EMIT_ENV: &str =
    "J2K_METAL_PROFILE_CLASSIC_TIER1_TOKEN_EMIT";
const METAL_PROFILE_CLASSIC_TIER1_SPLIT_TOKEN_EMIT_ENV: &str =
    "J2K_METAL_PROFILE_CLASSIC_TIER1_SPLIT_TOKEN_EMIT";
const METAL_PROFILE_CLASSIC_TIER1_TOKEN_PACK_ENV: &str =
    "J2K_METAL_PROFILE_CLASSIC_TIER1_TOKEN_PACK";
const CLASSIC_TIER1_GPU_TOKEN_PACK_ENV: &str = "J2K_METAL_CLASSIC_TIER1_GPU_TOKEN_PACK";
const CLASSIC_TIER1_SPLIT_GPU_TOKEN_PACK_ENV: &str = "J2K_METAL_CLASSIC_TIER1_SPLIT_GPU_TOKEN_PACK";
const CLASSIC_TIER1_SPLIT_MQ_BYTE_GPU_TOKEN_PACK_ENV: &str =
    "J2K_METAL_CLASSIC_TIER1_SPLIT_MQ_BYTE_GPU_TOKEN_PACK";

pub(crate) type HybridSignpostName = u32;

pub(crate) const SIGNPOST_DECODE_HYBRID_CPU_TIER1: HybridSignpostName = 1;
pub(crate) const SIGNPOST_DECODE_HYBRID_COEFFICIENT_UPLOAD: HybridSignpostName = 2;
pub(crate) const SIGNPOST_DECODE_HYBRID_COMMAND_WAIT: HybridSignpostName = 3;
pub(crate) const SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE: HybridSignpostName = 4;
pub(crate) const SIGNPOST_DECODE_HYBRID_STORE_COMMAND_ENCODE: HybridSignpostName = 5;
pub(crate) const SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE: HybridSignpostName = 6;
pub(crate) const SIGNPOST_ENCODE_HYBRID_COMMAND_WAIT: HybridSignpostName = 7;
pub(crate) const SIGNPOST_ENCODE_HYBRID_RESULT_HARVEST: HybridSignpostName = 8;
pub(crate) const SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_SETUP: HybridSignpostName = 9;
pub(crate) const SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_COMMAND_ENCODE: HybridSignpostName = 10;
pub(crate) const SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_PLAN: HybridSignpostName = 11;
pub(crate) const SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_BUFFER_SETUP: HybridSignpostName = 12;
pub(crate) const SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKETIZATION_COMMAND_ENCODE: HybridSignpostName =
    13;
pub(crate) const SIGNPOST_ENCODE_HYBRID_CLASSIC_PAYLOAD_COPY_COMMAND_ENCODE: HybridSignpostName =
    14;
pub(crate) const SIGNPOST_ENCODE_HYBRID_CLASSIC_CODESTREAM_ASSEMBLY_COMMAND_ENCODE:
    HybridSignpostName = 15;
pub(crate) const SIGNPOST_ENCODE_HYBRID_HT_TIER1_SETUP: HybridSignpostName = 16;
pub(crate) const SIGNPOST_ENCODE_HYBRID_HT_TIER1_COMMAND_ENCODE: HybridSignpostName = 17;
pub(crate) const SIGNPOST_ENCODE_HYBRID_HT_PACKET_PLAN: HybridSignpostName = 18;
pub(crate) const SIGNPOST_ENCODE_HYBRID_HT_PACKET_BUFFER_SETUP: HybridSignpostName = 19;
pub(crate) const SIGNPOST_ENCODE_HYBRID_HT_PACKET_BLOCK_PREP_COMMAND_ENCODE: HybridSignpostName =
    20;
pub(crate) const SIGNPOST_ENCODE_HYBRID_HT_PACKETIZATION_COMMAND_ENCODE: HybridSignpostName = 21;
pub(crate) const SIGNPOST_ENCODE_HYBRID_HT_PAYLOAD_COPY_COMMAND_ENCODE: HybridSignpostName = 22;
pub(crate) const SIGNPOST_ENCODE_HYBRID_HT_CODESTREAM_ASSEMBLY_COMMAND_ENCODE: HybridSignpostName =
    23;

#[cfg(test)]
std::thread_local! {
    static CLASSIC_GPU_TOKEN_PACK_ROUTE_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static METAL_PROFILE_STAGES_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

fn env_flag_enabled(name: &str) -> bool {
    matches!(std::env::var(name), Ok(value) if value == "1")
}

pub(crate) fn classic_selective_bypass_disabled() -> bool {
    matches!(std::env::var(CLASSIC_SELECTIVE_BYPASS_ENV), Ok(value) if value == "0")
}

#[cfg(test)]
pub(crate) struct ClassicGpuTokenPackRouteOverrideGuard {
    previous: Option<bool>,
}

#[cfg(test)]
impl Drop for ClassicGpuTokenPackRouteOverrideGuard {
    fn drop(&mut self) {
        CLASSIC_GPU_TOKEN_PACK_ROUTE_OVERRIDE.with(|slot| slot.set(self.previous));
    }
}

#[cfg(test)]
pub(crate) fn force_classic_gpu_token_pack_route_for_test(
    enabled: bool,
) -> ClassicGpuTokenPackRouteOverrideGuard {
    let previous = CLASSIC_GPU_TOKEN_PACK_ROUTE_OVERRIDE.with(|slot| slot.replace(Some(enabled)));
    ClassicGpuTokenPackRouteOverrideGuard { previous }
}

#[cfg(test)]
pub(crate) struct MetalProfileStagesOverrideGuard {
    previous: Option<bool>,
}

#[cfg(test)]
impl Drop for MetalProfileStagesOverrideGuard {
    fn drop(&mut self) {
        METAL_PROFILE_STAGES_OVERRIDE.with(|slot| slot.set(self.previous));
    }
}

#[cfg(test)]
pub(crate) fn force_metal_profile_stages_for_test(
    enabled: bool,
) -> MetalProfileStagesOverrideGuard {
    let previous = METAL_PROFILE_STAGES_OVERRIDE.with(|slot| slot.replace(Some(enabled)));
    MetalProfileStagesOverrideGuard { previous }
}

pub(crate) fn classic_tier1_gpu_token_pack_requested() -> bool {
    #[cfg(test)]
    if let Some(enabled) = CLASSIC_GPU_TOKEN_PACK_ROUTE_OVERRIDE.with(Cell::get) {
        return enabled;
    }
    env_flag_enabled(CLASSIC_TIER1_GPU_TOKEN_PACK_ENV)
}

pub(crate) fn classic_tier1_split_gpu_token_pack_requested() -> bool {
    env_flag_enabled(CLASSIC_TIER1_SPLIT_GPU_TOKEN_PACK_ENV)
}

fn classic_tier1_split_mq_byte_gpu_token_pack_setting() -> Option<bool> {
    match std::env::var(CLASSIC_TIER1_SPLIT_MQ_BYTE_GPU_TOKEN_PACK_ENV) {
        Ok(value) if value == "1" => Some(true),
        Ok(value) if value == "0" => Some(false),
        _ => None,
    }
}

pub(crate) fn classic_tier1_split_mq_byte_gpu_token_pack_requested() -> bool {
    classic_tier1_split_mq_byte_gpu_token_pack_setting() == Some(true)
}

pub(crate) fn classic_tier1_split_mq_byte_gpu_token_pack_disabled() -> bool {
    classic_tier1_split_mq_byte_gpu_token_pack_setting() == Some(false)
}

pub(crate) fn metal_profile_stages_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    #[cfg(test)]
    if let Some(enabled) = METAL_PROFILE_STAGES_OVERRIDE.with(Cell::get) {
        return enabled;
    }
    *ENABLED.get_or_init(|| env_flag_enabled(METAL_PROFILE_STAGES_ENV))
}

fn metal_profile_signposts_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled(METAL_PROFILE_SIGNPOSTS_ENV))
}

pub(crate) fn metal_profile_decode_split_commands_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    metal_profile_stages_enabled()
        && *ENABLED.get_or_init(|| env_flag_enabled(METAL_PROFILE_DECODE_SPLIT_COMMANDS_ENV))
}

pub(crate) fn metal_profile_coefficient_prep_split_commands_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    metal_profile_stages_enabled()
        && *ENABLED
            .get_or_init(|| env_flag_enabled(METAL_PROFILE_COEFFICIENT_PREP_SPLIT_COMMANDS_ENV))
}

pub(crate) fn metal_profile_classic_tier1_density_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    metal_profile_stages_enabled()
        && *ENABLED.get_or_init(|| env_flag_enabled(METAL_PROFILE_CLASSIC_TIER1_DENSITY_ENV))
}

pub(crate) fn metal_profile_classic_tier1_raw_pack_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    metal_profile_stages_enabled()
        && *ENABLED.get_or_init(|| env_flag_enabled(METAL_PROFILE_CLASSIC_TIER1_RAW_PACK_ENV))
}

pub(crate) fn metal_profile_classic_tier1_arithmetic_pack_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    metal_profile_stages_enabled()
        && *ENABLED
            .get_or_init(|| env_flag_enabled(METAL_PROFILE_CLASSIC_TIER1_ARITHMETIC_PACK_ENV))
}

pub(crate) fn metal_profile_classic_tier1_symbol_plan_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    metal_profile_stages_enabled()
        && *ENABLED.get_or_init(|| {
            env_flag_enabled(METAL_PROFILE_CLASSIC_TIER1_SYMBOL_PLAN_ENV)
                || env_flag_enabled(METAL_PROFILE_CLASSIC_TIER1_PASS_PLAN_ENV)
        })
}

pub(crate) fn metal_profile_classic_tier1_pass_plan_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    metal_profile_stages_enabled()
        && *ENABLED.get_or_init(|| env_flag_enabled(METAL_PROFILE_CLASSIC_TIER1_PASS_PLAN_ENV))
}

pub(crate) fn metal_profile_classic_tier1_token_emit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    metal_profile_stages_enabled()
        && *ENABLED.get_or_init(|| {
            env_flag_enabled(METAL_PROFILE_CLASSIC_TIER1_TOKEN_EMIT_ENV)
                || env_flag_enabled(METAL_PROFILE_CLASSIC_TIER1_TOKEN_PACK_ENV)
        })
}

pub(crate) fn metal_profile_classic_tier1_split_token_emit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    metal_profile_stages_enabled()
        && *ENABLED
            .get_or_init(|| env_flag_enabled(METAL_PROFILE_CLASSIC_TIER1_SPLIT_TOKEN_EMIT_ENV))
}

pub(crate) fn metal_profile_classic_tier1_token_pack_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    metal_profile_stages_enabled()
        && *ENABLED.get_or_init(|| env_flag_enabled(METAL_PROFILE_CLASSIC_TIER1_TOKEN_PACK_ENV))
}

pub(crate) fn decode_profile_label() -> String {
    std::env::var(METAL_PROFILE_DECODE_LABEL_ENV)
        .ok()
        .filter(|label| !label.is_empty())
        .map_or_else(
            || "unlabeled".to_string(),
            |label| sanitize_profile_label(&label),
        )
}

fn sanitize_profile_label(label: &str) -> String {
    label
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

pub(crate) fn label_command_buffer(command_buffer: &CommandBufferRef, label: &str) {
    if metal_profile_stages_enabled() {
        command_buffer.set_label(label);
    }
}

type OsSignpostId = u64;

const OS_SIGNPOST_ID_NULL: OsSignpostId = 0;
const OS_SIGNPOST_ID_INVALID: OsSignpostId = OsSignpostId::MAX;

unsafe extern "C" {
    fn j2k_metal_signpost_begin(name: HybridSignpostName) -> OsSignpostId;
    fn j2k_metal_signpost_end(name: HybridSignpostName, id: OsSignpostId);
}

pub(crate) struct HybridStageSignpost {
    id: OsSignpostId,
    name: HybridSignpostName,
}

impl Drop for HybridStageSignpost {
    fn drop(&mut self) {
        // SAFETY: The shim accepts the signpost id returned by the matching begin call.
        unsafe {
            j2k_metal_signpost_end(self.name, self.id);
        }
    }
}

pub(crate) fn hybrid_stage_signpost(name: HybridSignpostName) -> Option<HybridStageSignpost> {
    if !metal_profile_signposts_enabled() {
        return None;
    }
    // SAFETY: The signpost shim is provided by this crate's build and has no Rust aliasing contract.
    let id = unsafe { j2k_metal_signpost_begin(name) };
    if id == OS_SIGNPOST_ID_NULL || id == OS_SIGNPOST_ID_INVALID {
        return None;
    }
    Some(HybridStageSignpost { id, name })
}

pub(crate) fn label_compute_encoder(encoder: &ComputeCommandEncoderRef, label: &str) {
    if metal_profile_stages_enabled() {
        encoder.set_label(label);
    }
}
