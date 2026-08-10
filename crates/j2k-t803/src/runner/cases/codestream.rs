// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::Colorspace;
use j2k_native::Htj2kCapabilityMode;

use crate::{HtCodeBlockSetMode, Part15CodestreamEvidence};

#[derive(Clone, Debug)]
pub(in crate::runner) struct CodestreamRequirements {
    pub(in crate::runner) component_transform: Option<Colorspace>,
    #[cfg(any(feature = "cuda-runner", feature = "metal-runner"))]
    pub(in crate::runner) high_throughput: bool,
    pub(in crate::runner) part15: Option<Part15CodestreamEvidence>,
}

pub(in crate::runner) fn codestream_requirements(
    input: &[u8],
) -> Result<CodestreamRequirements, String> {
    let payload = j2k::extract_j2k_codestream_payload(input).map_err(|error| error.to_string())?;
    let codestream = payload.codestream();
    let header =
        j2k_native::inspect_j2k_codestream_header(codestream).map_err(|error| error.to_string())?;
    let capabilities =
        j2k_native::inspect_htj2k_capabilities(codestream).map_err(|error| error.to_string())?;
    let part15 = capabilities
        .map(|capabilities| {
            let mut corresponding_profile_words = Vec::new();
            if let Some(profile) = capabilities.corresponding_profile() {
                corresponding_profile_words
                    .try_reserve_exact(profile.words().len())
                    .map_err(|_| "cannot allocate CPF evidence words".to_string())?;
                corresponding_profile_words.extend_from_slice(profile.words());
            }
            Ok::<Part15CodestreamEvidence, String>(Part15CodestreamEvidence {
                pcap: capabilities.pcap(),
                ccap15: capabilities.ccap15(),
                mode: match capabilities.mode() {
                    Htj2kCapabilityMode::HtOnly => HtCodeBlockSetMode::HtOnly,
                    Htj2kCapabilityMode::HtDeclared => HtCodeBlockSetMode::HtDeclared,
                    Htj2kCapabilityMode::Mixed => HtCodeBlockSetMode::Mixed,
                },
                multiple_ht_sets: capabilities.multiple_ht_sets(),
                roi: capabilities.roi(),
                heterogeneous: capabilities.heterogeneous(),
                ht_irreversible: capabilities.ht_irreversible(),
                bmagb: capabilities.magnitude_bound(),
                quality_layers: capabilities.quality_layers(),
                default_ht_block_coding: capabilities.default_ht_block_coding(),
                default_mixed_block_coding: capabilities.default_mixed_block_coding(),
                corresponding_profile_words,
                reversible: header.reversible,
                component_count: header.components,
            })
        })
        .transpose()?;
    Ok(CodestreamRequirements {
        component_transform: header.has_mct.then_some(if header.reversible {
            Colorspace::Rct
        } else {
            Colorspace::Ict
        }),
        #[cfg(any(feature = "cuda-runner", feature = "metal-runner"))]
        high_throughput: header.high_throughput,
        part15,
    })
}
