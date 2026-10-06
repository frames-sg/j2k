// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::decoder::decode_profile::share_plan_wall_time;

use j2k::{DeviceDecodePlan, DeviceDecodeRequest};
use j2k_native::{ColorSpace, DecodeErrorClass, DecodeSettings};

use super::super::{
    build_cuda_htj2k_color_plans_from_bytes_with_profile_and_cap, profile,
    CudaHtj2kColorDecodePlans, Error, HostPhaseBudget, NativeDecoderContext, PixelFormat,
    CUDA_HTJ2K_KERNELS_NOT_READY,
};
use crate::decoder::plan::{
    build_cuda_htj2k_color_plan_from_referenced_direct_source,
    build_cuda_htj2k_color_plans_from_referenced_with_profile, next_payload_base,
    parse_with_host_cap, plan_in_waves, referenced_classic_payload_bytes,
    referenced_ht_payload_bytes,
};
use crate::error::native_decode_error;

const PLAN_OWNERS: &str = "j2k CUDA color batch plan owners";

#[derive(Clone, Copy)]
enum PreparedColorPayload<'a> {
    Borrowed(&'a [u8]),
    Owned,
}

struct PreparedColorInput<'a> {
    color: CudaHtj2kColorDecodePlans,
    payload: PreparedColorPayload<'a>,
}

pub(super) struct PreparedColorCudaResidentBatch<'a> {
    pub(super) colors: Vec<CudaHtj2kColorDecodePlans>,
    payloads: Vec<PreparedColorPayload<'a>>,
}

impl PreparedColorCudaResidentBatch<'_> {
    pub(super) fn retained_host_bytes(&self) -> Result<usize, Error> {
        let mut budget = HostPhaseBudget::new(PLAN_OWNERS);
        self.account_host_owners(&mut budget)?;
        Ok(budget.live_bytes())
    }

    pub(super) fn account_host_owners(&self, budget: &mut HostPhaseBudget) -> Result<(), Error> {
        super::super::host_owners::account_colors(budget, &self.colors)?;
        budget.account_vec(&self.payloads)?;
        Ok(())
    }

    pub(super) fn payload_parts_with_live_bytes(
        &self,
        external_live_bytes: usize,
    ) -> Result<Vec<&[u8]>, Error> {
        if self.colors.len() != self.payloads.len() {
            return Err(Error::capability_rejected(
                j2k_core::CapabilityRejection::contract_violation(
                    "prepared color payload owners do not match planned images",
                ),
            ));
        }
        let mut budget = HostPhaseBudget::with_live_bytes(
            "j2k CUDA color batch payload parts",
            external_live_bytes,
        )?;
        self.account_host_owners(&mut budget)?;
        let mut parts = budget.try_vec_with_capacity(self.colors.len())?;
        for (color, payload) in self.colors.iter().zip(&self.payloads) {
            parts.push(match payload {
                PreparedColorPayload::Borrowed(input) => *input,
                PreparedColorPayload::Owned => color.payload.as_slice(),
            });
        }
        Ok(parts)
    }
}

/// Plans one color batch. Each image keeps its own payload; block offsets are
/// rebased onto the batch layout, so the payloads upload back to back in input
/// order without a host-side concatenation.
pub(super) fn prepare_color_cuda_resident_batch<'a>(
    inputs: &[&'a [u8]],
    fmt: PixelFormat,
) -> Result<PreparedColorCudaResidentBatch<'a>, Error> {
    prepare_color_cuda_resident_batch_with_cap(
        inputs,
        fmt,
        j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    )
}

pub(super) fn prepare_color_cuda_resident_batch_with_cap<'a>(
    inputs: &[&'a [u8]],
    fmt: PixelFormat,
    host_cap: usize,
) -> Result<PreparedColorCudaResidentBatch<'a>, Error> {
    let mut budget = HostPhaseBudget::with_cap(PLAN_OWNERS, host_cap);
    let mut colors = budget.try_vec_with_capacity(inputs.len())?;
    let mut payloads = budget.try_vec_with_capacity(inputs.len())?;
    let plan_started = profile::profile_now(true);
    let mut payload_base = 0_u64;
    plan_in_waves(
        inputs,
        host_cap,
        &mut budget,
        |input: &&'a [u8], context: &mut NativeDecoderContext<'a>, worker_cap| {
            build_color_plans_with_cap(input, fmt, context, worker_cap)
        },
        |budget, _, prepared| {
            let PreparedColorInput { mut color, payload } = prepared;
            if color.components.len() != 3 {
                return Err(Error::capability_rejected(
                    j2k_core::CapabilityRejection::missing_prepared_plan(
                        CUDA_HTJ2K_KERNELS_NOT_READY,
                    ),
                ));
            }
            for component in &mut color.components {
                component.rebase_payload_offsets(payload_base)?;
            }
            let payload_len = match payload {
                PreparedColorPayload::Borrowed(input) => input.len(),
                PreparedColorPayload::Owned => color.payload.len(),
            };
            payload_base = next_payload_base(payload_base, payload_len)?;
            color.account_host_owners(budget)?;
            colors.push(color);
            payloads.push(payload);
            Ok(())
        },
    )?;
    let plan_wall_us = profile::elapsed_us(plan_started);
    share_plan_wall_time(&mut colors, plan_wall_us, |color| &mut color.report);
    Ok(PreparedColorCudaResidentBatch { colors, payloads })
}

fn build_color_plans_with_cap<'a>(
    input: &'a [u8],
    fmt: PixelFormat,
    context: &mut NativeDecoderContext<'a>,
    host_cap: usize,
) -> Result<PreparedColorInput<'a>, Error> {
    if let Some(color) = build_referenced_color_plans_with_cap(input, fmt, context, host_cap)? {
        return Ok(color);
    }
    build_cuda_htj2k_color_plans_from_bytes_with_profile_and_cap(input, fmt, context, host_cap).map(
        |color| PreparedColorInput {
            color,
            payload: PreparedColorPayload::Owned,
        },
    )
}

/// Plans a single-tile RGB input from codestream ranges. `Ok(None)` means the
/// input is not eligible for this route (not RGB, has alpha, has more than one
/// tile, needs an added alpha channel, or uses a feature the referenced
/// planner does not support), and the caller plans it from owned coefficients
/// instead.
fn build_referenced_color_plans_with_cap<'a>(
    input: &'a [u8],
    fmt: PixelFormat,
    context: &mut NativeDecoderContext<'a>,
    host_cap: usize,
) -> Result<Option<PreparedColorInput<'a>>, Error> {
    let total_start = profile::profile_now(true);
    let (image, mut host_budget) =
        parse_with_host_cap(input, &DecodeSettings::default(), host_cap, PLAN_OWNERS)?;
    let eligible = matches!(image.color_space(), ColorSpace::RGB)
        && !image.has_alpha()
        && image.tile_count() == 1
        && fmt.channels() == 3;
    if !eligible {
        return Ok(None);
    }
    let parse_us = profile::elapsed_us(total_start);
    let dimensions = (image.width(), image.height());
    let plan_start = profile::profile_now(true);
    let referenced = match image.build_referenced_htj2k_plan_region_with_context(
        context,
        (0, 0, dimensions.0, dimensions.1),
    ) {
        Ok(referenced) => referenced,
        Err(error) if matches!(error.classify(), DecodeErrorClass::Unsupported { .. }) => {
            return Ok(None)
        }
        Err(error) => return Err(native_decode_error(error)),
    };
    let plan_us = profile::elapsed_us(plan_start);
    let device_plan = DeviceDecodePlan::for_image(dimensions, DeviceDecodeRequest::Full)?;
    host_budget.account_bytes(
        referenced
            .retained_allocation_bytes()
            .map_err(j2k_native::DecodeError::from)
            .map_err(native_decode_error)?,
    )?;
    let direct = build_cuda_htj2k_color_plan_from_referenced_direct_source(
        input,
        &referenced,
        fmt,
        device_plan,
        &mut host_budget,
    )?;
    let (mut color, payload) = if let Some(color) = direct {
        (color, PreparedColorPayload::Borrowed(input))
    } else {
        let payload_capacity = referenced_ht_payload_bytes(referenced.payloads())?
            .checked_add(referenced_classic_payload_bytes(referenced.tiles())?)
            .ok_or(Error::capability_rejected(
                j2k_core::CapabilityRejection::resource_limit(
                    "prepared CUDA referenced payload size overflows",
                ),
            ))?;
        let mut payload = host_budget.try_vec_with_capacity(payload_capacity)?;
        let mut colors = build_cuda_htj2k_color_plans_from_referenced_with_profile(
            input,
            &referenced,
            fmt,
            device_plan,
            &mut payload,
            &mut host_budget,
        )?;
        let (Some(mut color), true) = (colors.pop(), colors.is_empty()) else {
            return Err(Error::capability_rejected(
                j2k_core::CapabilityRejection::contract_violation(
                    "single-tile referenced color plan must produce one image",
                ),
            ));
        };
        color.payload = payload;
        color.report.payload_bytes = color.payload.len();
        (color, PreparedColorPayload::Owned)
    };
    if color.output_index != 0
        || color.dimensions != dimensions
        || color.mct_dimensions != dimensions
    {
        return Err(Error::capability_rejected(
            j2k_core::CapabilityRejection::geometry_mismatch(
                "referenced color plan geometry differs from the parsed image",
            ),
        ));
    }
    color.report.parse_us = parse_us;
    color.report.plan_us = plan_us;
    color.report.total_us = profile::elapsed_us(total_start);
    Ok(Some(PreparedColorInput { color, payload }))
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use j2k_native::{encode, EncodeOptions};
    use j2k_test_support::{openhtj2k_sigprop_fixture, openjph_batch_fixtures};

    use super::{
        build_color_plans_with_cap, build_referenced_color_plans_with_cap,
        prepare_color_cuda_resident_batch, prepare_color_cuda_resident_batch_with_cap,
        CudaHtj2kColorDecodePlans, Error, NativeDecoderContext, PixelFormat, PreparedColorPayload,
    };
    use crate::decoder::plan::build_cuda_htj2k_color_plans_from_bytes_with_profile;

    /// A nonzero retained baseline matches batch planning, where each worker is
    /// capped below the whole host budget.
    const TEST_RETAINED_BASELINE: usize = 1024 * 1024;

    fn build_color_plans(
        input: &[u8],
        fmt: PixelFormat,
    ) -> Result<CudaHtj2kColorDecodePlans, Error> {
        build_color_plans_with_cap(
            input,
            fmt,
            &mut NativeDecoderContext::default(),
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )
        .map(|prepared| prepared.color)
    }

    fn build_referenced_color_plans(
        input: &[u8],
        fmt: PixelFormat,
    ) -> Option<CudaHtj2kColorDecodePlans> {
        build_referenced_color_plans_with_cap(
            input,
            fmt,
            &mut NativeDecoderContext::default(),
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES - TEST_RETAINED_BASELINE,
        )
        .expect("referenced color planning")
        .map(|prepared| prepared.color)
    }

    fn profile_fixture(width: u32, height: u32) -> Vec<u8> {
        let pixels = (0..width * height)
            .flat_map(|idx| {
                [
                    ((idx * 17 + idx / 3) & 0xff) as u8,
                    ((idx * 29 + 7) & 0xff) as u8,
                    ((idx * 43 + 19) & 0xff) as u8,
                ]
            })
            .collect::<Vec<_>>();
        let options = EncodeOptions {
            reversible: true,
            use_ht_block_coding: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        encode(&pixels, width, height, 3, 8, false, &options).expect("encode profile fixture")
    }

    fn encoded_fixture(width: u32, height: u32, reversible: bool, levels: u8) -> Vec<u8> {
        encoded_fixture_with_coding(width, height, reversible, levels, true)
    }

    fn encoded_fixture_with_coding(
        width: u32,
        height: u32,
        reversible: bool,
        levels: u8,
        use_ht_block_coding: bool,
    ) -> Vec<u8> {
        let pixels = (0..width * height)
            .flat_map(|idx| {
                [
                    (idx % 251) as u8,
                    ((idx * 7) % 253) as u8,
                    ((idx / 5) % 255) as u8,
                ]
            })
            .collect::<Vec<_>>();
        let options = EncodeOptions {
            reversible,
            use_ht_block_coding,
            num_decomposition_levels: levels,
            ..EncodeOptions::default()
        };
        encode(&pixels, width, height, 3, 8, false, &options).expect("encode fixture")
    }

    /// Single-tile inputs must take the range route and describe the same GPU
    /// work as the owned route: identical geometry and steps, and identical
    /// bytes per block. Inputs the owned route rejects must fail the same way.
    #[test]
    fn color_plans_match_owned_route_or_fail_like_it() {
        let ours = [
            ("5/3, 1 level", encoded_fixture(96, 80, true, 1)),
            ("9/7, 3 levels", encoded_fixture(150, 70, false, 3)),
            ("5/3, 5 levels", encoded_fixture(33, 129, true, 5)),
            (
                "classic 5/3",
                encoded_fixture_with_coding(64, 48, true, 2, false),
            ),
        ];
        let mut cases = Vec::new();
        cases.extend(
            ours.iter()
                .map(|(name, input)| (*name, input.as_slice(), PixelFormat::Rgb8)),
        );
        cases.extend(
            openjph_batch_fixtures()
                .iter()
                .filter(|fixture| fixture.components == 3 && !fixture.signed)
                .map(|fixture| {
                    let fmt = if fixture.precision <= 8 {
                        PixelFormat::Rgb8
                    } else {
                        PixelFormat::Rgb16
                    };
                    (fixture.name, fixture.encoded, fmt)
                }),
        );

        let (mut referenced_count, mut rejected_count, mut classic_blocks) = (0, 0, 0);
        for (name, input, fmt) in cases {
            let owned = match build_cuda_htj2k_color_plans_from_bytes_with_profile(
                input,
                fmt,
                &mut NativeDecoderContext::default(),
            ) {
                Ok(owned) => owned,
                Err(owned_error) => {
                    let Err(routed_error) = build_color_plans(input, fmt) else {
                        panic!("{name}: routed planning succeeded where the owned route failed");
                    };
                    assert_eq!(
                        format!("{routed_error:?}"),
                        format!("{owned_error:?}"),
                        "{name}"
                    );
                    rejected_count += 1;
                    continue;
                }
            };
            let referenced = build_referenced_color_plans(input, fmt)
                .unwrap_or_else(|| panic!("{name}: single-tile input must take the range route"));
            assert_same_gpu_work(name, input, &referenced, &owned);
            referenced_count += 1;
            classic_blocks += owned.report.classic_block_count;
        }
        assert!(referenced_count >= ours.len());
        assert!(
            rejected_count > 0,
            "a multi-tile fixture must exercise the fallback"
        );
        assert!(
            classic_blocks > 0,
            "a classic fixture must exercise classic blocks"
        );
    }

    fn assert_same_gpu_work(
        name: &str,
        input: &[u8],
        referenced: &CudaHtj2kColorDecodePlans,
        owned: &CudaHtj2kColorDecodePlans,
    ) {
        let referenced_payload = if referenced.payload.is_empty() {
            input
        } else {
            &referenced.payload
        };
        assert_eq!(referenced.output_index, owned.output_index, "{name}");
        assert_eq!(referenced.dimensions, owned.dimensions, "{name}");
        assert_eq!(referenced.mct_dimensions, owned.mct_dimensions, "{name}");
        assert_eq!(referenced.bit_depths, owned.bit_depths, "{name}");
        assert_eq!(referenced.mct, owned.mct, "{name}");
        assert_eq!(referenced.transform, owned.transform, "{name}");
        assert_eq!(
            referenced.components.len(),
            owned.components.len(),
            "{name}"
        );
        for (r, o) in referenced.components.iter().zip(&owned.components) {
            assert_eq!(r.dimensions(), o.dimensions(), "{name}");
            assert_eq!(r.subbands(), o.subbands(), "{name}");
            assert_eq!(r.idwt_steps(), o.idwt_steps(), "{name}");
            assert_eq!(r.store_steps(), o.store_steps(), "{name}");
            assert_classic_blocks_match(r, referenced_payload, o, &owned.payload);
            assert_eq!(r.code_blocks().len(), o.code_blocks().len(), "{name}");
            for (rb, ob) in r.code_blocks().iter().zip(o.code_blocks()) {
                assert_eq!(
                    block_bytes(referenced_payload, rb.payload_offset, rb.payload_len),
                    block_bytes(&owned.payload, ob.payload_offset, ob.payload_len),
                    "{name}",
                );
                let mut normalized = *rb;
                normalized.payload_offset = ob.payload_offset;
                assert_eq!(normalized, *ob, "{name}");
            }
        }
    }

    fn assert_classic_blocks_match(
        referenced: &crate::direct_plan::CudaHtj2kDecodePlan,
        referenced_payload: &[u8],
        owned: &crate::direct_plan::CudaHtj2kDecodePlan,
        owned_payload: &[u8],
    ) {
        let (r_blocks, o_blocks) = (
            referenced.classic_code_blocks(),
            owned.classic_code_blocks(),
        );
        assert_eq!(r_blocks.len(), o_blocks.len());
        for (rb, ob) in r_blocks.iter().zip(o_blocks) {
            assert_eq!(
                block_bytes(referenced_payload, rb.payload_offset, rb.payload_len),
                block_bytes(owned_payload, ob.payload_offset, ob.payload_len),
            );
            let mut normalized = *rb;
            normalized.payload_offset = ob.payload_offset;
            assert_eq!(normalized, *ob);
            // Segment offsets are relative to their codeblock, even when the
            // block's payload offset is rebased into a shared upload arena.
            let r_start = rb.segment_start as usize;
            let o_start = ob.segment_start as usize;
            assert_eq!(
                referenced.classic_segments()[r_start..r_start + rb.segment_count as usize],
                owned.classic_segments()[o_start..o_start + ob.segment_count as usize],
            );
        }
    }

    fn block_bytes(payload: &[u8], offset: u64, len: u32) -> &[u8] {
        let start = usize::try_from(offset).expect("offset fits usize");
        &payload[start..start + len as usize]
    }

    #[test]
    fn batch_offsets_address_each_image_in_back_to_back_upload_layout() {
        let small = profile_fixture(96, 80);
        let large = profile_fixture(128, 64);
        let inputs = [small.as_slice(), large.as_slice(), small.as_slice()];

        let colors = prepare_color_cuda_resident_batch(&inputs, PixelFormat::Rgb8).expect("batch");
        let payload_parts = colors
            .payload_parts_with_live_bytes(0)
            .expect("payload parts");
        let uploaded = payload_parts
            .iter()
            .flat_map(|part| part.iter().copied())
            .collect::<Vec<_>>();

        let mut base = 0_u64;
        for ((color, input), payload_part) in
            colors.colors.iter().zip(inputs).zip(payload_parts.iter())
        {
            let single_batch =
                prepare_color_cuda_resident_batch(&[input], PixelFormat::Rgb8).expect("single");
            let single_parts = single_batch
                .payload_parts_with_live_bytes(0)
                .expect("single payload parts");
            let single = &single_batch.colors[0];
            assert_eq!(color.components.len(), single.components.len());
            for (batched, alone) in color.components.iter().zip(&single.components) {
                assert_eq!(batched.code_blocks().len(), alone.code_blocks().len());
                for (block, reference) in batched.code_blocks().iter().zip(alone.code_blocks()) {
                    assert_eq!(block.payload_offset, reference.payload_offset + base);
                    assert_eq!(
                        block_bytes(&uploaded, block.payload_offset, block.payload_len),
                        block_bytes(
                            single_parts[0],
                            reference.payload_offset,
                            reference.payload_len
                        ),
                    );
                }
            }
            base += payload_part.len() as u64;
        }
        assert_eq!(base, uploaded.len() as u64);
    }

    #[test]
    fn cleanup_only_uses_borrowed_source_and_other_entropy_routes_keep_owned_payloads() {
        let cleanup_only = profile_fixture(96, 80);
        let prepared =
            prepare_color_cuda_resident_batch(&[cleanup_only.as_slice()], PixelFormat::Rgb8)
                .expect("cleanup-only source plan");
        let parts = prepared
            .payload_parts_with_live_bytes(0)
            .expect("cleanup-only payload parts");
        assert_eq!(prepared.colors[0].payload, [] as [u8; 0]);
        assert!(matches!(
            prepared.payloads.as_slice(),
            [PreparedColorPayload::Borrowed(_)]
        ));
        assert_eq!(parts[0].as_ptr(), cleanup_only.as_ptr());
        assert_eq!(parts[0].len(), cleanup_only.len());

        let refinement = openhtj2k_sigprop_fixture();
        let prepared = prepare_color_cuda_resident_batch(&[refinement], PixelFormat::Rgb16)
            .expect("refinement fallback plan");
        let parts = prepared
            .payload_parts_with_live_bytes(0)
            .expect("refinement payload parts");
        assert!(matches!(
            prepared.payloads.as_slice(),
            [PreparedColorPayload::Owned]
        ));
        assert_ne!(prepared.colors[0].payload, [] as [u8; 0]);
        assert_eq!(parts[0].as_ptr(), prepared.colors[0].payload.as_ptr());

        let classic = encoded_fixture_with_coding(64, 48, true, 2, false);
        let prepared = prepare_color_cuda_resident_batch(&[classic.as_slice()], PixelFormat::Rgb8)
            .expect("classic fallback plan");
        assert!(matches!(
            prepared.payloads.as_slice(),
            [PreparedColorPayload::Owned]
        ));
        assert_ne!(prepared.colors[0].payload, [] as [u8; 0]);
    }

    #[test]
    fn rgb_input_with_alpha_output_plans_through_the_owned_route() {
        let input = profile_fixture(32, 32);
        assert!(build_referenced_color_plans(&input, PixelFormat::Rgba8).is_none());
        let prepared = prepare_color_cuda_resident_batch(&[input.as_slice()], PixelFormat::Rgba8)
            .expect("RGB input planned for RGBA output");
        assert!(matches!(
            prepared.payloads.as_slice(),
            [PreparedColorPayload::Owned]
        ));
    }

    #[test]
    fn batch_plan_report_times_do_not_exceed_wall_time() {
        let fixture = profile_fixture(256, 256);
        let inputs = [fixture.as_slice(); 8];
        let start = Instant::now();
        let colors = prepare_color_cuda_resident_batch(&inputs, PixelFormat::Rgb8).expect("batch");
        let wall_us = start.elapsed().as_micros();
        assert!(staged_us(&colors.colors) <= wall_us);
    }

    #[test]
    fn parallel_color_planning_returns_the_first_input_error() {
        let valid = profile_fixture(32, 32);
        let first_invalid = [0xff_u8, 0x4f, 0x00];
        let later_invalid = [0x00_u8, 0x01];
        let expected =
            prepare_color_cuda_resident_batch(&[first_invalid.as_slice()], PixelFormat::Rgb8)
                .err()
                .expect("first malformed input must fail");
        let actual = prepare_color_cuda_resident_batch(
            &[
                valid.as_slice(),
                first_invalid.as_slice(),
                later_invalid.as_slice(),
            ],
            PixelFormat::Rgb8,
        )
        .err()
        .expect("batch containing malformed inputs must fail");

        assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
    }

    #[test]
    fn color_batch_cap_covers_retained_results_and_one_live_planner() {
        let fixture = profile_fixture(64, 64);
        let one = [fixture.as_slice()];
        let two = [fixture.as_slice(), fixture.as_slice()];
        let one_cap = minimum_batch_cap(&one);
        let two_cap = minimum_batch_cap(&two);

        assert!(
            two_cap > one_cap,
            "the second retained plan must consume aggregate headroom"
        );
        let Err(error) =
            prepare_color_cuda_resident_batch_with_cap(&two, PixelFormat::Rgb8, two_cap - 1)
        else {
            panic!("one byte below the discovered aggregate cap must fail");
        };
        assert!(
            error.is_host_planning_limit(),
            "unexpected error: {error:?}"
        );
    }

    fn minimum_batch_cap(inputs: &[&[u8]]) -> usize {
        crate::decoder::plan::minimum_planning_cap(|cap| {
            prepare_color_cuda_resident_batch_with_cap(inputs, PixelFormat::Rgb8, cap)
        })
    }

    fn staged_us(colors: &[CudaHtj2kColorDecodePlans]) -> u128 {
        colors
            .iter()
            .map(|color| color.report.parse_us + color.report.plan_us + color.report.flatten_us)
            .sum()
    }
}
