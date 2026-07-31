// SPDX-License-Identifier: MIT OR Apache-2.0

use std::hint::black_box;
use std::panic::catch_unwind;
use std::sync::OnceLock;

use j2k_alloc_probe::{assert_allocations, measure, Budget};
use j2k_native::{
    decode_ht_code_block_scalar_with_workspace, encode,
    encode_precomputed_htj2k_97_with_accelerator_and_max_host_bytes,
    CpuOnlyJ2kEncodeStageAccelerator, DecodeSettings, DecoderContext, EncodeError, EncodeOptions,
    HtCodeBlockDecodeJob, HtCodeBlockDecodeWorkspace, Image, J2kForwardDwt97Output,
    PrecomputedHtj2k97Component, PrecomputedHtj2k97Image,
};
use rayon::{ThreadPool, ThreadPoolBuilder};

const KIB: u64 = 1024;
static PROBE_POOL: OnceLock<ThreadPool> = OnceLock::new();

fn main() {
    harness_counts_allocation_and_deallocation();
    harness_counts_reallocation_and_shrink();
    harness_selftest_catches_budget_violation();
    peak_budget_allows_released_zeroed_allocation();
    panicking_measurement_releases_global_meter();
    warmed_scalar_decode_workspace_reuses_without_allocating();
    profile_row_formatting_is_single_allocation();

    probe_pool().broadcast(|_| {});
    warmed_decoder_context_has_bounded_transients();
    precomputed_encode_obeys_ledger_boundary_and_allocator_budget();
}

fn harness_counts_allocation_and_deallocation() {
    let (allocation, retained) = measure(|| black_box(Box::new([0_u8; 4096])));
    assert_eq!(retained.allocations, 1);
    assert_eq!(retained.reallocations, 0);
    assert!(retained.allocated_bytes >= 4096);
    assert!(retained.peak_live_bytes >= 4096);
    assert!(retained.retained_bytes >= 4096);
    drop(allocation);

    let ((), released) = measure(|| {
        drop(black_box(Box::new([0_u8; 4096])));
    });
    assert_eq!(released.allocations, 1);
    assert_eq!(released.reallocations, 0);
    assert!(released.peak_live_bytes >= 4096);
    assert_eq!(released.retained_bytes, 0);
}

fn harness_counts_reallocation_and_shrink() {
    let ((), stats) = measure(|| {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(16)
            .expect("small allocation must succeed");
        bytes.resize(16, 1_u8);
        bytes
            .try_reserve_exact(4096)
            .expect("growth allocation must succeed");
        bytes.shrink_to_fit();
        black_box(&bytes);
    });

    assert_eq!(stats.allocations, 1);
    assert!(stats.reallocations >= 1);
    assert!(stats.peak_live_bytes >= 4096);
    assert_eq!(stats.retained_bytes, 0);
}

fn harness_selftest_catches_budget_violation() {
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let violation = catch_unwind(|| {
        let allocation = assert_allocations("planted violation", Budget::zero(), || {
            black_box(Box::new([7_u8; 128]))
        });
        drop(allocation);
    });
    std::panic::set_hook(original_hook);
    assert!(
        violation.is_err(),
        "the planted allocation escaped the probe"
    );
}

fn peak_budget_allows_released_zeroed_allocation() {
    assert_allocations("released zeroed allocation", Budget::peak(8 * KIB), || {
        let bytes = vec![0_u8; 4096];
        black_box(&bytes);
        drop(bytes);
    });
}

fn panicking_measurement_releases_global_meter() {
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let panic = catch_unwind(|| measure(|| panic!("planted measured panic")));
    std::panic::set_hook(original_hook);
    assert!(panic.is_err(), "measured panic must propagate");

    let ((), stats) = measure(|| ());
    assert_eq!(stats.allocations, 0);
    assert_eq!(stats.retained_bytes, 0);
}

fn warmed_scalar_decode_workspace_reuses_without_allocating() {
    const SIGPROP_BLOCK: [u8; 12] = [
        0x0E, 0xB2, 0x3E, 0x30, 0xFD, 0x6B, 0x5C, 0x7A, 0xF7, 0x56, 0x00, 0x02,
    ];
    let job = HtCodeBlockDecodeJob {
        data: &SIGPROP_BLOCK,
        cleanup_length: 11,
        refinement_length: 1,
        width: 4,
        height: 4,
        output_stride: 4,
        missing_bit_planes: 4,
        number_of_coding_passes: 2,
        num_bitplanes: 6,
        roi_shift: 0,
        stripe_causal: false,
        strict: true,
        dequantization_step: 1.0,
    };
    let mut output = [0.0_f32; 16];
    let mut workspace = HtCodeBlockDecodeWorkspace::default();

    decode_ht_code_block_scalar_with_workspace(job, &mut output, &mut workspace)
        .expect("warm scalar HT workspace");
    assert_allocations("warm scalar HT workspace", Budget::zero(), || {
        decode_ht_code_block_scalar_with_workspace(job, &mut output, &mut workspace)
    })
    .expect("decode with warm scalar HT workspace");
    assert!(
        output.iter().any(|coefficient| *coefficient != 0.0),
        "fixture must exercise nonzero decode work"
    );
}

fn warmed_decoder_context_has_bounded_transients() {
    let pool = probe_pool();
    let pixels = (0_u8..64).collect::<Vec<_>>();
    let encoded = pool
        .install(|| {
            encode(
                &pixels,
                8,
                8,
                1,
                8,
                false,
                &EncodeOptions {
                    reversible: true,
                    num_decomposition_levels: 1,
                    ..EncodeOptions::default()
                },
            )
        })
        .expect("encode decoder fixture");
    let image = Image::new(&encoded, &DecodeSettings::default()).expect("parse decoder fixture");
    let mut context = DecoderContext::default();
    let mut output = vec![0_u8; 64];

    pool.install(|| image.decode_into(&mut output, &mut context))
        .expect("warm decoder context");
    assert_allocations(
        "warm DecoderContext reuse",
        Budget::peak_retaining(8 * KIB, 2 * KIB).with_max_allocations(64),
        || pool.install(|| image.decode_into(&mut output, &mut context)),
    )
    .expect("decode with warm context");
    assert_eq!(output, pixels);
}

fn profile_row_formatting_is_single_allocation() {
    let fields = [("route", "scalar"), ("result", "success")];
    drop(
        j2k_profile::format_profile_key_value_fields(&fields).expect("warm profile row formatting"),
    );

    let row = assert_allocations(
        "profile row formatting",
        Budget::peak_retaining(KIB, KIB).with_max_allocations(1),
        || j2k_profile::format_profile_key_value_fields(&fields),
    )
    .expect("format profile fields");
    assert_eq!(row, " route=scalar result=success");
}

fn precomputed_encode_obeys_ledger_boundary_and_allocator_budget() {
    let image = precomputed_image();
    let options = precomputed_options();
    let pool = probe_pool();
    let exact_cap = minimum_successful_cap(pool, &image, &options);
    assert!(
        exact_cap > 0,
        "ledger boundary must charge at least one byte"
    );

    let encoded = assert_allocations(
        "precomputed HTJ2K encode",
        Budget::peak_retaining(256 * KIB, 64 * KIB).with_max_allocations(256),
        || encode_precomputed_at_cap(pool, &image, &options, exact_cap),
    )
    .expect("exact ledger cap must encode");
    assert!(!encoded.is_empty());

    let error = encode_precomputed_at_cap(pool, &image, &options, exact_cap - 1)
        .expect_err("one byte below the exact ledger cap must fail");
    assert!(matches!(
        error,
        EncodeError::AllocationTooLarge { cap, requested, .. }
            if cap == exact_cap - 1 && requested > cap
    ));

    let (measured, stats) =
        measure(|| encode_precomputed_at_cap(pool, &image, &options, exact_cap));
    measured.expect("measured encode at exact cap");
    assert!(stats.allocations > 0);
    assert!(stats.peak_live_bytes > 0);
    let ledger_cap = u64::try_from(exact_cap).unwrap_or(u64::MAX);
    assert!(
        stats.peak_live_bytes <= ledger_cap,
        "host allocation ledger undercounted observed peak live bytes: \
         exact_cap={exact_cap}, stats={stats:?}"
    );
    assert!(
        stats.peak_live_bytes <= 256 * KIB,
        "actual encode peak escaped cross-platform headroom: {stats:?}"
    );
}

fn minimum_successful_cap(
    pool: &ThreadPool,
    image: &PrecomputedHtj2k97Image,
    options: &EncodeOptions,
) -> usize {
    let mut high = 1_usize;
    loop {
        match encode_precomputed_at_cap(pool, image, options, high) {
            Ok(output) => {
                drop(output);
                break;
            }
            Err(EncodeError::AllocationTooLarge { .. }) => {
                high = high.checked_mul(2).expect("ledger boundary must fit usize");
            }
            Err(error) => panic!("unexpected encode error while finding cap: {error}"),
        }
    }

    let mut low = 0_usize;
    while low < high {
        let midpoint = low + (high - low) / 2;
        match encode_precomputed_at_cap(pool, image, options, midpoint) {
            Ok(output) => {
                drop(output);
                high = midpoint;
            }
            Err(EncodeError::AllocationTooLarge { .. }) => low = midpoint + 1,
            Err(error) => panic!("unexpected encode error while narrowing cap: {error}"),
        }
    }
    low
}

fn encode_precomputed_at_cap(
    pool: &ThreadPool,
    image: &PrecomputedHtj2k97Image,
    options: &EncodeOptions,
    cap: usize,
) -> Result<Vec<u8>, EncodeError> {
    pool.install(|| {
        encode_precomputed_htj2k_97_with_accelerator_and_max_host_bytes(
            image,
            options,
            &mut CpuOnlyJ2kEncodeStageAccelerator,
            cap,
        )
    })
}

fn precomputed_options() -> EncodeOptions {
    EncodeOptions {
        num_decomposition_levels: 0,
        reversible: false,
        guard_bits: 2,
        use_ht_block_coding: true,
        code_block_width_exp: 2,
        code_block_height_exp: 2,
        ..EncodeOptions::default()
    }
}

fn precomputed_image() -> PrecomputedHtj2k97Image {
    PrecomputedHtj2k97Image {
        width: 1,
        height: 1,
        bit_depth: 8,
        signed: false,
        components: vec![PrecomputedHtj2k97Component {
            x_rsiz: 1,
            y_rsiz: 1,
            dwt: J2kForwardDwt97Output {
                ll: vec![1.0],
                ll_width: 1,
                ll_height: 1,
                levels: Vec::new(),
            },
        }],
    }
}

fn probe_pool() -> &'static ThreadPool {
    PROBE_POOL.get_or_init(|| {
        ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("build fixed two-thread probe pool")
    })
}
