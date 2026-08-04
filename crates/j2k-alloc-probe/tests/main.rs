// SPDX-License-Identifier: MIT OR Apache-2.0

use std::hint::black_box;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{mpsc, OnceLock};

use j2k_alloc_probe::{assert_allocations, measure, Budget};
use j2k_native::{
    decode_ht_code_block_scalar_with_workspace, encode,
    encode_precomputed_htj2k_97_with_accelerator_and_max_host_bytes,
    CpuOnlyJ2kEncodeStageAccelerator, DecodeSettings, DecoderContext, EncodeError, EncodeOptions,
    HtCodeBlockDecodeJob, HtCodeBlockDecodeWorkspace, Image, J2kForwardDwt97Output,
    PrecomputedHtj2k97Component, PrecomputedHtj2k97Image,
};
use proptest::{
    collection::vec,
    prop_assert, prop_assert_eq,
    test_runner::{Config, TestRunner},
};
use rayon::{ThreadPool, ThreadPoolBuilder};

const KIB: u64 = 1024;
static PROBE_POOL: OnceLock<ThreadPool> = OnceLock::new();

fn main() {
    // Keep the Rayon pool uninitialized while exact allocator self-tests run.
    // The process-wide meter intentionally records unrelated worker activity.
    let cases: &[(&str, fn())] = &[
        (
            "harness_counts_allocation_and_deallocation",
            harness_counts_allocation_and_deallocation,
        ),
        ("harness_counts_reallocation", harness_counts_reallocation),
        (
            "preexisting_deallocation_cannot_credit_budget",
            preexisting_deallocation_cannot_credit_budget,
        ),
        (
            "preexisting_frees_never_hide_requested_bytes",
            preexisting_frees_never_hide_requested_bytes,
        ),
        (
            "concurrent_measurement_is_rejected_without_resetting_counters",
            concurrent_measurement_is_rejected_without_resetting_counters,
        ),
        (
            "panicking_measurement_releases_global_meter",
            panicking_measurement_releases_global_meter,
        ),
        (
            "warmed_scalar_decode_workspace_reuses_without_allocating",
            warmed_scalar_decode_workspace_reuses_without_allocating,
        ),
        (
            "warmed_decoder_context_has_bounded_transients",
            warmed_decoder_context_has_bounded_transients,
        ),
        (
            "profile_row_formatting_is_single_allocation",
            profile_row_formatting_is_single_allocation,
        ),
        (
            "precomputed_encode_obeys_ledger_and_allocator_budgets",
            precomputed_encode_obeys_ledger_and_allocator_budgets,
        ),
    ];

    let mut failures = 0_u32;
    for (name, case) in cases {
        print!("test {name} ... ");
        let result = catch_unwind(AssertUnwindSafe(case));
        if result.is_ok() {
            println!("ok");
        } else {
            failures += 1;
            println!("FAILED");
        }
    }

    println!(
        "test result: {}. {} passed; {} failed",
        if failures == 0 { "ok" } else { "FAILED" },
        cases.len() - usize::try_from(failures).expect("failure count fits usize"),
        failures
    );
    if failures != 0 {
        std::process::exit(1);
    }
}

fn harness_counts_allocation_and_deallocation() {
    // Allocate the backing storage directly. In debug builds, constructing a
    // large array inside `Box::new` can use an additional temporary allocation.
    let (allocation, retained) = measure(|| black_box(Box::<[u8]>::new_uninit_slice(4096)));
    assert_eq!(retained.allocations, 1);
    assert_eq!(retained.reallocations, 0);
    assert_eq!(retained.deallocations, 0);
    assert!(retained.requested_bytes >= 4096);
    drop(allocation);

    let ((), released) = measure(|| {
        drop(black_box(Box::<[u8]>::new_uninit_slice(4096)));
    });
    assert_eq!(released.allocations, 1);
    assert_eq!(released.reallocations, 0);
    assert_eq!(released.deallocations, 1);
    assert!(released.requested_bytes >= 4096);
}

fn harness_counts_reallocation() {
    let ((), stats) = measure(|| {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(16)
            .expect("small allocation must succeed");
        bytes.resize(16, 1_u8);
        bytes
            .try_reserve_exact(4096)
            .expect("growth allocation must succeed");
        black_box(&bytes);
    });

    assert_eq!(stats.allocations, 1);
    assert!(stats.reallocations >= 1);
    assert!(stats.deallocations >= 1);
    assert!(stats.requested_bytes >= 4096);
}

fn preexisting_deallocation_cannot_credit_budget() {
    let preexisting = black_box(Box::new([0_u8; 8192]));
    let violation = without_panic_output(|| {
        catch_unwind(|| {
            let replacement = assert_allocations(
                "replacement after pre-existing free",
                Budget::total_bytes(2047),
                || {
                    drop(preexisting);
                    black_box(Box::new([1_u8; 2048]))
                },
            );
            drop(replacement);
        })
    });
    assert!(
        violation.is_err(),
        "a pre-existing free incorrectly credited the byte budget"
    );
}

fn preexisting_frees_never_hide_requested_bytes() {
    let mut runner = TestRunner::new(Config {
        cases: 128,
        failure_persistence: None,
        ..Config::default()
    });
    runner
        .run(
            &(1_usize..8192, vec(1_usize..8192, 1..48)),
            |(preexisting_size, requested_sizes)| {
                let preexisting = vec![0_u8; preexisting_size].into_boxed_slice();
                let expected_bytes = requested_sizes.iter().try_fold(0_u64, |total, &size| {
                    total.checked_add(u64::try_from(size).unwrap_or(u64::MAX))
                });
                let expected_bytes = expected_bytes.unwrap_or(u64::MAX);

                let ((), stats) = measure(|| {
                    drop(preexisting);
                    for size in &requested_sizes {
                        let allocation = vec![0_u8; *size].into_boxed_slice();
                        black_box(&allocation);
                        drop(allocation);
                    }
                });

                prop_assert!(stats.requested_bytes >= expected_bytes);
                prop_assert!(stats.allocations >= requested_sizes.len() as u64);
                prop_assert_eq!(stats.reallocations, 0);
                Ok(())
            },
        )
        .expect("allocation trace property");
}

fn concurrent_measurement_is_rejected_without_resetting_counters() {
    let (entered_sender, entered_receiver) = mpsc::sync_channel(0);
    let (release_sender, release_receiver) = mpsc::sync_channel(0);
    let first = std::thread::spawn(move || {
        measure(|| {
            let marker = black_box(Box::new([3_u8; 16 * 1024]));
            entered_sender
                .send(())
                .expect("announce active measurement");
            release_receiver.recv().expect("release active measurement");
            marker
        })
    });
    entered_receiver
        .recv()
        .expect("first measurement became active");

    let second = without_panic_output(|| catch_unwind(|| measure(|| ())));
    assert!(second.is_err(), "concurrent measurement must be rejected");

    release_sender
        .send(())
        .expect("release first measurement after rejected contender");
    let (allocation, stats) = first.join().expect("first measurement thread");
    assert!(stats.allocations >= 1);
    assert!(
        stats.requested_bytes >= 16 * KIB,
        "rejected contender reset the active measurement: {stats:?}"
    );
    drop(allocation);
}

fn panicking_measurement_releases_global_meter() {
    let panic = without_panic_output(|| catch_unwind(|| measure(|| panic!("planted panic"))));
    assert!(panic.is_err(), "measured panic must propagate");

    let ((), stats) = measure(|| ());
    assert_eq!(stats.allocations, 0);
    assert_eq!(stats.reallocations, 0);
    assert_eq!(stats.requested_bytes, 0);
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
        Budget::total_bytes(64 * KIB).with_max_calls(64),
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
        Budget::total_bytes(KIB).with_max_calls(1),
        || j2k_profile::format_profile_key_value_fields(&fields),
    )
    .expect("format profile fields");
    assert_eq!(row, " route=scalar result=success");
}

fn precomputed_encode_obeys_ledger_and_allocator_budgets() {
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
        Budget::total_bytes(512 * KIB).with_max_calls(256),
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

fn without_panic_output<R>(operation: impl FnOnce() -> R) -> R {
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = operation();
    std::panic::set_hook(original_hook);
    result
}
