// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
};

use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExperimentRecord {
    schema_version: u32,
    experiment_id: String,
    status: String,
    environment: Environment,
    workloads: Vec<Workload>,
    measurements: Vec<Measurement>,
    decision: Decision,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Environment {
    commit: String,
    branch: String,
    dirty: bool,
    cpu: String,
    gpu: String,
    ram_bytes: u64,
    os: String,
    driver_runtime: String,
    rust_version: String,
    llvm_version: String,
    gpu_toolchain: String,
    build_profile: String,
    feature_flags: Vec<String>,
    #[serde(rename = "environment_variables")]
    variables: BTreeMap<String, String>,
    input_corpus_sha256: String,
    sample_count: u32,
    warm_up_seconds: f64,
    measurement_seconds: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Workload {
    id: String,
    transform: String,
    entropy: String,
    code_block: String,
    image: String,
    batch: u32,
    components: u8,
    output: String,
    operation: String,
    axis_class: String,
    jpeg_sampling: Option<String>,
    jpeg_restart: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Measurement {
    workload_id: String,
    variant: String,
    input_sha256: Option<String>,
    wall_time_ns: f64,
    wall_time_ci_lower_ns: f64,
    wall_time_ci_upper_ns: f64,
    gpu_time_ns: Option<f64>,
    stage_times_ns: BTreeMap<String, f64>,
    dispatch_count: Option<u64>,
    host_to_device_bytes: Option<u64>,
    device_to_host_bytes: Option<u64>,
    device_read_bytes: Option<u64>,
    device_write_bytes: Option<u64>,
    registers_per_thread: Option<u32>,
    private_bytes_per_thread: Option<u64>,
    shared_bytes_per_group: Option<u64>,
    thread_execution_width: Option<u32>,
    max_threads_per_group: Option<u32>,
    code_blocks_per_second: Option<f64>,
    occupancy_percent: Option<f64>,
    spill_loads: Option<u64>,
    spill_stores: Option<u64>,
    cache_observation: Option<String>,
    temporary_float_band_bytes: Option<u64>,
    temporary_float_band_traffic_bytes: Option<u64>,
    cpu_fallback_jobs: Option<u64>,
    resident_dwt_handoffs: Option<u64>,
    ht_codeblock_dispatches: Option<u64>,
    independent_decode_passed: Option<bool>,
    launch_geometries: Option<BTreeMap<String, LaunchGeometry>>,
    checkpoint_count: Option<u64>,
    component_workspace_bytes: Option<u64>,
    coefficient_scratch_bytes: Option<u64>,
    coefficient_scratch_traffic_bytes: Option<u64>,
    output_sha256: String,
    exact_parity: bool,
    conformance: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LaunchGeometry {
    grid: [u32; 3],
    block: [u32; 3],
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Decision {
    priority_workload_id: String,
    confidence_interval_supports_improvement: bool,
    representative_regression_percent: f64,
    complexity_is_proportional: bool,
    rationale: String,
}

pub(crate) fn gpu_experiment(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let Some(command) = args.next() else {
        return Err(usage());
    };
    if command != "validate" {
        return Err(usage());
    }
    let path = PathBuf::from(args.next().ok_or_else(usage)?);
    if args.next().is_some() {
        return Err(usage());
    }
    let source = fs::read_to_string(&path)
        .map_err(|error| format!("read experiment record {}: {error}", path.display()))?;
    let record = serde_json::from_str(&source)
        .map_err(|error| format!("parse experiment record {}: {error}", path.display()))?;
    validate(&record)
}

fn validate(record: &ExperimentRecord) -> Result<(), String> {
    if !matches!(record.schema_version, 1..=3) {
        return Err("unsupported GPU experiment schema version".to_string());
    }
    require_text(&record.experiment_id, "experiment id")?;
    if !matches!(
        record.status.as_str(),
        "measured" | "promoted" | "rejected" | "blocked"
    ) {
        return Err("experiment status is invalid".to_string());
    }
    validate_environment(&record.environment)?;
    if record.workloads.is_empty() {
        return Err("experiment must record at least one workload".to_string());
    }
    let mut workload_ids = BTreeSet::new();
    for workload in &record.workloads {
        validate_workload(workload)?;
        if !workload_ids.insert(workload.id.as_str()) {
            return Err(format!("duplicate workload id {}", workload.id));
        }
    }
    require_text(&record.decision.rationale, "decision rationale")?;
    if !workload_ids.contains(record.decision.priority_workload_id.as_str()) {
        return Err("priority workload id is not in the workload matrix".to_string());
    }
    if !record
        .decision
        .representative_regression_percent
        .is_finite()
        || record.decision.representative_regression_percent < 0.0
    {
        return Err("representative regression percent is invalid".to_string());
    }

    let mut variants_by_workload: MeasurementVariants<'_> = BTreeMap::new();
    for measurement in &record.measurements {
        validate_measurement(measurement, &workload_ids)?;
        if variants_by_workload
            .entry(&measurement.workload_id)
            .or_default()
            .insert(&measurement.variant, measurement)
            .is_some()
        {
            return Err(format!(
                "duplicate {} measurement for {}",
                measurement.variant, measurement.workload_id
            ));
        }
    }
    validate_workload_measurement_pairs(record, &variants_by_workload)?;
    validate_p13_schema_v2(record, &variants_by_workload)?;
    validate_p19_schema_v3(record, &variants_by_workload)?;
    validate_p19_defusion_schema_v3(record, &variants_by_workload)?;
    validate_p6_available_metrics(record, &variants_by_workload)?;
    validate_decision(record, &variants_by_workload)?;
    Ok(())
}

type MeasurementVariants<'a> = BTreeMap<&'a str, BTreeMap<&'a str, &'a Measurement>>;

fn validate_workload_measurement_pairs(
    record: &ExperimentRecord,
    variants_by_workload: &MeasurementVariants<'_>,
) -> Result<(), String> {
    if record.status == "blocked" {
        return Ok(());
    }
    for workload in &record.workloads {
        let Some(variants) = variants_by_workload.get(workload.id.as_str()) else {
            return Err(format!(
                "workload {} requires baseline and treatment measurements",
                workload.id
            ));
        };
        if !variants.contains_key("baseline") || !variants.contains_key("treatment") {
            return Err(format!(
                "workload {} requires baseline and treatment measurements",
                workload.id
            ));
        }
        let baseline = variants["baseline"];
        let treatment = variants["treatment"];
        validate_metric_pair(workload, baseline, treatment)?;
        if !baseline.exact_parity
            || !treatment.exact_parity
            || baseline.output_sha256 != treatment.output_sha256
        {
            return Err(format!(
                "workload {} does not preserve exact parity",
                workload.id
            ));
        }
    }
    Ok(())
}

fn validate_p6_available_metrics(
    record: &ExperimentRecord,
    variants_by_workload: &MeasurementVariants<'_>,
) -> Result<(), String> {
    if record.experiment_id != "P6-METAL-COMPILER-RESOURCE-EVIDENCE" {
        return Ok(());
    }
    for workload in &record.workloads {
        let measurement = variants_by_workload
            .get(workload.id.as_str())
            .and_then(|variants| variants.get("baseline"))
            .ok_or_else(|| {
                format!(
                    "P6 workload {} lacks its available baseline resource measurement",
                    workload.id
                )
            })?;
        if measurement.shared_bytes_per_group.is_none()
            || measurement.thread_execution_width.is_none()
            || measurement.max_threads_per_group.is_none()
            || measurement.code_blocks_per_second.is_none()
        {
            return Err(format!(
                "P6 workload {} omits an available static pipeline or throughput metric",
                workload.id
            ));
        }
    }
    Ok(())
}

const P13_STAGE_WORKLOAD_ID: &str = "cuda-dwt97-resident-preencode-512-b16";
const P13_PRODUCT_WORKLOAD_ID: &str = "cuda-jpeg-to-htj2k-srgb420-512-b16";

fn validate_p13_schema_v2(
    record: &ExperimentRecord,
    variants_by_workload: &MeasurementVariants<'_>,
) -> Result<(), String> {
    const STAGE_KEYS: [&str; 6] = [
        "column_lift",
        "ht_encode",
        "idct_row_lift",
        "pack_upload",
        "quantize_codeblock",
        "readback",
    ];
    const PRODUCT_KEYS: [&str; 5] = [
        "column_lift",
        "idct_row_lift",
        "pack_upload",
        "quantize_codeblock",
        "readback",
    ];

    if record.schema_version != 2 || record.experiment_id != "P13-CUDA-DWT97-COLUMN-QUANTIZE" {
        return Ok(());
    }

    validate_p13_workload_matrix(record)?;

    for (workload_id, expected_stage_keys) in [
        (P13_STAGE_WORKLOAD_ID, STAGE_KEYS.as_slice()),
        (P13_PRODUCT_WORKLOAD_ID, PRODUCT_KEYS.as_slice()),
    ] {
        let variants = variants_by_workload
            .get(workload_id)
            .ok_or_else(|| format!("P13 workload {workload_id} lacks measurements"))?;
        let baseline = variants
            .get("baseline")
            .ok_or_else(|| format!("P13 workload {workload_id} lacks a baseline measurement"))?;
        let treatment = variants
            .get("treatment")
            .ok_or_else(|| format!("P13 workload {workload_id} lacks a treatment measurement"))?;

        validate_p13_stage_keys(workload_id, baseline, expected_stage_keys)?;
        validate_p13_stage_keys(workload_id, treatment, expected_stage_keys)?;
        validate_p13_input_hashes(workload_id, baseline, treatment)?;
        validate_p13_route_and_traffic(workload_id, baseline, treatment)?;
    }

    let stage_variants = variants_by_workload
        .get(P13_STAGE_WORKLOAD_ID)
        .expect("P13 stage variants were validated");
    for measurement in [stage_variants["baseline"], stage_variants["treatment"]] {
        if measurement
            .ht_codeblock_dispatches
            .is_none_or(|count| count == 0)
        {
            return Err(format!(
                "P13 stage {} measurement requires a nonzero HT code-block dispatch count",
                measurement.variant
            ));
        }
    }

    let product_variants = variants_by_workload
        .get(P13_PRODUCT_WORKLOAD_ID)
        .expect("P13 product variants were validated");
    for measurement in [product_variants["baseline"], product_variants["treatment"]] {
        if measurement.cpu_fallback_jobs != Some(0) {
            return Err(format!(
                "P13 product {} measurement must report zero CPU fallback jobs",
                measurement.variant
            ));
        }
        if measurement
            .resident_dwt_handoffs
            .is_none_or(|count| count == 0)
        {
            return Err(format!(
                "P13 product {} measurement requires a resident DWT handoff",
                measurement.variant
            ));
        }
        if measurement.dispatch_count.is_none_or(|count| count == 0) {
            return Err(format!(
                "P13 product {} measurement requires a nonzero dispatch count",
                measurement.variant
            ));
        }
        if measurement.independent_decode_passed != Some(true) {
            return Err(format!(
                "P13 product {} measurement requires an independent decode pass",
                measurement.variant
            ));
        }
    }
    Ok(())
}

fn validate_p13_workload_matrix(record: &ExperimentRecord) -> Result<(), String> {
    let expected_workloads = BTreeSet::from([P13_STAGE_WORKLOAD_ID, P13_PRODUCT_WORKLOAD_ID]);
    let actual_workloads = record
        .workloads
        .iter()
        .map(|workload| workload.id.as_str())
        .collect::<BTreeSet<_>>();
    if actual_workloads != expected_workloads {
        return Err("P13 schema-v2 record requires the exact workload matrix".to_string());
    }
    if record.decision.priority_workload_id != P13_PRODUCT_WORKLOAD_ID {
        return Err("P13 priority workload must be the JPEG-to-HTJ2K product".to_string());
    }
    let stage_workload = record
        .workloads
        .iter()
        .find(|workload| workload.id == P13_STAGE_WORKLOAD_ID)
        .expect("exact P13 workload set contains the stage workload");
    if stage_workload.output != "preencoded_htj2k_codeblocks" {
        return Err("P13 stage output must use preencoded_htj2k_codeblocks vocabulary".to_string());
    }
    Ok(())
}

fn validate_p13_stage_keys(
    workload_id: &str,
    measurement: &Measurement,
    expected: &[&str],
) -> Result<(), String> {
    let actual = measurement
        .stage_times_ns
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(format!(
            "P13 workload {workload_id} {} measurement has incorrect stage-time keys",
            measurement.variant
        ));
    }
    Ok(())
}

fn validate_p13_input_hashes(
    workload_id: &str,
    baseline: &Measurement,
    treatment: &Measurement,
) -> Result<(), String> {
    let baseline_hash = baseline
        .input_sha256
        .as_deref()
        .ok_or_else(|| format!("P13 workload {workload_id} baseline lacks an input SHA-256"))?;
    let treatment_hash = treatment
        .input_sha256
        .as_deref()
        .ok_or_else(|| format!("P13 workload {workload_id} treatment lacks an input SHA-256"))?;
    if baseline_hash != treatment_hash {
        return Err(format!(
            "P13 workload {workload_id} input SHA-256 differs between variants"
        ));
    }
    Ok(())
}

fn validate_p13_route_and_traffic(
    workload_id: &str,
    baseline: &Measurement,
    treatment: &Measurement,
) -> Result<(), String> {
    if !positive_finite(baseline.stage_times_ns["column_lift"])
        || treatment.stage_times_ns["column_lift"] != 0.0
    {
        return Err(format!(
            "P13 workload {workload_id} requires baseline column_lift > 0 and treatment column_lift = 0"
        ));
    }
    if !positive_finite(baseline.stage_times_ns["quantize_codeblock"])
        || !positive_finite(treatment.stage_times_ns["quantize_codeblock"])
    {
        return Err(format!(
            "P13 workload {workload_id} requires positive quantize_codeblock timing"
        ));
    }

    let baseline_bytes = baseline.temporary_float_band_bytes.ok_or_else(|| {
        format!("P13 workload {workload_id} baseline lacks temporary float-band bytes")
    })?;
    let baseline_traffic = baseline.temporary_float_band_traffic_bytes.ok_or_else(|| {
        format!("P13 workload {workload_id} baseline lacks temporary float-band traffic")
    })?;
    if baseline_bytes == 0 || baseline_bytes.checked_mul(2) != Some(baseline_traffic) {
        return Err(format!(
            "P13 workload {workload_id} baseline temporary float-band traffic must be twice its nonzero bytes"
        ));
    }
    if treatment.temporary_float_band_bytes != Some(0)
        || treatment.temporary_float_band_traffic_bytes != Some(0)
    {
        return Err(format!(
            "P13 workload {workload_id} treatment temporary float-band bytes and traffic must be zero"
        ));
    }
    Ok(())
}

const P19_EXPERIMENT_ID: &str = "P19-CUDA-JPEG-PACKED-CHECKPOINTS";
const P19_PRIORITY_WORKLOAD_ID: &str = "ybr420_512x512_batch16_restart_none";
const P19_WORKLOAD_IDS: [&str; 10] = [
    P19_PRIORITY_WORKLOAD_ID,
    "ybr420_512x512_batch1_restart_none",
    "ybr420_512x512_batch16_restart16",
    "ybr420_512x512_batch1_restart16",
    "ybr422_512x512_batch16_restart_none",
    "ybr422_512x512_batch1_restart_none",
    "ybr444_512x512_batch16_restart_none",
    "ybr444_512x512_batch1_restart_none",
    "ybr420_64x64_batch1_restart_none",
    "ybr420_1024x1024_batch1_restart_none",
];
const P19_STAGE_KEYS: [&str; 5] = [
    "resource_upload",
    "fused_decode_kernel",
    "conversion",
    "status_readback",
    "profiled_product_wall",
];
const P19_CHECKPOINT_LAUNCH: &str = "checkpoint_decode";
const P19_PACKED_CHECKPOINT_MIN_COUNT: u32 = 128;
const P19_PACKED_CHECKPOINT_THREADS: u32 = 128;

fn validate_p19_schema_v3(
    record: &ExperimentRecord,
    variants_by_workload: &MeasurementVariants<'_>,
) -> Result<(), String> {
    if record.experiment_id != P19_EXPERIMENT_ID {
        return Ok(());
    }
    if record.schema_version != 3 {
        return Err("P19 packed-checkpoint record requires schema version 3".to_string());
    }

    validate_p19_workload_matrix(record)?;
    for workload_id in P19_WORKLOAD_IDS {
        let variants = variants_by_workload
            .get(workload_id)
            .ok_or_else(|| format!("P19 workload {workload_id} lacks measurements"))?;
        let baseline = variants
            .get("baseline")
            .ok_or_else(|| format!("P19 workload {workload_id} lacks a baseline measurement"))?;
        let treatment = variants
            .get("treatment")
            .ok_or_else(|| format!("P19 workload {workload_id} lacks a treatment measurement"))?;

        validate_p19_stage_keys(workload_id, baseline)?;
        validate_p19_stage_keys(workload_id, treatment)?;
        validate_p19_paired_evidence(workload_id, baseline, treatment)?;
        validate_p19_launch_geometry(workload_id, baseline, treatment)?;
    }
    Ok(())
}

fn validate_p19_workload_matrix(record: &ExperimentRecord) -> Result<(), String> {
    let expected = P19_WORKLOAD_IDS.into_iter().collect::<BTreeSet<_>>();
    let actual = record
        .workloads
        .iter()
        .map(|workload| workload.id.as_str())
        .collect::<BTreeSet<_>>();
    if actual != expected {
        return Err("P19 schema-v3 record requires the exact workload matrix".to_string());
    }
    if record.decision.priority_workload_id != P19_PRIORITY_WORKLOAD_ID {
        return Err(
            "P19 priority workload must be 4:2:0 512x512 batch 16 without restart markers"
                .to_string(),
        );
    }
    Ok(())
}

fn validate_p19_stage_keys(workload_id: &str, measurement: &Measurement) -> Result<(), String> {
    let actual = measurement
        .stage_times_ns
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let expected = P19_STAGE_KEYS.into_iter().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(format!(
            "P19 workload {workload_id} {} measurement has incorrect stage-time keys",
            measurement.variant
        ));
    }
    Ok(())
}

fn validate_p19_paired_evidence(
    workload_id: &str,
    baseline: &Measurement,
    treatment: &Measurement,
) -> Result<(), String> {
    let baseline_hash = baseline
        .input_sha256
        .as_deref()
        .ok_or_else(|| format!("P19 workload {workload_id} baseline lacks an input SHA-256"))?;
    let treatment_hash = treatment
        .input_sha256
        .as_deref()
        .ok_or_else(|| format!("P19 workload {workload_id} treatment lacks an input SHA-256"))?;
    if baseline_hash != treatment_hash {
        return Err(format!(
            "P19 workload {workload_id} input SHA-256 differs between variants"
        ));
    }

    let baseline_checkpoints = baseline
        .checkpoint_count
        .ok_or_else(|| format!("P19 workload {workload_id} baseline lacks checkpoint_count"))?;
    let treatment_checkpoints = treatment
        .checkpoint_count
        .ok_or_else(|| format!("P19 workload {workload_id} treatment lacks checkpoint_count"))?;
    if baseline_checkpoints != treatment_checkpoints {
        return Err(format!(
            "P19 workload {workload_id} checkpoint_count differs between variants"
        ));
    }

    let baseline_workspace = baseline.component_workspace_bytes.ok_or_else(|| {
        format!("P19 workload {workload_id} baseline lacks component_workspace_bytes")
    })?;
    let treatment_workspace = treatment.component_workspace_bytes.ok_or_else(|| {
        format!("P19 workload {workload_id} treatment lacks component_workspace_bytes")
    })?;
    if baseline_workspace != treatment_workspace {
        return Err(format!(
            "P19 workload {workload_id} component_workspace_bytes differs between variants"
        ));
    }

    if baseline.coefficient_scratch_bytes != Some(0)
        || treatment.coefficient_scratch_bytes != Some(0)
    {
        return Err(format!(
            "P19 workload {workload_id} coefficient_scratch_bytes must be zero for both variants"
        ));
    }
    Ok(())
}

fn validate_p19_launch_geometry(
    workload_id: &str,
    baseline: &Measurement,
    treatment: &Measurement,
) -> Result<(), String> {
    let checkpoint_count = baseline
        .checkpoint_count
        .expect("P19 paired evidence validated checkpoint_count");
    let checkpoint_count = u32::try_from(checkpoint_count).map_err(|_| {
        format!("P19 workload {workload_id} checkpoint_count exceeds CUDA grid vocabulary")
    })?;
    let baseline_geometry = p19_checkpoint_geometry(workload_id, baseline)?;
    let treatment_geometry = p19_checkpoint_geometry(workload_id, treatment)?;

    validate_p19_exact_geometry(
        workload_id,
        "baseline",
        baseline_geometry,
        checkpoint_count,
        1,
    )?;
    let (treatment_grid, treatment_block) = if checkpoint_count < P19_PACKED_CHECKPOINT_MIN_COUNT {
        (checkpoint_count, 1)
    } else {
        (
            checkpoint_count.div_ceil(P19_PACKED_CHECKPOINT_THREADS),
            P19_PACKED_CHECKPOINT_THREADS,
        )
    };
    validate_p19_exact_geometry(
        workload_id,
        "treatment",
        treatment_geometry,
        treatment_grid,
        treatment_block,
    )?;
    if checkpoint_count >= P19_PACKED_CHECKPOINT_MIN_COUNT
        && treatment.wall_time_ns > baseline.wall_time_ns
    {
        return Err(format!(
            "P19 workload {workload_id} geometry-changing treatment must not be slower than baseline"
        ));
    }
    Ok(())
}

fn p19_checkpoint_geometry<'a>(
    workload_id: &str,
    measurement: &'a Measurement,
) -> Result<&'a LaunchGeometry, String> {
    let geometries = measurement.launch_geometries.as_ref().ok_or_else(|| {
        format!(
            "P19 workload {workload_id} {} measurement lacks launch_geometries",
            measurement.variant
        )
    })?;
    if geometries.len() != 1 || !geometries.contains_key(P19_CHECKPOINT_LAUNCH) {
        return Err(format!(
            "P19 workload {workload_id} {} launch_geometries must contain only checkpoint_decode",
            measurement.variant
        ));
    }
    Ok(&geometries[P19_CHECKPOINT_LAUNCH])
}

fn validate_p19_exact_geometry(
    workload_id: &str,
    variant: &str,
    geometry: &LaunchGeometry,
    grid_x: u32,
    block_x: u32,
) -> Result<(), String> {
    if geometry.grid != [grid_x, 1, 1] || geometry.block != [block_x, 1, 1] {
        return Err(format!(
            "P19 workload {workload_id} {variant} checkpoint_decode launch geometry is invalid"
        ));
    }
    Ok(())
}

const P19_DEFUSION_EXPERIMENT_ID: &str = "P19-CUDA-JPEG-DECODE-DEFUSION";
const P19_DEFUSION_STAGE_KEYS: [&str; 8] = [
    "resource_upload",
    "coefficient_scratch_clear",
    "entropy_coefficients",
    "idct_deposit",
    "fused_decode_kernel",
    "conversion",
    "status_readback",
    "profiled_product_wall",
];
const P19_IDCT_LAUNCH: &str = "idct_deposit";
const P19_JPEG_420_BLOCKS_PER_MCU: u64 = 6;
const P19_JPEG_COEFFICIENTS_PER_BLOCK: u64 = 64;
const P19_I32_BYTES: u64 = 4;
const P19_IDCT_THREADS: u32 = 128;

fn validate_p19_defusion_schema_v3(
    record: &ExperimentRecord,
    variants_by_workload: &MeasurementVariants<'_>,
) -> Result<(), String> {
    if record.experiment_id != P19_DEFUSION_EXPERIMENT_ID {
        return Ok(());
    }
    if record.schema_version != 3 || record.status != "rejected" {
        return Err(
            "P19 defusion record requires schema version 3 and rejected status".to_string(),
        );
    }
    validate_p19_workload_matrix(record)?;

    for workload in &record.workloads {
        let variants = variants_by_workload
            .get(workload.id.as_str())
            .ok_or_else(|| format!("P19 defusion workload {} lacks measurements", workload.id))?;
        let baseline = variants["baseline"];
        let treatment = variants["treatment"];
        validate_p19_defusion_stage_keys(baseline)?;
        validate_p19_defusion_stage_keys(treatment)?;
        validate_p19_defusion_paired_evidence(workload, baseline, treatment)?;
        validate_p19_defusion_route(workload, baseline, treatment)?;
        validate_p19_defusion_launches(workload, baseline, treatment)?;
    }
    Ok(())
}

fn validate_p19_defusion_stage_keys(measurement: &Measurement) -> Result<(), String> {
    let actual = measurement
        .stage_times_ns
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let expected = P19_DEFUSION_STAGE_KEYS.into_iter().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(format!(
            "P19 defusion {} has incorrect stage-time keys",
            measurement.variant
        ));
    }
    Ok(())
}

fn validate_p19_defusion_paired_evidence(
    workload: &Workload,
    baseline: &Measurement,
    treatment: &Measurement,
) -> Result<(), String> {
    let baseline_hash = baseline.input_sha256.as_deref().ok_or_else(|| {
        format!(
            "P19 defusion workload {} baseline lacks input SHA-256",
            workload.id
        )
    })?;
    if treatment.input_sha256.as_deref() != Some(baseline_hash) {
        return Err(format!(
            "P19 defusion workload {} input SHA-256 differs between variants",
            workload.id
        ));
    }
    if baseline.conformance != treatment.conformance {
        return Err(format!(
            "P19 defusion workload {} conformance evidence differs between variants",
            workload.id
        ));
    }
    if baseline.checkpoint_count.is_none()
        || baseline.checkpoint_count != treatment.checkpoint_count
        || baseline.component_workspace_bytes.is_none()
        || baseline.component_workspace_bytes != treatment.component_workspace_bytes
    {
        return Err(format!(
            "P19 defusion workload {} checkpoint or component workspace differs between variants",
            workload.id
        ));
    }
    Ok(())
}

fn validate_p19_defusion_route(
    workload: &Workload,
    baseline: &Measurement,
    treatment: &Measurement,
) -> Result<(), String> {
    let expected_scratch = p19_expected_coefficient_scratch(workload)?;
    validate_p19_defusion_baseline(workload, baseline)?;
    if treatment.coefficient_scratch_bytes != Some(expected_scratch)
        || treatment.coefficient_scratch_traffic_bytes != expected_scratch.checked_mul(3)
    {
        return Err(format!(
            "P19 defusion workload {} treatment scratch or logical traffic is invalid",
            workload.id
        ));
    }

    let split = expected_scratch != 0;
    for key in [
        "coefficient_scratch_clear",
        "entropy_coefficients",
        "idct_deposit",
    ] {
        let value = treatment.stage_times_ns[key];
        if (split && !positive_finite(value)) || (!split && value != 0.0) {
            return Err(format!(
                "P19 defusion workload {} treatment split-stage timing is invalid",
                workload.id
            ));
        }
    }
    let fused = treatment.stage_times_ns["fused_decode_kernel"];
    if (split && fused != 0.0) || (!split && !positive_finite(fused)) {
        return Err(format!(
            "P19 defusion workload {} treatment fused-stage timing is invalid",
            workload.id
        ));
    }
    Ok(())
}

fn validate_p19_defusion_baseline(
    workload: &Workload,
    baseline: &Measurement,
) -> Result<(), String> {
    if baseline.coefficient_scratch_bytes != Some(0)
        || baseline.coefficient_scratch_traffic_bytes != Some(0)
        || !positive_finite(baseline.stage_times_ns["fused_decode_kernel"])
        || [
            "coefficient_scratch_clear",
            "entropy_coefficients",
            "idct_deposit",
        ]
        .into_iter()
        .any(|key| baseline.stage_times_ns[key] != 0.0)
    {
        return Err(format!(
            "P19 defusion workload {} baseline fused-route evidence is invalid",
            workload.id
        ));
    }
    Ok(())
}

fn p19_expected_coefficient_scratch(workload: &Workload) -> Result<u64, String> {
    if workload.jpeg_sampling.as_deref() != Some("4:2:0") {
        return Ok(0);
    }
    let (width, height) = parse_dimensions(&workload.image).ok_or_else(|| {
        format!(
            "P19 defusion workload {} lacks exact dimensions",
            workload.id
        )
    })?;
    width
        .div_ceil(16)
        .checked_mul(height.div_ceil(16))
        .and_then(|value| value.checked_mul(P19_JPEG_420_BLOCKS_PER_MCU))
        .and_then(|value| value.checked_mul(P19_JPEG_COEFFICIENTS_PER_BLOCK))
        .and_then(|value| value.checked_mul(P19_I32_BYTES))
        .and_then(|value| value.checked_mul(u64::from(workload.batch)))
        .ok_or_else(|| {
            format!(
                "P19 defusion workload {} scratch size overflowed",
                workload.id
            )
        })
}

fn validate_p19_defusion_launches(
    workload: &Workload,
    baseline: &Measurement,
    treatment: &Measurement,
) -> Result<(), String> {
    let checkpoint_count = u32::try_from(
        baseline
            .checkpoint_count
            .expect("P19 defusion paired checkpoint evidence was validated"),
    )
    .map_err(|_| {
        format!(
            "P19 defusion workload {} checkpoint count is too large",
            workload.id
        )
    })?;
    let (checkpoint_grid, checkpoint_block) = if checkpoint_count < P19_PACKED_CHECKPOINT_MIN_COUNT
    {
        (checkpoint_count, 1)
    } else {
        (
            checkpoint_count.div_ceil(P19_PACKED_CHECKPOINT_THREADS),
            P19_PACKED_CHECKPOINT_THREADS,
        )
    };
    validate_p19_defusion_launch_map(workload, baseline, checkpoint_grid, checkpoint_block, false)?;
    validate_p19_defusion_launch_map(
        workload,
        treatment,
        checkpoint_grid,
        checkpoint_block,
        workload.jpeg_sampling.as_deref() == Some("4:2:0"),
    )
}

fn validate_p19_defusion_launch_map(
    workload: &Workload,
    measurement: &Measurement,
    checkpoint_grid: u32,
    checkpoint_block: u32,
    split: bool,
) -> Result<(), String> {
    let geometries = measurement.launch_geometries.as_ref().ok_or_else(|| {
        format!(
            "P19 defusion workload {} {} lacks launch geometries",
            workload.id, measurement.variant
        )
    })?;
    let expected_len = if split { 2 } else { 1 };
    if geometries.len() != expected_len {
        return Err(format!(
            "P19 defusion workload {} {} launch map is incomplete",
            workload.id, measurement.variant
        ));
    }
    let checkpoint = geometries.get(P19_CHECKPOINT_LAUNCH).ok_or_else(|| {
        format!(
            "P19 defusion workload {} lacks checkpoint launch",
            workload.id
        )
    })?;
    validate_p19_exact_geometry(
        &workload.id,
        &measurement.variant,
        checkpoint,
        checkpoint_grid,
        checkpoint_block,
    )?;
    if split {
        let idct = geometries
            .get(P19_IDCT_LAUNCH)
            .ok_or_else(|| format!("P19 defusion workload {} lacks IDCT launch", workload.id))?;
        let scratch_per_tile = measurement
            .coefficient_scratch_bytes
            .expect("P19 defusion treatment scratch was validated")
            / u64::from(workload.batch);
        let block_count = scratch_per_tile / (P19_JPEG_COEFFICIENTS_PER_BLOCK * P19_I32_BYTES);
        let grid_x =
            u32::try_from(block_count.div_ceil(u64::from(P19_IDCT_THREADS))).map_err(|_| {
                format!(
                    "P19 defusion workload {} IDCT grid is too large",
                    workload.id
                )
            })?;
        if idct.grid != [grid_x, 1, 1] || idct.block != [P19_IDCT_THREADS, 1, 1] {
            return Err(format!(
                "P19 defusion workload {} {} idct_deposit launch geometry is invalid",
                workload.id, measurement.variant
            ));
        }
    }
    Ok(())
}

fn validate_decision(
    record: &ExperimentRecord,
    variants_by_workload: &MeasurementVariants<'_>,
) -> Result<(), String> {
    if record.status != "blocked" {
        let priority = variants_by_workload
            .get(record.decision.priority_workload_id.as_str())
            .ok_or_else(|| "priority workload lacks measurements".to_string())?;
        let intervals_support_improvement = priority["treatment"].wall_time_ci_upper_ns
            < priority["baseline"].wall_time_ci_lower_ns;
        if record.decision.confidence_interval_supports_improvement != intervals_support_improvement
        {
            return Err(if intervals_support_improvement {
                "recorded confidence intervals support improvement but the decision says they do not"
                    .to_string()
            } else {
                "recorded confidence intervals do not support improvement".to_string()
            });
        }
    }
    if record.status == "promoted" {
        if !record.decision.confidence_interval_supports_improvement {
            return Err("promoted experiment lacks confidence interval support".to_string());
        }
        if !record.decision.complexity_is_proportional {
            return Err("promoted experiment has disproportionate complexity".to_string());
        }
        if record.decision.representative_regression_percent > 2.0 {
            return Err("promoted experiment has a material representative regression".to_string());
        }
    }
    Ok(())
}

fn validate_environment(environment: &Environment) -> Result<(), String> {
    if !is_hex(&environment.commit, 40) {
        return Err("environment commit is not a full Git SHA".to_string());
    }
    for (value, label) in [
        (&environment.branch, "branch"),
        (&environment.cpu, "CPU"),
        (&environment.gpu, "GPU"),
        (&environment.os, "OS"),
        (&environment.driver_runtime, "driver/runtime"),
        (&environment.rust_version, "Rust version"),
        (&environment.llvm_version, "LLVM version"),
        (&environment.gpu_toolchain, "GPU toolchain"),
        (&environment.build_profile, "build profile"),
    ] {
        require_text(value, label)?;
    }
    let _ = environment.dirty;
    if environment.ram_bytes == 0 {
        return Err("environment RAM must be nonzero".to_string());
    }
    if !is_hex(&environment.input_corpus_sha256, 64) {
        return Err("environment input corpus SHA-256 is invalid".to_string());
    }
    if environment.sample_count < 2
        || !positive_finite(environment.warm_up_seconds)
        || !positive_finite(environment.measurement_seconds)
    {
        return Err("experiment sampling configuration is invalid".to_string());
    }
    for feature in &environment.feature_flags {
        require_text(feature, "feature flag")?;
    }
    for (name, value) in &environment.variables {
        require_text(name, "environment variable name")?;
        require_text(value, "environment variable value")?;
    }
    Ok(())
}

fn validate_workload(workload: &Workload) -> Result<(), String> {
    for (value, label) in [
        (&workload.id, "workload id"),
        (&workload.transform, "transform"),
        (&workload.entropy, "entropy"),
        (&workload.code_block, "code block"),
        (&workload.image, "image"),
        (&workload.output, "output"),
        (&workload.operation, "operation"),
        (&workload.axis_class, "axis class"),
    ] {
        require_text(value, label)?;
    }
    if workload.batch == 0 || workload.components == 0 {
        return Err(format!(
            "workload {} has an empty batch or component count",
            workload.id
        ));
    }
    validate_transform_and_entropy(workload)?;
    validate_workload_dimensions(workload)?;
    validate_output_operation_and_axis(workload)?;
    if workload.jpeg_sampling.as_deref().is_some_and(str::is_empty)
        || workload.jpeg_restart.as_deref().is_some_and(str::is_empty)
    {
        return Err(format!(
            "workload {} has an empty JPEG dimension",
            workload.id
        ));
    }
    Ok(())
}

fn validate_transform_and_entropy(workload: &Workload) -> Result<(), String> {
    require_one_of(
        &workload.transform,
        "transform",
        &[
            "none",
            "reversible_5_3",
            "irreversible_9_7",
            "reversible_and_irreversible",
            "rct_and_ict",
            "jpeg_fdct",
            "jpeg_idct",
        ],
    )?;
    require_one_of(
        &workload.entropy,
        "entropy",
        &[
            "none",
            "ht",
            "classic",
            "prequantized_ht",
            "ht_decode",
            "ht_encode",
            "classic_decode",
            "classic_encode",
            "huffman",
        ],
    )
}

fn validate_workload_dimensions(workload: &Workload) -> Result<(), String> {
    if workload.code_block != "none" && parse_dimensions(&workload.code_block).is_none() {
        return Err(format!(
            "workload {} has invalid code block vocabulary",
            workload.id
        ));
    }
    if parse_dimensions(&workload.image).is_none()
        && !matches!(
            workload.image.as_str(),
            "code_block"
                | "representative_matrix"
                | "rgb_matrix"
                | "rgb8_matrix"
                | "jpeg_matrix"
                | "2592-wide"
        )
    {
        return Err(format!(
            "workload {} has invalid image vocabulary",
            workload.id
        ));
    }
    Ok(())
}

fn validate_output_operation_and_axis(workload: &Workload) -> Result<(), String> {
    require_one_of(
        &workload.output,
        "output",
        &[
            "rgb8",
            "rgba8",
            "native",
            "rgb8_host",
            "rgb8_resident",
            "j2k_codestream",
            "htj2k_codestream",
            "f32_subbands_host",
            "prequantized_codeblocks",
            "preencoded_htj2k_codeblocks",
            "coefficients",
            "coded_block",
            "component_planes",
            "transformed_coefficients",
            "decoded_samples",
            "jpeg_codestream",
            "rgb8_or_texture",
        ],
    )?;
    require_one_of(
        &workload.operation,
        "operation",
        &[
            "full",
            "full_encode",
            "roi",
            "half_scale",
            "transform_stage",
            "transcode_transform_quantize",
            "kernel_resource_profile",
            "cuda_column_lift_ab",
            "cuda_wide_idwt",
            "cuda_fdwt_staging",
            "cuda_input_fusion",
            "cuda_final_store",
            "cuda_jpeg_encode_pipeline",
            "metal_jpeg_encode_pipeline",
            "cuda_jpeg_decode_defusion",
            "metal_jpeg_decode_defusion",
        ],
    )?;
    require_one_of(
        &workload.axis_class,
        "axis class",
        &[
            "below_512",
            "at_512",
            "above_512",
            "below_1024",
            "at_1024",
            "above_1024",
            "wide",
            "code_block",
            "matrix",
        ],
    )
}

fn validate_measurement(
    measurement: &Measurement,
    workload_ids: &BTreeSet<&str>,
) -> Result<(), String> {
    if !workload_ids.contains(measurement.workload_id.as_str()) {
        return Err(format!(
            "measurement references unknown workload {}",
            measurement.workload_id
        ));
    }
    if !matches!(measurement.variant.as_str(), "baseline" | "treatment") {
        return Err("measurement variant must be baseline or treatment".to_string());
    }
    if !positive_finite(measurement.wall_time_ns) {
        return Err("measurement wall time must be positive and finite".to_string());
    }
    if !positive_finite(measurement.wall_time_ci_lower_ns)
        || !positive_finite(measurement.wall_time_ci_upper_ns)
        || measurement.wall_time_ci_lower_ns > measurement.wall_time_ns
        || measurement.wall_time_ns > measurement.wall_time_ci_upper_ns
    {
        return Err("measurement wall-time confidence interval is invalid".to_string());
    }
    for value in measurement
        .gpu_time_ns
        .iter()
        .chain(measurement.stage_times_ns.values())
    {
        if !value.is_finite() || *value < 0.0 {
            return Err("GPU or stage time is invalid".to_string());
        }
    }
    if let Some(value) = measurement.occupancy_percent {
        if !value.is_finite() || !(0.0..=100.0).contains(&value) {
            return Err("occupancy percent is invalid".to_string());
        }
    }
    if let Some(value) = measurement.code_blocks_per_second {
        if !positive_finite(value) {
            return Err("code-block throughput must be positive and finite".to_string());
        }
    }
    if let Some(width) = measurement.thread_execution_width {
        if width == 0 {
            return Err("thread execution width must be nonzero".to_string());
        }
    }
    if let Some(max_threads) = measurement.max_threads_per_group {
        if max_threads == 0
            || measurement
                .thread_execution_width
                .is_some_and(|width| max_threads < width)
        {
            return Err("maximum threads per group is invalid".to_string());
        }
    }
    let _ = (
        measurement.temporary_float_band_bytes,
        measurement.temporary_float_band_traffic_bytes,
        measurement.cpu_fallback_jobs,
        measurement.resident_dwt_handoffs,
        measurement.ht_codeblock_dispatches,
        measurement.independent_decode_passed,
        measurement.dispatch_count,
        measurement.host_to_device_bytes,
        measurement.device_to_host_bytes,
        measurement.device_read_bytes,
        measurement.device_write_bytes,
        measurement.registers_per_thread,
        measurement.private_bytes_per_thread,
        measurement.shared_bytes_per_group,
        measurement.thread_execution_width,
        measurement.max_threads_per_group,
        measurement.spill_loads,
        measurement.spill_stores,
        measurement.component_workspace_bytes,
        measurement.coefficient_scratch_bytes,
        measurement.coefficient_scratch_traffic_bytes,
    );
    validate_launch_evidence(measurement)?;
    if measurement
        .cache_observation
        .as_deref()
        .is_some_and(str::is_empty)
    {
        return Err("cache observation must not be empty".to_string());
    }
    if !is_hex(&measurement.output_sha256, 64) {
        return Err("measurement output SHA-256 is invalid".to_string());
    }
    if measurement
        .input_sha256
        .as_deref()
        .is_some_and(|hash| !is_hex(hash, 64))
    {
        return Err("measurement input SHA-256 is invalid".to_string());
    }
    require_text(&measurement.conformance, "conformance result")?;
    let conformance = measurement.conformance.to_ascii_lowercase();
    if conformance.contains("fail") || !conformance.contains("pass") {
        return Err("conformance result must report pass".to_string());
    }
    Ok(())
}

fn validate_launch_evidence(measurement: &Measurement) -> Result<(), String> {
    if measurement.checkpoint_count == Some(0) {
        return Err("checkpoint count must be nonzero when recorded".to_string());
    }
    let Some(geometries) = &measurement.launch_geometries else {
        return Ok(());
    };
    if geometries.is_empty() {
        return Err("launch geometries must not be empty when recorded".to_string());
    }
    for (name, geometry) in geometries {
        require_text(name, "launch geometry name")?;
        if geometry.grid.contains(&0) || geometry.block.contains(&0) {
            return Err(format!(
                "launch geometry {name} must use nonzero grid and block dimensions"
            ));
        }
    }
    Ok(())
}

fn validate_metric_pair(
    workload: &Workload,
    baseline: &Measurement,
    treatment: &Measurement,
) -> Result<(), String> {
    macro_rules! require_same_availability {
        ($field:ident) => {
            if baseline.$field.is_some() != treatment.$field.is_some() {
                return Err(format!(
                    "workload {} {} availability differs between baseline and treatment",
                    workload.id,
                    stringify!($field)
                ));
            }
        };
    }
    require_same_availability!(gpu_time_ns);
    require_same_availability!(input_sha256);
    require_same_availability!(dispatch_count);
    require_same_availability!(host_to_device_bytes);
    require_same_availability!(device_to_host_bytes);
    require_same_availability!(device_read_bytes);
    require_same_availability!(device_write_bytes);
    require_same_availability!(registers_per_thread);
    require_same_availability!(private_bytes_per_thread);
    require_same_availability!(shared_bytes_per_group);
    require_same_availability!(thread_execution_width);
    require_same_availability!(max_threads_per_group);
    require_same_availability!(code_blocks_per_second);
    require_same_availability!(occupancy_percent);
    require_same_availability!(spill_loads);
    require_same_availability!(spill_stores);
    require_same_availability!(cache_observation);
    require_same_availability!(temporary_float_band_bytes);
    require_same_availability!(temporary_float_band_traffic_bytes);
    require_same_availability!(cpu_fallback_jobs);
    require_same_availability!(resident_dwt_handoffs);
    require_same_availability!(ht_codeblock_dispatches);
    require_same_availability!(independent_decode_passed);
    require_same_availability!(launch_geometries);
    require_same_availability!(checkpoint_count);
    require_same_availability!(component_workspace_bytes);
    require_same_availability!(coefficient_scratch_bytes);
    require_same_availability!(coefficient_scratch_traffic_bytes);

    let baseline_stages = baseline.stage_times_ns.keys().collect::<BTreeSet<_>>();
    let treatment_stages = treatment.stage_times_ns.keys().collect::<BTreeSet<_>>();
    if baseline_stages != treatment_stages {
        return Err(format!(
            "workload {} stage-time availability differs between baseline and treatment",
            workload.id
        ));
    }
    Ok(())
}

fn require_one_of(value: &str, label: &str, allowed: &[&str]) -> Result<(), String> {
    if allowed.contains(&value) {
        Ok(())
    } else {
        Err(format!("{label} has unsupported vocabulary value {value}"))
    }
}

fn parse_dimensions(value: &str) -> Option<(u64, u64)> {
    let (width, height) = value.split_once('x')?;
    let width = width.parse::<u64>().ok()?;
    let height = height.parse::<u64>().ok()?;
    (width > 0 && height > 0).then_some((width, height))
}

fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn require_text(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{label} must not be empty"))
    } else {
        Ok(())
    }
}

fn is_hex(value: &str, len: usize) -> bool {
    value.len() == len && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn usage() -> String {
    "usage: cargo xtask gpu-experiment validate RECORD.json".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record() -> ExperimentRecord {
        serde_json::from_str(
            r#"{
              "schema_version": 1,
              "experiment_id": "P1-IDWT53-METAL",
              "status": "rejected",
              "environment": {
                "commit": "0123456789abcdef0123456789abcdef01234567",
                "branch": "main",
                "dirty": true,
                "cpu": "Apple M4 Pro",
                "gpu": "Apple M4 Pro 16-core GPU",
                "ram_bytes": 51539607552,
                "os": "macOS 26.5.2",
                "driver_runtime": "Metal 32023.883",
                "rust_version": "rustc 1.96.0",
                "llvm_version": "LLVM 22",
                "gpu_toolchain": "metal 32023.883",
                "build_profile": "release-bench",
                "feature_flags": ["metal"],
                "environment_variables": {"J2K_METAL_PROFILE_STAGES":"summary"},
                "input_corpus_sha256": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
                "sample_count": 10,
                "warm_up_seconds": 1.0,
                "measurement_seconds": 3.0
              },
              "workloads": [{
                "id":"rgb-1024-b16","transform":"reversible_5_3","entropy":"ht",
                "code_block":"64x64","image":"1024x1024","batch":16,"components":3,
                "output":"rgb8","operation":"full","axis_class":"above_1024",
                "jpeg_sampling":null,"jpeg_restart":null
              }],
              "measurements": [
                {"workload_id":"rgb-1024-b16","variant":"baseline","wall_time_ns":100.0,"wall_time_ci_lower_ns":99.0,"wall_time_ci_upper_ns":101.0,
                 "gpu_time_ns":80.0,"stage_times_ns":{"idwt":50.0},"dispatch_count":9,
                 "host_to_device_bytes":1,"device_to_host_bytes":1,"device_read_bytes":2,
                 "device_write_bytes":2,"registers_per_thread":12,"private_bytes_per_thread":0,
                 "shared_bytes_per_group":1024,"occupancy_percent":75.0,"spill_loads":0,
                 "spill_stores":0,"cache_observation":"not captured","output_sha256":"abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
                 "exact_parity":true,"conformance":"pass"},
                {"workload_id":"rgb-1024-b16","variant":"treatment","wall_time_ns":95.0,"wall_time_ci_lower_ns":94.0,"wall_time_ci_upper_ns":96.0,
                 "gpu_time_ns":75.0,"stage_times_ns":{"idwt":45.0},"dispatch_count":6,
                 "host_to_device_bytes":1,"device_to_host_bytes":1,"device_read_bytes":2,
                 "device_write_bytes":2,"registers_per_thread":14,"private_bytes_per_thread":0,
                 "shared_bytes_per_group":2048,"occupancy_percent":70.0,"spill_loads":0,
                 "spill_stores":0,"cache_observation":"not captured","output_sha256":"abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
                 "exact_parity":true,"conformance":"pass"}
              ],
              "decision":{"priority_workload_id":"rgb-1024-b16","confidence_interval_supports_improvement":true,
                "representative_regression_percent":0.0,"complexity_is_proportional":false,
                "rationale":"No end-to-end confidence-interval win."}
            }"#,
        )
        .expect("experiment fixture")
    }

    fn p13_record() -> ExperimentRecord {
        let mut record = record();
        record.schema_version = 2;
        record.experiment_id = "P13-CUDA-DWT97-COLUMN-QUANTIZE".to_string();
        record.status = "measured".to_string();

        let mut stage_workload = record.workloads[0].clone();
        stage_workload.id = "cuda-dwt97-resident-preencode-512-b16".to_string();
        stage_workload.transform = "irreversible_9_7".to_string();
        stage_workload.entropy = "prequantized_ht".to_string();
        stage_workload.image = "512x512".to_string();
        stage_workload.batch = 16;
        stage_workload.components = 1;
        stage_workload.output = "preencoded_htj2k_codeblocks".to_string();
        stage_workload.operation = "cuda_column_lift_ab".to_string();
        stage_workload.axis_class = "at_512".to_string();

        let mut product_workload = stage_workload.clone();
        product_workload.id = "cuda-jpeg-to-htj2k-srgb420-512-b16".to_string();
        product_workload.components = 3;
        product_workload.output = "htj2k_codestream".to_string();
        product_workload.jpeg_sampling = Some("ybr420".to_string());
        record.workloads = vec![stage_workload, product_workload];

        let input_sha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let mut stage_baseline = record.measurements[0].clone();
        stage_baseline.workload_id = "cuda-dwt97-resident-preencode-512-b16".to_string();
        stage_baseline.input_sha256 = Some(input_sha.to_string());
        stage_baseline.stage_times_ns = BTreeMap::from([
            ("column_lift".to_string(), 10.0),
            ("ht_encode".to_string(), 30.0),
            ("idct_row_lift".to_string(), 10.0),
            ("pack_upload".to_string(), 10.0),
            ("quantize_codeblock".to_string(), 20.0),
            ("readback".to_string(), 10.0),
        ]);
        stage_baseline.temporary_float_band_bytes = Some(100);
        stage_baseline.temporary_float_band_traffic_bytes = Some(200);
        stage_baseline.ht_codeblock_dispatches = Some(64);

        let mut stage_treatment = stage_baseline.clone();
        stage_treatment.variant = "treatment".to_string();
        stage_treatment.wall_time_ns = 95.0;
        stage_treatment.wall_time_ci_lower_ns = 94.0;
        stage_treatment.wall_time_ci_upper_ns = 96.0;
        stage_treatment
            .stage_times_ns
            .insert("column_lift".to_string(), 0.0);
        stage_treatment.temporary_float_band_bytes = Some(0);
        stage_treatment.temporary_float_band_traffic_bytes = Some(0);

        let mut product_baseline = stage_baseline.clone();
        product_baseline.workload_id = "cuda-jpeg-to-htj2k-srgb420-512-b16".to_string();
        product_baseline.stage_times_ns.remove("ht_encode");
        product_baseline.ht_codeblock_dispatches = None;
        product_baseline.cpu_fallback_jobs = Some(0);
        product_baseline.resident_dwt_handoffs = Some(16);
        product_baseline.independent_decode_passed = Some(true);

        let mut product_treatment = product_baseline.clone();
        product_treatment.variant = "treatment".to_string();
        product_treatment.wall_time_ns = 95.0;
        product_treatment.wall_time_ci_lower_ns = 94.0;
        product_treatment.wall_time_ci_upper_ns = 96.0;
        product_treatment
            .stage_times_ns
            .insert("column_lift".to_string(), 0.0);
        product_treatment.temporary_float_band_bytes = Some(0);
        product_treatment.temporary_float_band_traffic_bytes = Some(0);

        record.measurements = vec![
            stage_baseline,
            stage_treatment,
            product_baseline,
            product_treatment,
        ];
        record.decision.priority_workload_id = "cuda-jpeg-to-htj2k-srgb420-512-b16".to_string();
        record
    }

    fn p19_record() -> ExperimentRecord {
        let mut record = record();
        record.schema_version = 3;
        record.experiment_id = P19_EXPERIMENT_ID.to_string();
        record.status = "measured".to_string();

        record.workloads = P19_WORKLOAD_IDS
            .into_iter()
            .map(|id| {
                let mut workload = record.workloads[0].clone();
                workload.id = id.to_string();
                workload.transform = "jpeg_idct".to_string();
                workload.entropy = "huffman".to_string();
                workload.code_block = "8x8".to_string();
                workload.components = 3;
                workload.output = "rgb8".to_string();
                workload.operation = "cuda_jpeg_decode_defusion".to_string();
                workload.jpeg_sampling = Some(id[3..6].to_string());
                workload.jpeg_restart = Some(if id.ends_with("restart16") {
                    "16".to_string()
                } else {
                    "none".to_string()
                });
                if id.contains("64x64") {
                    workload.image = "64x64".to_string();
                    workload.axis_class = "below_512".to_string();
                } else if id.contains("1024x1024") {
                    workload.image = "1024x1024".to_string();
                    workload.axis_class = "at_1024".to_string();
                } else {
                    workload.image = "512x512".to_string();
                    workload.axis_class = "at_512".to_string();
                }
                workload.batch = if id.contains("batch16") { 16 } else { 1 };
                workload
            })
            .collect();

        let input_sha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let mut measurements = Vec::new();
        for workload in &record.workloads {
            let checkpoint_count = if workload.image == "64x64" { 16 } else { 128 };
            let workspace_bytes = if workload.jpeg_sampling.as_deref() == Some("444") {
                0
            } else {
                786_432 * u64::from(workload.batch)
            };
            let stages = BTreeMap::from([
                ("resource_upload".to_string(), 10.0),
                ("fused_decode_kernel".to_string(), 60.0),
                ("conversion".to_string(), 10.0),
                ("status_readback".to_string(), 5.0),
                ("profiled_product_wall".to_string(), 90.0),
            ]);

            let mut baseline = record.measurements[0].clone();
            baseline.workload_id.clone_from(&workload.id);
            baseline.input_sha256 = Some(input_sha.to_string());
            baseline.stage_times_ns = stages.clone();
            baseline.launch_geometries = Some(BTreeMap::from([(
                P19_CHECKPOINT_LAUNCH.to_string(),
                LaunchGeometry {
                    grid: [checkpoint_count, 1, 1],
                    block: [1, 1, 1],
                },
            )]));
            baseline.checkpoint_count = Some(u64::from(checkpoint_count));
            baseline.component_workspace_bytes = Some(workspace_bytes);
            baseline.coefficient_scratch_bytes = Some(0);

            let mut treatment = record.measurements[1].clone();
            treatment.workload_id.clone_from(&workload.id);
            treatment.input_sha256 = Some(input_sha.to_string());
            treatment.stage_times_ns = stages;
            let (grid_x, block_x) = if checkpoint_count < P19_PACKED_CHECKPOINT_MIN_COUNT {
                (checkpoint_count, 1)
            } else {
                (
                    checkpoint_count.div_ceil(P19_PACKED_CHECKPOINT_THREADS),
                    P19_PACKED_CHECKPOINT_THREADS,
                )
            };
            treatment.launch_geometries = Some(BTreeMap::from([(
                P19_CHECKPOINT_LAUNCH.to_string(),
                LaunchGeometry {
                    grid: [grid_x, 1, 1],
                    block: [block_x, 1, 1],
                },
            )]));
            treatment.checkpoint_count = Some(u64::from(checkpoint_count));
            treatment.component_workspace_bytes = Some(workspace_bytes);
            treatment.coefficient_scratch_bytes = Some(0);
            measurements.extend([baseline, treatment]);
        }
        record.measurements = measurements;
        record.decision.priority_workload_id = P19_PRIORITY_WORKLOAD_ID.to_string();
        record
    }

    fn p19_defusion_record() -> ExperimentRecord {
        let mut record = p19_record();
        record.experiment_id = P19_DEFUSION_EXPERIMENT_ID.to_string();
        record.status = "rejected".to_string();
        record.decision.confidence_interval_supports_improvement = false;

        for index in 0..record.workloads.len() {
            let workload = &mut record.workloads[index];
            workload.jpeg_sampling = Some(
                if workload.id.starts_with("ybr420") {
                    "4:2:0"
                } else if workload.id.starts_with("ybr422") {
                    "4:2:2"
                } else {
                    "4:4:4"
                }
                .to_string(),
            );
            let expected_scratch =
                p19_expected_coefficient_scratch(workload).expect("P19 defusion fixture scratch");
            let [baseline, treatment] = &mut record.measurements[index * 2..index * 2 + 2] else {
                unreachable!("P19 defusion fixture has paired measurements")
            };
            baseline.stage_times_ns = p19_defusion_fixture_stages(false, 90.0);
            baseline.coefficient_scratch_bytes = Some(0);
            baseline.coefficient_scratch_traffic_bytes = Some(0);
            let checkpoint_count = u32::try_from(
                baseline
                    .checkpoint_count
                    .expect("P19 defusion fixture checkpoints"),
            )
            .expect("fixture checkpoint count");
            let (checkpoint_grid, checkpoint_block) =
                if checkpoint_count < P19_PACKED_CHECKPOINT_MIN_COUNT {
                    (checkpoint_count, 1)
                } else {
                    (
                        checkpoint_count.div_ceil(P19_PACKED_CHECKPOINT_THREADS),
                        P19_PACKED_CHECKPOINT_THREADS,
                    )
                };
            baseline.launch_geometries = Some(BTreeMap::from([(
                P19_CHECKPOINT_LAUNCH.to_string(),
                LaunchGeometry {
                    grid: [checkpoint_grid, 1, 1],
                    block: [checkpoint_block, 1, 1],
                },
            )]));

            treatment.wall_time_ns = 102.0;
            treatment.wall_time_ci_lower_ns = 101.0;
            treatment.wall_time_ci_upper_ns = 103.0;
            treatment.conformance.clone_from(&baseline.conformance);
            treatment.stage_times_ns = p19_defusion_fixture_stages(expected_scratch != 0, 95.0);
            treatment.coefficient_scratch_bytes = Some(expected_scratch);
            treatment.coefficient_scratch_traffic_bytes = expected_scratch.checked_mul(3);
            let mut geometries = BTreeMap::from([(
                P19_CHECKPOINT_LAUNCH.to_string(),
                LaunchGeometry {
                    grid: [checkpoint_grid, 1, 1],
                    block: [checkpoint_block, 1, 1],
                },
            )]);
            if expected_scratch != 0 {
                let scratch_per_tile = expected_scratch / u64::from(workload.batch);
                let block_count =
                    scratch_per_tile / (P19_JPEG_COEFFICIENTS_PER_BLOCK * P19_I32_BYTES);
                geometries.insert(
                    P19_IDCT_LAUNCH.to_string(),
                    LaunchGeometry {
                        grid: [
                            u32::try_from(block_count.div_ceil(u64::from(P19_IDCT_THREADS)))
                                .expect("fixture IDCT grid"),
                            1,
                            1,
                        ],
                        block: [P19_IDCT_THREADS, 1, 1],
                    },
                );
            }
            treatment.launch_geometries = Some(geometries);
        }
        record
    }

    fn p19_defusion_fixture_stages(split: bool, product_wall: f64) -> BTreeMap<String, f64> {
        BTreeMap::from([
            ("resource_upload".to_string(), 10.0),
            (
                "coefficient_scratch_clear".to_string(),
                if split { 5.0 } else { 0.0 },
            ),
            (
                "entropy_coefficients".to_string(),
                if split { 30.0 } else { 0.0 },
            ),
            ("idct_deposit".to_string(), if split { 20.0 } else { 0.0 }),
            (
                "fused_decode_kernel".to_string(),
                if split { 0.0 } else { 60.0 },
            ),
            ("conversion".to_string(), 10.0),
            ("status_readback".to_string(), 5.0),
            ("profiled_product_wall".to_string(), product_wall),
        ])
    }

    #[test]
    fn complete_record_is_accepted() {
        validate(&record()).expect("valid experiment record");
    }

    #[test]
    fn schema_v3_keeps_new_measurement_evidence_optional() {
        let mut record = record();
        record.schema_version = 3;
        validate(&record).expect("generic schema-v3 record with no P19-only metrics");
    }

    #[test]
    fn launch_geometry_uses_three_dimensional_grid_and_block_arrays() {
        let geometry: LaunchGeometry =
            serde_json::from_str(r#"{"grid":[2,1,1],"block":[128,1,1]}"#)
                .expect("launch geometry array vocabulary");
        assert_eq!(geometry.grid, [2, 1, 1]);
        assert_eq!(geometry.block, [128, 1, 1]);
    }

    #[test]
    fn malformed_environment_identity_is_rejected() {
        let mut record = record();
        record.environment.input_corpus_sha256 = "unknown".to_string();
        assert!(validate(&record).unwrap_err().contains("corpus SHA-256"));
    }

    #[test]
    fn decisions_require_baseline_treatment_and_exact_parity() {
        let mut missing_treatment = record();
        missing_treatment.measurements.pop();
        assert!(validate(&missing_treatment)
            .unwrap_err()
            .contains("baseline and treatment"));

        let mut parity_failure = record();
        parity_failure.measurements[1].exact_parity = false;
        assert!(validate(&parity_failure)
            .unwrap_err()
            .contains("exact parity"));
    }

    #[test]
    fn promoted_experiment_must_meet_acceptance_policy() {
        let mut record = record();
        record.status = "promoted".to_string();
        record.decision.confidence_interval_supports_improvement = false;
        assert!(validate(&record)
            .unwrap_err()
            .contains("confidence interval"));
    }

    #[test]
    fn promoted_confidence_claim_is_derived_from_recorded_intervals() {
        let mut record = record();
        record.status = "promoted".to_string();
        record.decision.confidence_interval_supports_improvement = true;
        record.decision.complexity_is_proportional = true;
        record.measurements[1].wall_time_ns = 105.0;
        record.measurements[1].wall_time_ci_lower_ns = 104.0;
        record.measurements[1].wall_time_ci_upper_ns = 106.0;

        assert!(validate(&record)
            .unwrap_err()
            .contains("recorded confidence intervals do not support improvement"));
    }

    #[test]
    fn promoted_record_uses_repository_two_percent_regression_limit() {
        let mut record = record();
        record.status = "promoted".to_string();
        record.decision.confidence_interval_supports_improvement = true;
        record.decision.complexity_is_proportional = true;
        record.decision.representative_regression_percent = 2.01;

        assert!(validate(&record)
            .unwrap_err()
            .contains("material representative regression"));
    }

    #[test]
    fn workload_vocabulary_and_conformance_are_fail_closed() {
        let mut invalid_workload = record();
        invalid_workload.workloads[0].code_block = "nonsense".to_string();
        assert!(validate(&invalid_workload)
            .unwrap_err()
            .contains("code block"));

        let mut failed_conformance = record();
        failed_conformance.measurements[1].conformance = "FAILED".to_string();
        assert!(validate(&failed_conformance)
            .unwrap_err()
            .contains("conformance result must report pass"));
    }

    #[test]
    fn available_metrics_must_be_paired_across_variants() {
        let mut missing_dispatch = record();
        missing_dispatch.measurements[1].dispatch_count = None;
        assert!(validate(&missing_dispatch)
            .unwrap_err()
            .contains("dispatch_count availability differs"));

        let mut different_stages = record();
        different_stages.measurements[1].stage_times_ns = BTreeMap::new();
        assert!(validate(&different_stages)
            .unwrap_err()
            .contains("stage-time availability differs"));
    }

    #[test]
    fn new_optional_metrics_must_be_valid_and_paired_across_variants() {
        macro_rules! missing_treatment_metric_is_rejected {
            ($field:ident, $value:expr) => {{
                let mut record = record();
                record.measurements[0].$field = Some($value);
                assert!(validate(&record)
                    .unwrap_err()
                    .contains(concat!(stringify!($field), " availability differs")));
            }};
        }

        missing_treatment_metric_is_rejected!(input_sha256, "a".repeat(64));
        missing_treatment_metric_is_rejected!(temporary_float_band_bytes, 1);
        missing_treatment_metric_is_rejected!(temporary_float_band_traffic_bytes, 2);
        missing_treatment_metric_is_rejected!(cpu_fallback_jobs, 0);
        missing_treatment_metric_is_rejected!(resident_dwt_handoffs, 1);
        missing_treatment_metric_is_rejected!(ht_codeblock_dispatches, 1);
        missing_treatment_metric_is_rejected!(independent_decode_passed, true);
        missing_treatment_metric_is_rejected!(checkpoint_count, 1);
        missing_treatment_metric_is_rejected!(component_workspace_bytes, 0);
        missing_treatment_metric_is_rejected!(coefficient_scratch_bytes, 0);
        missing_treatment_metric_is_rejected!(coefficient_scratch_traffic_bytes, 0);

        let mut launch_geometry = record();
        launch_geometry.measurements[0].launch_geometries = Some(BTreeMap::from([(
            "decode".to_string(),
            LaunchGeometry {
                grid: [1, 1, 1],
                block: [1, 1, 1],
            },
        )]));
        assert!(validate(&launch_geometry)
            .unwrap_err()
            .contains("launch_geometries availability differs"));

        let mut zero_geometry = record();
        for measurement in &mut zero_geometry.measurements {
            measurement.launch_geometries = Some(BTreeMap::from([(
                "decode".to_string(),
                LaunchGeometry {
                    grid: [0, 1, 1],
                    block: [1, 1, 1],
                },
            )]));
        }
        assert!(validate(&zero_geometry)
            .unwrap_err()
            .contains("must use nonzero grid and block dimensions"));

        let mut invalid_hash = record();
        invalid_hash.measurements[0].input_sha256 = Some("invalid".to_string());
        invalid_hash.measurements[1].input_sha256 = Some("invalid".to_string());
        assert!(validate(&invalid_hash)
            .unwrap_err()
            .contains("input SHA-256 is invalid"));
    }

    #[test]
    fn p13_schema_v2_complete_record_is_accepted() {
        validate(&p13_record()).expect("valid P13 schema-v2 record");
    }

    #[test]
    fn p19_schema_v3_requires_its_exact_workload_matrix() {
        let mut record = record();
        record.schema_version = 3;
        record.experiment_id = P19_EXPERIMENT_ID.to_string();

        assert!(validate(&record)
            .unwrap_err()
            .contains("P19 schema-v3 record requires the exact workload matrix"));
    }

    #[test]
    fn p19_schema_v3_complete_record_is_accepted() {
        validate(&p19_record()).expect("valid P19 schema-v3 record");
    }

    #[test]
    fn p19_schema_v3_requires_version_priority_and_stage_contract() {
        let mut wrong_version = p19_record();
        wrong_version.schema_version = 2;
        assert!(validate(&wrong_version)
            .unwrap_err()
            .contains("requires schema version 3"));

        let mut wrong_priority = p19_record();
        wrong_priority.decision.priority_workload_id = P19_WORKLOAD_IDS[1].to_string();
        assert!(validate(&wrong_priority)
            .unwrap_err()
            .contains("P19 priority workload"));

        let mut wrong_stages = p19_record();
        wrong_stages.measurements[0]
            .stage_times_ns
            .remove("profiled_product_wall");
        wrong_stages.measurements[1]
            .stage_times_ns
            .remove("profiled_product_wall");
        assert!(validate(&wrong_stages)
            .unwrap_err()
            .contains("incorrect stage-time keys"));
    }

    #[test]
    fn p19_schema_v3_requires_paired_hash_checkpoint_workspace_and_zero_scratch() {
        let mut different_input = p19_record();
        different_input.measurements[1].input_sha256 = Some("b".repeat(64));
        assert!(validate(&different_input)
            .unwrap_err()
            .contains("input SHA-256 differs"));

        let mut different_checkpoints = p19_record();
        different_checkpoints.measurements[1].checkpoint_count = Some(129);
        assert!(validate(&different_checkpoints)
            .unwrap_err()
            .contains("checkpoint_count differs"));

        let mut different_workspace = p19_record();
        different_workspace.measurements[1].component_workspace_bytes = Some(1);
        assert!(validate(&different_workspace)
            .unwrap_err()
            .contains("component_workspace_bytes differs"));

        let mut scratch = p19_record();
        scratch.measurements[1].coefficient_scratch_bytes = Some(1);
        assert!(validate(&scratch)
            .unwrap_err()
            .contains("coefficient_scratch_bytes must be zero"));
    }

    #[test]
    fn p19_schema_v3_checks_serial_and_adaptive_checkpoint_launches() {
        let mut wrong_baseline = p19_record();
        wrong_baseline.measurements[0]
            .launch_geometries
            .as_mut()
            .expect("P19 launch evidence")
            .get_mut(P19_CHECKPOINT_LAUNCH)
            .expect("checkpoint launch")
            .block = [128, 1, 1];
        assert!(validate(&wrong_baseline)
            .unwrap_err()
            .contains("baseline checkpoint_decode launch geometry is invalid"));

        let mut wrong_packed_treatment = p19_record();
        wrong_packed_treatment.measurements[1]
            .launch_geometries
            .as_mut()
            .expect("P19 launch evidence")
            .insert(
                P19_CHECKPOINT_LAUNCH.to_string(),
                LaunchGeometry {
                    grid: [128, 1, 1],
                    block: [1, 1, 1],
                },
            );
        assert!(validate(&wrong_packed_treatment)
            .unwrap_err()
            .contains("treatment checkpoint_decode launch geometry is invalid"));

        let small_treatment_index = P19_WORKLOAD_IDS
            .iter()
            .position(|id| *id == "ybr420_64x64_batch1_restart_none")
            .expect("small P19 workload")
            * 2
            + 1;
        let mut wrong_small_treatment = p19_record();
        wrong_small_treatment.measurements[small_treatment_index]
            .launch_geometries
            .as_mut()
            .expect("P19 launch evidence")
            .insert(
                P19_CHECKPOINT_LAUNCH.to_string(),
                LaunchGeometry {
                    grid: [1, 1, 1],
                    block: [128, 1, 1],
                },
            );
        assert!(validate(&wrong_small_treatment)
            .unwrap_err()
            .contains("treatment checkpoint_decode launch geometry is invalid"));
    }

    #[test]
    fn p19_schema_v3_rounds_up_a_packed_checkpoint_tail() {
        let mut record = p19_record();
        for (measurement, (grid_x, block_x)) in record
            .measurements
            .iter_mut()
            .take(2)
            .zip([(129, 1), (2, 128)])
        {
            measurement.checkpoint_count = Some(129);
            measurement
                .launch_geometries
                .as_mut()
                .expect("P19 launch evidence")
                .insert(
                    P19_CHECKPOINT_LAUNCH.to_string(),
                    LaunchGeometry {
                        grid: [grid_x, 1, 1],
                        block: [block_x, 1, 1],
                    },
                );
        }
        validate(&record).expect("P19 packed tail uses ceiling division");
    }

    #[test]
    fn p19_schema_v3_rejects_a_slower_geometry_changing_treatment() {
        let mut record = p19_record();
        let treatment = &mut record.measurements[3];
        treatment.wall_time_ns = 102.0;
        treatment.wall_time_ci_lower_ns = 101.0;
        treatment.wall_time_ci_upper_ns = 103.0;

        assert!(validate(&record)
            .unwrap_err()
            .contains("geometry-changing treatment must not be slower"));
    }

    #[test]
    fn p19_defusion_schema_v3_requires_split_stage_contract() {
        let mut record = p19_record();
        record.experiment_id = "P19-CUDA-JPEG-DECODE-DEFUSION".to_string();
        record.status = "rejected".to_string();

        assert!(validate(&record)
            .unwrap_err()
            .contains("P19 defusion baseline has incorrect stage-time keys"));
    }

    #[test]
    fn p19_defusion_schema_v3_complete_record_is_accepted() {
        validate(&p19_defusion_record()).expect("valid P19 defusion record");
    }

    #[test]
    fn p19_defusion_schema_v3_requires_exact_scratch_traffic_and_route_stages() {
        let mut baseline_scratch = p19_defusion_record();
        baseline_scratch.measurements[0].coefficient_scratch_bytes = Some(1);
        assert!(validate(&baseline_scratch)
            .unwrap_err()
            .contains("baseline fused-route evidence is invalid"));

        let mut wrong_traffic = p19_defusion_record();
        wrong_traffic.measurements[1].coefficient_scratch_traffic_bytes = Some(1);
        assert!(validate(&wrong_traffic)
            .unwrap_err()
            .contains("scratch or logical traffic is invalid"));

        let mut missing_split_stage = p19_defusion_record();
        missing_split_stage.measurements[1]
            .stage_times_ns
            .insert("idct_deposit".to_string(), 0.0);
        assert!(validate(&missing_split_stage)
            .unwrap_err()
            .contains("treatment split-stage timing is invalid"));

        let mut non420_split = p19_defusion_record();
        non420_split.measurements[9]
            .stage_times_ns
            .insert("fused_decode_kernel".to_string(), 0.0);
        assert!(validate(&non420_split)
            .unwrap_err()
            .contains("treatment fused-stage timing is invalid"));
    }

    #[test]
    fn p19_defusion_schema_v3_requires_truthful_launches_and_paired_conformance() {
        let mut missing_idct = p19_defusion_record();
        missing_idct.measurements[1]
            .launch_geometries
            .as_mut()
            .expect("P19 defusion launch map")
            .remove(P19_IDCT_LAUNCH);
        assert!(validate(&missing_idct)
            .unwrap_err()
            .contains("launch map is incomplete"));

        let mut wrong_idct = p19_defusion_record();
        wrong_idct.measurements[1]
            .launch_geometries
            .as_mut()
            .expect("P19 defusion launch map")
            .get_mut(P19_IDCT_LAUNCH)
            .expect("P19 defusion IDCT launch")
            .grid = [2, 1, 1];
        assert!(validate(&wrong_idct)
            .unwrap_err()
            .contains("idct_deposit launch geometry is invalid"));

        let mut conformance = p19_defusion_record();
        conformance.measurements[1].conformance = "pass: different evidence".to_string();
        assert!(validate(&conformance)
            .unwrap_err()
            .contains("conformance evidence differs"));
    }

    #[test]
    fn p13_schema_v2_requires_exact_workloads_priority_and_stage_contract() {
        let mut wrong_workload = p13_record();
        wrong_workload.workloads[0].id = "wrong".to_string();
        wrong_workload.measurements[0].workload_id = "wrong".to_string();
        wrong_workload.measurements[1].workload_id = "wrong".to_string();
        assert!(validate(&wrong_workload)
            .unwrap_err()
            .contains("exact workload matrix"));

        let mut wrong_priority = p13_record();
        wrong_priority.decision.priority_workload_id =
            "cuda-dwt97-resident-preencode-512-b16".to_string();
        assert!(validate(&wrong_priority)
            .unwrap_err()
            .contains("priority workload"));

        let mut wrong_output = p13_record();
        wrong_output.workloads[0].output = "prequantized_codeblocks".to_string();
        assert!(validate(&wrong_output)
            .unwrap_err()
            .contains("preencoded_htj2k_codeblocks"));

        let mut wrong_stages = p13_record();
        wrong_stages.measurements[0]
            .stage_times_ns
            .remove("ht_encode");
        wrong_stages.measurements[1]
            .stage_times_ns
            .remove("ht_encode");
        assert!(validate(&wrong_stages)
            .unwrap_err()
            .contains("stage-time keys"));
    }

    #[test]
    fn p13_schema_v2_requires_route_traffic_hash_and_product_evidence() {
        let mut different_input = p13_record();
        different_input.measurements[1].input_sha256 = Some("b".repeat(64));
        assert!(validate(&different_input)
            .unwrap_err()
            .contains("input SHA-256 differs"));

        let mut wrong_column_route = p13_record();
        wrong_column_route.measurements[1]
            .stage_times_ns
            .insert("column_lift".to_string(), 1.0);
        assert!(validate(&wrong_column_route)
            .unwrap_err()
            .contains("column_lift"));

        let mut wrong_traffic = p13_record();
        wrong_traffic.measurements[0].temporary_float_band_traffic_bytes = Some(199);
        assert!(validate(&wrong_traffic)
            .unwrap_err()
            .contains("temporary float-band traffic"));

        let mut fallback = p13_record();
        fallback.measurements[2].cpu_fallback_jobs = Some(1);
        assert!(validate(&fallback).unwrap_err().contains("CPU fallback"));

        let mut no_independent_decode = p13_record();
        no_independent_decode.measurements[2].independent_decode_passed = Some(false);
        assert!(validate(&no_independent_decode)
            .unwrap_err()
            .contains("independent decode"));

        let mut no_ht_dispatch = p13_record();
        no_ht_dispatch.measurements[0].ht_codeblock_dispatches = Some(0);
        assert!(validate(&no_ht_dispatch)
            .unwrap_err()
            .contains("HT code-block dispatch"));
    }

    #[test]
    fn p6_requires_every_available_static_and_throughput_metric() {
        let mut record = record();
        record.experiment_id = "P6-METAL-COMPILER-RESOURCE-EVIDENCE".to_string();
        record.status = "blocked".to_string();
        record.measurements.pop();
        let measurement = &mut record.measurements[0];
        measurement.thread_execution_width = Some(32);
        measurement.max_threads_per_group = Some(1024);
        measurement.code_blocks_per_second = Some(1_000.0);
        validate(&record).expect("P6 accepts measured public pipeline evidence");

        record.measurements[0].code_blocks_per_second = None;
        assert!(validate(&record)
            .unwrap_err()
            .contains("omits an available static pipeline or throughput metric"));
    }
}
