// SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;
use std::{fs::OpenOptions, io::Write as _, path::PathBuf};

use j2k_profile::{ProfileError, ProfileLimits, ProfileResult};

use super::CudaHtj2kEncodeProfileReport;
#[cfg(any(feature = "cuda-runtime", test))]
use super::CudaHtj2kProfileReport;

const CUDA_TRACE_ENV_VAR: &str = "J2K_CUDA_TRACE";

#[cfg(feature = "cuda-runtime")]
pub(super) fn export_trace_if_requested(path: &str, report: &CudaHtj2kProfileReport) {
    let Some(trace_path) = std::env::var_os(CUDA_TRACE_ENV_VAR) else {
        return;
    };
    let trace = match chrome_trace_json(path, report) {
        Ok(trace) => trace,
        Err(error) => {
            j2k_profile::emit_profile_error("cuda_htj2k_trace_format", &error);
            return;
        }
    };
    let trace_path = PathBuf::from(trace_path);
    if let Err(error) = write_trace_file(&trace_path, &trace) {
        emit_trace_write_error("cuda_htj2k_trace_write", &error);
    }
}

pub(super) fn export_encode_trace_if_requested(path: &str, report: &CudaHtj2kEncodeProfileReport) {
    let Some(trace_path) = std::env::var_os(CUDA_TRACE_ENV_VAR) else {
        return;
    };
    let trace = match chrome_encode_trace_json(path, report) {
        Ok(trace) => trace,
        Err(error) => {
            j2k_profile::emit_profile_error("cuda_htj2k_encode_trace_format", &error);
            return;
        }
    };
    let trace_path = PathBuf::from(trace_path);
    if let Err(error) = write_trace_file(&trace_path, &trace) {
        emit_trace_write_error("cuda_htj2k_encode_trace_write", &error);
    }
}

pub(super) fn write_trace_file(path: &std::path::Path, trace: &str) -> std::io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(trace.as_bytes())
}

fn emit_trace_write_error(operation: &'static str, error: &std::io::Error) {
    let what = match error.kind() {
        std::io::ErrorKind::AlreadyExists => "CUDA trace path already exists",
        std::io::ErrorKind::NotFound => "CUDA trace parent path was not found",
        std::io::ErrorKind::PermissionDenied => "CUDA trace path permission denied",
        std::io::ErrorKind::WriteZero => "CUDA trace file write returned zero bytes",
        _ => "CUDA trace file create or write failed",
    };
    j2k_profile::emit_profile_error(operation, &ProfileError::InvalidInput { what });
}

#[cfg(any(feature = "cuda-runtime", test))]
pub(super) fn chrome_trace_json(
    path: &str,
    report: &CudaHtj2kProfileReport,
) -> ProfileResult<String> {
    let stages = [
        ("parse", report.parse_us),
        ("plan", report.plan_us),
        ("flatten", report.flatten_us),
        ("h2d", report.h2d_us),
        ("classic_tier1", report.classic_tier1_us),
        ("ht_cleanup", report.ht_cleanup_us),
        ("ht_refine", report.ht_refine_us),
        ("status_d2h", report.detail.status_d2h_us),
        ("dequant", report.dequant_us),
        ("idwt", report.idwt_us),
        ("mct", report.mct_us),
        ("store", report.store_us),
    ];
    let mut trace = BoundedTraceWriter::new(ProfileLimits::default());
    trace.include_input(path, "CUDA trace category")?;
    trace.write_fragment(format_args!("{{\"traceEvents\":["))?;
    let mut timestamp = 0u128;
    for (index, (name, duration)) in stages.iter().enumerate() {
        let event_timestamp = if *name == "ht_refine" {
            timestamp.saturating_sub(report.ht_cleanup_us)
        } else {
            timestamp
        };
        trace.write_event(index, name, path, event_timestamp, *duration)?;
        if *name != "ht_refine" {
            timestamp = timestamp.saturating_add(*duration);
        }
    }
    trace.write_fragment(format_args!("]}}"))?;
    Ok(trace.finish())
}

pub(super) fn chrome_encode_trace_json(
    path: &str,
    report: &CudaHtj2kEncodeProfileReport,
) -> ProfileResult<String> {
    let stages = [
        ("deinterleave", report.deinterleave_us),
        ("mct", report.mct_us),
        ("dwt", report.dwt_us),
        ("quantize", report.quantize_us),
        ("ht_encode", report.ht_encode_us),
        ("packetize", report.packetize_us),
    ];
    let mut trace = BoundedTraceWriter::new(ProfileLimits::default());
    trace.include_input(path, "CUDA encode trace category")?;
    trace.write_fragment(format_args!("{{\"traceEvents\":["))?;
    let mut timestamp = 0u128;
    for (index, (name, duration)) in stages.iter().enumerate() {
        trace.write_event(index, name, path, timestamp, *duration)?;
        timestamp = timestamp.saturating_add(*duration);
    }
    trace.write_fragment(format_args!("]}}"))?;
    Ok(trace.finish())
}

struct BoundedTraceWriter {
    output: String,
    limits: ProfileLimits,
    input_bytes: usize,
    error: Option<ProfileError>,
}

impl BoundedTraceWriter {
    const fn new(limits: ProfileLimits) -> Self {
        Self {
            output: String::new(),
            limits,
            input_bytes: 0,
            error: None,
        }
    }

    fn include_input(&mut self, text: &str, what: &'static str) -> ProfileResult<()> {
        ensure_limit(text.len(), self.limits.max_token_bytes(), what)?;
        self.input_bytes =
            self.input_bytes
                .checked_add(text.len())
                .ok_or(ProfileError::SizeOverflow {
                    what: "CUDA trace input bytes",
                })?;
        ensure_limit(
            self.input_bytes,
            self.limits.max_input_bytes(),
            "CUDA trace input bytes",
        )
    }

    fn write_event(
        &mut self,
        index: usize,
        name: &str,
        category: &str,
        timestamp: u128,
        duration: u128,
    ) -> ProfileResult<()> {
        if index != 0 {
            self.write_fragment(format_args!(","))?;
        }
        self.write_fragment(format_args!("{{\"name\":"))?;
        self.write_json_string(name)?;
        self.write_fragment(format_args!(",\"cat\":"))?;
        self.write_json_string(category)?;
        self.write_fragment(format_args!(
            ",\"ph\":\"X\",\"pid\":1,\"tid\":1,\"ts\":{timestamp},\"dur\":{duration}}}"
        ))
    }

    fn write_json_string(&mut self, text: &str) -> ProfileResult<()> {
        self.write_fragment(format_args!("\""))?;
        for character in text.chars() {
            match character {
                '"' => self.write_fragment(format_args!("\\\""))?,
                '\\' => self.write_fragment(format_args!("\\\\"))?,
                '\u{08}' => self.write_fragment(format_args!("\\b"))?,
                '\u{0c}' => self.write_fragment(format_args!("\\f"))?,
                '\n' => self.write_fragment(format_args!("\\n"))?,
                '\r' => self.write_fragment(format_args!("\\r"))?,
                '\t' => self.write_fragment(format_args!("\\t"))?,
                control if control <= '\u{1f}' => {
                    self.write_fragment(format_args!("\\u{:04x}", u32::from(control)))?;
                }
                other => self.write_fragment(format_args!("{other}"))?,
            }
        }
        self.write_fragment(format_args!("\""))
    }

    fn write_fragment(&mut self, arguments: fmt::Arguments<'_>) -> ProfileResult<()> {
        if fmt::write(self, arguments).is_err() {
            return Err(self.error.take().unwrap_or(ProfileError::InvalidInput {
                what: "CUDA trace formatter failed",
            }));
        }
        Ok(())
    }

    fn try_write_str(&mut self, text: &str) -> ProfileResult<()> {
        let required =
            self.output
                .len()
                .checked_add(text.len())
                .ok_or(ProfileError::SizeOverflow {
                    what: "CUDA trace output bytes",
                })?;
        ensure_limit(
            required,
            self.limits.max_output_bytes(),
            "CUDA trace output bytes",
        )?;
        if required > self.output.capacity() {
            self.output
                .try_reserve_exact(required - self.output.len())
                .map_err(|_| ProfileError::AllocationFailed {
                    what: "CUDA trace output",
                    requested: required,
                })?;
            ensure_limit(
                self.output.capacity(),
                self.limits.max_output_bytes(),
                "CUDA trace allocator capacity",
            )?;
        }
        self.output.push_str(text);
        Ok(())
    }

    fn finish(self) -> String {
        self.output
    }
}

impl fmt::Write for BoundedTraceWriter {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        if self.error.is_some() {
            return Err(fmt::Error);
        }
        if let Err(error) = self.try_write_str(text) {
            self.error = Some(error);
            return Err(fmt::Error);
        }
        Ok(())
    }
}

fn ensure_limit(requested: usize, limit: usize, what: &'static str) -> ProfileResult<()> {
    if requested > limit {
        return Err(ProfileError::LimitExceeded {
            what,
            requested,
            limit,
        });
    }
    Ok(())
}
