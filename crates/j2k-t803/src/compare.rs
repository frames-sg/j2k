use thiserror::Error;

/// Inclusive T.803 error bounds for one reference component.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErrorBounds {
    /// Maximum absolute sample error.
    pub peak: u64,
    /// Maximum mean squared error.
    pub mse: f64,
}

/// Error metrics and their inclusive pass result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Comparison {
    /// Measured maximum absolute sample error.
    pub peak: u64,
    /// Measured mean squared error.
    pub mse: f64,
    /// Whether both measured values are within their inclusive bounds.
    pub passed: bool,
}

/// Peak-error metric and its inclusive pass result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeakComparison {
    /// Measured maximum absolute sample error.
    pub peak: u64,
    /// Whether the measured value is within the inclusive bound.
    pub passed: bool,
}

/// Error returned when sample arrays cannot be compared.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ComparisonError {
    /// T.803 metrics are undefined for an empty component.
    #[error("cannot compare empty components")]
    Empty,
    /// Reference and decoded components contain different sample counts.
    #[error("sample count mismatch: reference {reference}, decoded {decoded}")]
    Length {
        /// Reference sample count.
        reference: usize,
        /// Decoded sample count.
        decoded: usize,
    },
    /// The configured MSE bound is negative, infinite, or NaN.
    #[error("MSE bound must be finite and non-negative")]
    InvalidMseBound,
    /// Accumulating squared error exceeded the metric representation.
    #[error("squared-error sum overflowed")]
    Overflow,
}

/// Measure peak error and MSE and apply inclusive T.803 bounds.
pub fn compare_samples(
    reference: &[i64],
    decoded: &[i64],
    bounds: ErrorBounds,
) -> Result<Comparison, ComparisonError> {
    if !bounds.mse.is_finite() || bounds.mse < 0.0 {
        return Err(ComparisonError::InvalidMseBound);
    }
    let (peak, squared_error) = error_sums(reference, decoded)?;
    let mse = mse_from_exact_sum(squared_error, reference.len())?;
    Ok(Comparison {
        peak,
        mse,
        passed: peak <= bounds.peak && mse <= bounds.mse,
    })
}

/// Measure peak error and apply an inclusive peak-only bound.
pub fn compare_peak_samples(
    reference: &[i64],
    decoded: &[i64],
    bound: u64,
) -> Result<PeakComparison, ComparisonError> {
    let (peak, _) = error_sums(reference, decoded)?;
    Ok(PeakComparison {
        peak,
        passed: peak <= bound,
    })
}

fn error_sums(reference: &[i64], decoded: &[i64]) -> Result<(u64, u128), ComparisonError> {
    if reference.is_empty() {
        return Err(ComparisonError::Empty);
    }
    if reference.len() != decoded.len() {
        return Err(ComparisonError::Length {
            reference: reference.len(),
            decoded: decoded.len(),
        });
    }
    let mut peak = 0_u64;
    let mut squared_error = 0_u128;
    for (&reference, &decoded) in reference.iter().zip(decoded) {
        let difference = (i128::from(reference) - i128::from(decoded)).unsigned_abs();
        peak = peak.max(u64::try_from(difference).map_err(|_| ComparisonError::Overflow)?);
        squared_error = squared_error
            .checked_add(
                difference
                    .checked_mul(difference)
                    .ok_or(ComparisonError::Overflow)?,
            )
            .ok_or(ComparisonError::Overflow)?;
    }
    Ok((peak, squared_error))
}

fn mse_from_exact_sum(squared_error: u128, sample_count: usize) -> Result<f64, ComparisonError> {
    // T.803 defines MSE as a real-valued average. Accumulation stays exact;
    // conversion happens once at the required floating-point comparison.
    let sample_count = u128::try_from(sample_count).map_err(|_| ComparisonError::Overflow)?;
    Ok(u128_as_f64(squared_error)? / u128_as_f64(sample_count)?)
}

fn u128_as_f64(value: u128) -> Result<f64, ComparisonError> {
    const TWO_POW_64: f64 = 18_446_744_073_709_551_616.0;
    let low = u64::try_from(value & u128::from(u64::MAX)).map_err(|_| ComparisonError::Overflow)?;
    let high = u64::try_from(value >> 64).map_err(|_| ComparisonError::Overflow)?;
    Ok(u64_as_f64(high)? * TWO_POW_64 + u64_as_f64(low)?)
}

pub(crate) fn u64_as_f64(value: u64) -> Result<f64, ComparisonError> {
    const TWO_POW_32: f64 = 4_294_967_296.0;
    let low = u32::try_from(value & u64::from(u32::MAX)).map_err(|_| ComparisonError::Overflow)?;
    let high = u32::try_from(value >> 32).map_err(|_| ComparisonError::Overflow)?;
    Ok(f64::from(high) * TWO_POW_32 + f64::from(low))
}
