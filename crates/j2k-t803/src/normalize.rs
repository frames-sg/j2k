use thiserror::Error;

/// One decoded component before T.803 output normalization.
#[derive(Clone, Copy, Debug)]
pub struct Component<'a> {
    /// Component width in samples.
    pub width: u32,
    /// Component height in samples.
    pub height: u32,
    /// Nominal component precision.
    pub bit_depth: u8,
    /// Whether samples are signed.
    pub signed: bool,
    /// Horizontal and vertical strides used to undo decoder-side replication.
    ///
    /// Use `(1, 1)` when the decoder retained the codestream component's
    /// native sampling grid.
    pub post_decode_subsampling: (u8, u8),
    /// Canonical integer samples in row-major order.
    pub samples: &'a [i64],
}

/// Reference shape and representation required by one T.803 comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormalizationTarget {
    /// Reference width in samples.
    pub width: u32,
    /// Reference height in samples.
    pub height: u32,
    /// Reference precision.
    pub bit_depth: u8,
    /// Whether reference samples are signed.
    pub signed: bool,
}

/// Error returned when decoded output cannot be normalized as required.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum NormalizationError {
    /// Source metadata or storage is internally inconsistent.
    #[error("invalid decoded component: {0}")]
    InvalidSource(&'static str),
    /// The target cannot be cropped from the source component.
    #[error("reference dimensions exceed decoded component dimensions")]
    Dimensions,
    /// Reference and decoded components disagree on signedness.
    #[error("reference signedness differs from decoded component signedness")]
    Signedness,
    /// The reference precision cannot be obtained through T.803 downscaling.
    #[error("reference bit depth exceeds decoded component bit depth")]
    BitDepth,
    /// The normalized output allocation could not be reserved.
    #[error("cannot allocate normalized component")]
    Allocation,
}

/// Apply T.803 clipping, precision reduction, and upper-left cropping.
pub fn normalize_component(
    source: Component<'_>,
    target: NormalizationTarget,
) -> Result<Vec<i64>, NormalizationError> {
    validate_source(source)?;
    let (horizontal_step, vertical_step) = source.post_decode_subsampling;
    let sampled_width = source.width.div_ceil(u32::from(horizontal_step));
    let sampled_height = source.height.div_ceil(u32::from(vertical_step));
    if target.width == 0
        || target.height == 0
        || target.width > sampled_width
        || target.height > sampled_height
    {
        return Err(NormalizationError::Dimensions);
    }
    if target.signed != source.signed {
        return Err(NormalizationError::Signedness);
    }
    if target.bit_depth == 0 || target.bit_depth > source.bit_depth {
        return Err(NormalizationError::BitDepth);
    }

    let source_width = source.width as usize;
    let target_width = target.width as usize;
    let target_height = target.height as usize;
    let horizontal_step = usize::from(horizontal_step);
    let vertical_step = usize::from(vertical_step);
    let shift = source.bit_depth - target.bit_depth;
    let (minimum, maximum) = sample_range(source.bit_depth, source.signed);
    let target_len = target_width
        .checked_mul(target_height)
        .ok_or(NormalizationError::Dimensions)?;
    let mut normalized = Vec::new();
    normalized
        .try_reserve_exact(target_len)
        .map_err(|_| NormalizationError::Allocation)?;
    for row in source
        .samples
        .chunks_exact(source_width)
        .step_by(vertical_step)
        .take(target_height)
    {
        normalized.extend(
            row.iter()
                .step_by(horizontal_step)
                .take(target_width)
                .map(|sample| sample.clamp(&minimum, &maximum) >> shift),
        );
    }
    Ok(normalized)
}

fn validate_source(source: Component<'_>) -> Result<(), NormalizationError> {
    if source.width == 0
        || source.height == 0
        || !(1..=32).contains(&source.bit_depth)
        || source.post_decode_subsampling.0 == 0
        || source.post_decode_subsampling.1 == 0
    {
        return Err(NormalizationError::InvalidSource(
            "dimensions and bit depth must be non-zero",
        ));
    }
    let expected = (source.width as usize)
        .checked_mul(source.height as usize)
        .ok_or(NormalizationError::InvalidSource("dimensions overflow"))?;
    if source.samples.len() != expected {
        return Err(NormalizationError::InvalidSource(
            "sample length does not match dimensions",
        ));
    }
    Ok(())
}

fn sample_range(bit_depth: u8, signed: bool) -> (i64, i64) {
    if signed {
        let limit = 1_i64 << (bit_depth - 1);
        (-limit, limit - 1)
    } else {
        (0, (1_i64 << bit_depth) - 1)
    }
}
