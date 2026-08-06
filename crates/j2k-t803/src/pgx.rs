use thiserror::Error;

/// A decoded T.803 PGX component.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PgxImage {
    /// Component width in samples.
    pub width: u32,
    /// Component height in samples.
    pub height: u32,
    /// Nominal sample precision.
    pub bit_depth: u8,
    /// Whether samples are signed.
    pub signed: bool,
    /// Canonical integer samples in row-major order.
    pub samples: Vec<i64>,
}

/// Error returned for malformed or unsupported PGX data.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PgxError {
    /// The header line is absent or malformed.
    #[error("invalid PGX header: {0}")]
    Header(&'static str),
    /// The declared byte order is not a recognized PGX byte order.
    #[error("PGX byte order must be ML or LM")]
    ByteOrder,
    /// The declared bit depth is outside 1 through 32.
    #[error("PGX bit depth must be between 1 and 32")]
    BitDepth,
    /// A numeric header field is invalid.
    #[error("invalid PGX {field}")]
    Number {
        /// Field being parsed.
        field: &'static str,
    },
    /// Width or height is zero or cannot be represented safely.
    #[error("invalid PGX dimensions")]
    Dimensions,
    /// The binary payload does not exactly match the declared dimensions.
    #[error("PGX payload length is {actual}, expected {expected}")]
    PayloadLength {
        /// Expected byte count.
        expected: usize,
        /// Actual byte count.
        actual: usize,
    },
    /// A signed sample is not sign-extended to its storage boundary.
    #[error("PGX signed sample has invalid sign extension")]
    SignExtension,
    /// An unsigned sample uses bits outside its declared precision.
    #[error("PGX unsigned sample exceeds its declared precision")]
    Precision,
    /// The declared component cannot be allocated.
    #[error("cannot allocate PGX component")]
    Allocation,
}

/// Parse the PGX representation used by the T.803 electronic attachment.
///
/// The parser requires an exact payload length and validates sign extension or
/// zero extension outside the declared precision. It accepts the attachment's
/// declared `ML` and legacy `LM` storage orders and only ASCII spaces as field
/// separators.
pub fn parse_pgx(bytes: &[u8]) -> Result<PgxImage, PgxError> {
    let newline = bytes
        .iter()
        .position(|byte| *byte == b'\n')
        .ok_or(PgxError::Header("missing newline"))?;
    let header_bytes = bytes[..newline]
        .strip_suffix(b"\r")
        .unwrap_or(&bytes[..newline]);
    let header =
        core::str::from_utf8(header_bytes).map_err(|_| PgxError::Header("header is not UTF-8"))?;
    if header.is_empty()
        || header.starts_with(' ')
        || header.ends_with(' ')
        || !header.is_ascii()
        || header
            .bytes()
            .any(|byte| byte != b' ' && !(0x21..=0x7e).contains(&byte))
    {
        return Err(PgxError::Header("fields must use ASCII spaces"));
    }

    let mut fields = header.split(' ').filter(|field| !field.is_empty());
    if fields.next() != Some("PG") {
        return Err(PgxError::Header("expected PG ML precision width height"));
    }
    let byte_order = match fields.next() {
        Some("ML") => ByteOrder::Big,
        Some("LM") => ByteOrder::Little,
        _ => return Err(PgxError::ByteOrder),
    };

    let precision = fields
        .next()
        .ok_or(PgxError::Header("missing precision field"))?;
    let (signed, depth_field) = match precision {
        "+" => (
            false,
            fields.next().ok_or(PgxError::Header("missing bit depth"))?,
        ),
        "-" => (
            true,
            fields.next().ok_or(PgxError::Header("missing bit depth"))?,
        ),
        _ => parse_attached_precision(precision)?,
    };
    let width_field = fields.next().ok_or(PgxError::Header("missing width"))?;
    let height_field = fields.next().ok_or(PgxError::Header("missing height"))?;
    if fields.next().is_some() {
        return Err(PgxError::Header("too many header fields"));
    }
    let bit_depth = depth_field.parse::<u8>().map_err(|_| PgxError::BitDepth)?;
    if !(1..=32).contains(&bit_depth) {
        return Err(PgxError::BitDepth);
    }
    let width = parse_dimension(width_field, "width")?;
    let height = parse_dimension(height_field, "height")?;
    let sample_count = usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or(PgxError::Dimensions)?;
    let bytes_per_sample = match bit_depth {
        1..=8 => 1,
        9..=16 => 2,
        17..=32 => 4,
        _ => unreachable!(),
    };
    let expected = sample_count
        .checked_mul(bytes_per_sample)
        .ok_or(PgxError::Dimensions)?;
    let payload = &bytes[newline + 1..];
    if payload.len() != expected {
        return Err(PgxError::PayloadLength {
            expected,
            actual: payload.len(),
        });
    }

    let mut samples = Vec::new();
    samples
        .try_reserve_exact(sample_count)
        .map_err(|_| PgxError::Allocation)?;
    for storage in payload.chunks_exact(bytes_per_sample) {
        samples.push(decode_sample(storage, bit_depth, signed, byte_order)?);
    }
    Ok(PgxImage {
        width,
        height,
        bit_depth,
        signed,
        samples,
    })
}

#[derive(Clone, Copy)]
enum ByteOrder {
    Big,
    Little,
}

fn parse_attached_precision(precision: &str) -> Result<(bool, &str), PgxError> {
    let (signed, depth) = match precision.as_bytes().first() {
        Some(b'+') => (false, &precision[1..]),
        Some(b'-') => (true, &precision[1..]),
        Some(byte) if byte.is_ascii_digit() => (false, precision),
        _ => return Err(PgxError::Header("invalid precision field")),
    };
    if depth.is_empty() {
        return Err(PgxError::Header("missing bit depth"));
    }
    Ok((signed, depth))
}

fn parse_dimension(field: &str, name: &'static str) -> Result<u32, PgxError> {
    let value = field
        .parse::<u32>()
        .map_err(|_| PgxError::Number { field: name })?;
    if value == 0 {
        return Err(PgxError::Dimensions);
    }
    Ok(value)
}

fn decode_sample(
    storage: &[u8],
    bit_depth: u8,
    signed: bool,
    byte_order: ByteOrder,
) -> Result<i64, PgxError> {
    let unsigned = match (storage, byte_order) {
        ([value], _) => u64::from(*value),
        ([a, b], ByteOrder::Big) => u64::from(u16::from_be_bytes([*a, *b])),
        ([a, b], ByteOrder::Little) => u64::from(u16::from_le_bytes([*a, *b])),
        ([a, b, c, d], ByteOrder::Big) => u64::from(u32::from_be_bytes([*a, *b, *c, *d])),
        ([a, b, c, d], ByteOrder::Little) => u64::from(u32::from_le_bytes([*a, *b, *c, *d])),
        _ => unreachable!(),
    };
    if !signed {
        let maximum = (1_u64 << bit_depth) - 1;
        if unsigned > maximum {
            return Err(PgxError::Precision);
        }
        return i64::try_from(unsigned).map_err(|_| PgxError::Precision);
    }

    let value = match (storage, byte_order) {
        ([value], _) => i64::from(i8::from_be_bytes([*value])),
        ([a, b], ByteOrder::Big) => i64::from(i16::from_be_bytes([*a, *b])),
        ([a, b], ByteOrder::Little) => i64::from(i16::from_le_bytes([*a, *b])),
        ([a, b, c, d], ByteOrder::Big) => i64::from(i32::from_be_bytes([*a, *b, *c, *d])),
        ([a, b, c, d], ByteOrder::Little) => i64::from(i32::from_le_bytes([*a, *b, *c, *d])),
        _ => unreachable!(),
    };
    let limit = 1_i64 << (bit_depth - 1);
    if (-limit..limit).contains(&value) {
        Ok(value)
    } else {
        Err(PgxError::SignExtension)
    }
}
