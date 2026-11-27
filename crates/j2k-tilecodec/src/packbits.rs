// SPDX-License-Identifier: MIT OR Apache-2.0

//! `PackBits` runs, shared by TIFF `PackBits` and DICOM RLE segments.

use crate::{error::truncated_input_at, pool::NoPool, TileCodecError};
use j2k_core::{BufferError, TileDecompress};

/// One decoded `PackBits` run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Run<'a> {
    /// Header `n` in 0..=127: the next `n + 1` bytes, copied as they are.
    Literal(&'a [u8]),
    /// Header `n` in -127..=-1: the next byte, repeated `1 - n` times.
    Replicate { byte: u8, count: usize },
    /// Header -128: no output.
    NoOp,
}

impl Run<'_> {
    /// Number of bytes the run decodes to.
    pub(crate) const fn len(&self) -> usize {
        match *self {
            Self::Literal(bytes) => bytes.len(),
            Self::Replicate { count, .. } => count,
            Self::NoOp => 0,
        }
    }

    /// Write the run into `dest`, which must be exactly [`Self::len`] bytes.
    fn fill(&self, dest: &mut [u8]) {
        match *self {
            Self::Literal(bytes) => dest.copy_from_slice(bytes),
            Self::Replicate { byte, .. } => dest.fill(byte),
            Self::NoOp => {}
        }
    }

    /// Write up to [`Self::len`] bytes of the run into the next `slots`,
    /// stopping early when `slots` runs out.
    pub(crate) fn write_to<'s>(&self, slots: &mut impl Iterator<Item = &'s mut u8>) {
        match *self {
            // `bytes` comes first so that `zip` never takes a slot it does
            // not fill.
            Self::Literal(bytes) => {
                for (&byte, slot) in bytes.iter().zip(slots) {
                    *slot = byte;
                }
            }
            Self::Replicate { byte, count } => {
                for slot in slots.take(count) {
                    *slot = byte;
                }
            }
            Self::NoOp => {}
        }
    }
}

/// A run whose header promises more bytes than the input holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TruncatedRun<'a> {
    /// Literal bytes present after the header; empty for a replicate run.
    pub(crate) available: &'a [u8],
}

/// Iterator over the `PackBits` runs of a byte stream. After a truncated run
/// it yields nothing more.
#[derive(Debug, Clone)]
pub(crate) struct Runs<'a> {
    rest: &'a [u8],
}

impl<'a> Runs<'a> {
    pub(crate) const fn new(input: &'a [u8]) -> Self {
        Self { rest: input }
    }

    /// Bytes not consumed yet.
    pub(crate) const fn remaining(&self) -> &'a [u8] {
        self.rest
    }
}

impl<'a> Iterator for Runs<'a> {
    type Item = Result<Run<'a>, TruncatedRun<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        let (&header, rest) = self.rest.split_first()?;
        self.rest = &[];
        let run = match header {
            0..=127 => {
                let Some((literal, tail)) = rest.split_at_checked(usize::from(header) + 1) else {
                    return Some(Err(TruncatedRun { available: rest }));
                };
                self.rest = tail;
                Run::Literal(literal)
            }
            128 => {
                self.rest = rest;
                Run::NoOp
            }
            129..=255 => {
                let Some((&byte, tail)) = rest.split_first() else {
                    return Some(Err(TruncatedRun { available: &[] }));
                };
                self.rest = tail;
                // As a signed byte the header is n = header - 256, and the
                // run repeats 1 - n = 257 - header times.
                Run::Replicate {
                    byte,
                    count: 257 - usize::from(header),
                }
            }
        };
        Some(Ok(run))
    }
}

/// Decoder for TIFF `PackBits` (compression 32773) tile payloads.
///
/// Decodes every run in the payload: a literal run with `n` in 0 to 127
/// copies the next `n + 1` bytes, a replicate run with `n` in -127 to -1
/// repeats the next byte `1 - n` times, and -128 is a no-op. A run that
/// does not fit in the output buffer returns
/// [`BufferError::OutputTooSmall`] with the payload's full decoded length,
/// and a run cut off by the end of the payload returns a truncated-input
/// error. `expected_size` returns the decoded length, found by reading the
/// run headers.
pub struct PackBitsCodec;

#[doc(hidden)]
impl TileDecompress for PackBitsCodec {
    type Error = TileCodecError;
    type Pool = NoPool;

    fn expected_size(input: &[u8]) -> Result<Option<usize>, Self::Error> {
        decoded_len(input).map(Some)
    }

    fn decompress_into(
        _pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
    ) -> Result<usize, Self::Error> {
        let mut written = 0usize;
        for run in Runs::new(input) {
            let run = run.map_err(|_| truncated_input_at(input.len(), "PackBits run"))?;
            let dest = written
                .checked_add(run.len())
                .and_then(|end| out.get_mut(written..end));
            let Some(dest) = dest else {
                return Err(output_too_small(input, out.len()));
            };
            run.fill(dest);
            written += run.len();
        }
        Ok(written)
    }
}

/// Total decoded length of a `PackBits` stream.
fn decoded_len(input: &[u8]) -> Result<usize, TileCodecError> {
    Runs::new(input).try_fold(0usize, |total, run| {
        let run = run.map_err(|_| truncated_input_at(input.len(), "PackBits run"))?;
        total
            .checked_add(run.len())
            .ok_or(TileCodecError::Buffer(BufferError::SizeOverflow {
                what: "PackBits decoded length",
            }))
    })
}

/// The error for a payload that decodes past `have` bytes: the full decoded
/// length when the payload is otherwise valid, or the payload's own error.
fn output_too_small(input: &[u8], have: usize) -> TileCodecError {
    match decoded_len(input) {
        Ok(required) => TileCodecError::Buffer(BufferError::OutputTooSmall { required, have }),
        Err(error) => error,
    }
}

#[cfg(test)]
mod tests {
    use super::{Run, Runs, TruncatedRun};

    #[test]
    fn headers_map_to_runs_across_the_whole_signed_range() {
        let runs = Runs::new(&[0, 7, 127]).collect::<Vec<_>>();
        assert_eq!(
            runs,
            [Ok(Run::Literal(&[7])), Err(TruncatedRun { available: &[] })]
        );

        let replicate = |header: u8| match Runs::new(&[header, 9]).next() {
            Some(Ok(Run::Replicate { byte, count })) => Some((byte, count)),
            _ => None,
        };
        assert_eq!(replicate(255), Some((9, 2)));
        assert_eq!(replicate(193), Some((9, 64)));
        assert_eq!(
            Runs::new(&[129, 9]).next(),
            Some(Ok(Run::Replicate {
                byte: 9,
                count: 128
            }))
        );
        assert_eq!(Runs::new(&[128]).next(), Some(Ok(Run::NoOp)));
    }

    #[test]
    fn a_truncated_run_ends_iteration_and_reports_the_bytes_present() {
        let mut runs = Runs::new(&[3, 1, 2]);
        assert_eq!(runs.next(), Some(Err(TruncatedRun { available: &[1, 2] })));
        assert_eq!(runs.next(), None);
        assert_eq!(runs.remaining(), []);

        let mut runs = Runs::new(&[200]);
        assert_eq!(runs.next(), Some(Err(TruncatedRun { available: &[] })));
        assert_eq!(runs.next(), None);
    }

    #[test]
    fn strided_writes_stop_when_the_slots_run_out() {
        let mut out = [0_u8; 6];
        let mut slots = out.iter_mut().step_by(2);
        Run::Literal(&[1, 2]).write_to(&mut slots);
        Run::Replicate { byte: 5, count: 4 }.write_to(&mut slots);
        assert_eq!(slots.next(), None);
        assert_eq!(out, [1, 0, 2, 0, 5, 0]);
    }
}
