// SPDX-License-Identifier: MIT OR Apache-2.0

use super::cleanup::CleanupCoefficientSource;

/// Samples per row a row pair holds: the widest code block plus the columns
/// a final, partial quad pair reads past the right edge.
const ROW_PAIR_COLUMNS: usize = 1024 + 4;
const QUADS: usize = ROW_PAIR_COLUMNS / 2;

/// Two code-block rows as sample exponents and `MagSgn` values, zero past the
/// block's right and bottom edges, with every quad's cleanup coding values
/// derived up front. The quad walk then loads them without per-sample
/// branches, edge tests or the serial marker rows a quad's context and
/// `kappa` would otherwise come from.
pub(super) struct RowPair {
    exponents: [[u8; ROW_PAIR_COLUMNS]; 2],
    values: [[u32; ROW_PAIR_COLUMNS]; 2],
    /// Per quad: its significance pattern `rho`, largest exponent, and which
    /// samples reach it.
    rho: [u8; QUADS],
    e_qmax: [u8; QUADS],
    reach: [u8; QUADS],
    /// Per quad, from the row above it (the previous pair's lower row): its
    /// context bits 0 and 2, and the largest exponent of the four samples
    /// above and diagonally above it.
    above_context: [u8; QUADS],
    above_exponent: [u8; QUADS],
    /// Per quad: its `CxtVLC` table index `(c_q << 8) | (rho << 4) | eps`, its
    /// `u_q`, and `max(e_qmax, kappa)`, which bounds its `MagSgn` payloads.
    index: [u16; QUADS],
    u_q: [u8; QUADS],
    bound: [u8; QUADS],
}

impl RowPair {
    pub(super) fn new() -> Self {
        Self {
            exponents: [[0; ROW_PAIR_COLUMNS]; 2],
            values: [[0; ROW_PAIR_COLUMNS]; 2],
            rho: [0; QUADS],
            e_qmax: [0; QUADS],
            reach: [0; QUADS],
            above_context: [0; QUADS],
            above_exponent: [0; QUADS],
            index: [0; QUADS],
            u_q: [0; QUADS],
            bound: [0; QUADS],
        }
    }

    /// Load rows `y` and `y + 1` of a `width` x `height` block at cleanup shift `p`.
    pub(super) fn fill<C: CleanupCoefficientSource + ?Sized>(
        &mut self,
        coefficients: &C,
        y: usize,
        width: usize,
        height: usize,
        p: u32,
    ) -> Result<(), &'static str> {
        let quads = width.div_ceil(2);
        if y > 0 {
            self.fill_above(quads);
        }
        let rows = self.exponents.iter_mut().zip(&mut self.values);
        for (row, (exponents, values)) in rows.enumerate() {
            let filled = if y + row < height {
                let start = (y + row) * width;
                coefficients.fill_row(start, p, &mut exponents[..width], &mut values[..width])?;
                width
            } else {
                0
            };
            exponents[filled..width + 4].fill(0);
        }
        // An indexed loop over quads vectorizes; the walk then loads each
        // quad's pattern and maximum instead of deriving them per sample.
        let [upper, lower] = &self.exponents;
        for quad in 0..width / 2 + 2 {
            let (left, right) = (2 * quad, 2 * quad + 1);
            let samples = [upper[left], lower[left], upper[right], lower[right]];
            self.rho[quad] = u8::from(samples[0] != 0)
                | u8::from(samples[1] != 0) << 1
                | u8::from(samples[2] != 0) << 2
                | u8::from(samples[3] != 0) << 3;
            let e_qmax = samples[0].max(samples[1]).max(samples[2].max(samples[3]));
            self.e_qmax[quad] = e_qmax;
            self.reach[quad] = u8::from(samples[0] == e_qmax)
                | u8::from(samples[1] == e_qmax) << 1
                | u8::from(samples[2] == e_qmax) << 2
                | u8::from(samples[3] == e_qmax) << 3;
        }
        self.fill_coding(quads, y == 0);
        Ok(())
    }

    /// What the encoder's marker rows held for each quad of the next pair:
    /// read from the lower row before it is overwritten. Column pair `j` is
    /// columns `2j - 1` and `2j` (column -1 lies outside the block); quad `j`
    /// sees column pairs `j` and `j + 1`. Zipped slices keep both passes
    /// vectorized.
    fn fill_above(&mut self, quads: usize) {
        let mut pair_max = [0_u8; QUADS + 1];
        let above = &self.exponents[1][..=2 * quads];
        pair_max[0] = above[0];
        let pairs = above[1..].as_chunks::<2>().0;
        for (max, pair) in pair_max[1..=quads].iter_mut().zip(pairs) {
            *max = pair[0].max(pair[1]);
        }
        let (left, right) = (&pair_max[..quads], &pair_max[1..=quads]);
        let targets = self.above_exponent[..quads].iter_mut();
        let contexts = self.above_context[..quads].iter_mut();
        for (((exponent, context), &left), &right) in targets.zip(contexts).zip(left).zip(right) {
            *exponent = left.max(right);
            *context = u8::from(left != 0) | u8::from(right != 0) << 2;
        }
    }

    /// Each quad's `CxtVLC` index, `u_q` and payload bound. A first-row quad's
    /// context comes from its left neighbor alone and its `kappa` is 1; below
    /// the first row the context adds the row above, and a quad with two or
    /// more significant samples takes `kappa` from the exponents above it.
    fn fill_coding(&mut self, quads: usize, initial: bool) {
        let mut left_rho = [0_u8; QUADS];
        left_rho[1..quads].copy_from_slice(&self.rho[..quads - 1]);
        let fields = self.rho[..quads]
            .iter()
            .zip(&left_rho[..quads])
            .zip(&self.e_qmax[..quads])
            .zip(&self.reach[..quads])
            .zip(&self.above_context[..quads])
            .zip(&self.above_exponent[..quads]);
        let outputs = self.index[..quads]
            .iter_mut()
            .zip(&mut self.u_q[..quads])
            .zip(&mut self.bound[..quads]);
        for ((((((&rho, &left), &e_qmax), &reach), &above), &above_e), ((index, u_q), bound)) in
            fields.zip(outputs)
        {
            let (context, kappa) = if initial {
                ((left >> 1) | (left & 1), 1)
            } else {
                let multiple = u8::from(rho & rho.wrapping_sub(1) != 0);
                let kappa = (multiple * above_e.saturating_sub(1)).max(1);
                (above | (((left >> 2) | (left >> 3)) & 1) << 1, kappa)
            };
            *bound = e_qmax.max(kappa);
            *u_q = *bound - kappa;
            let eps = reach & (u8::from(*u_q > 0) * 0xF);
            *index = u16::from(context) << 8 | u16::from(rho) << 4 | u16::from(eps);
        }
    }
}

impl RowPair {
    /// Whether the quad pair at column `x` codes as MEL zeros alone: no
    /// significant sample and zero context in either quad, so its `CxtVLC`
    /// codewords, UVLC and `MagSgn` bits are empty.
    #[expect(
        clippy::inline_always,
        reason = "tested once per quad pair in the cleanup hot path"
    )]
    #[inline(always)]
    pub(super) fn pair_is_zero(&self, x: usize, width: usize) -> bool {
        let quad = x / 2;
        self.index[quad] == 0 && (x + 2 >= width || self.index[quad + 1] == 0)
    }

    #[expect(
        clippy::inline_always,
        reason = "quad loads are fused into the per-quad cleanup hot path"
    )]
    #[inline(always)]
    fn coded_quad(&self, quad: usize) -> CodedQuad {
        let (column, [upper, lower]) = (2 * quad, &self.values);
        CodedQuad {
            index: self.index[quad],
            e_qmax: self.e_qmax[quad],
            u_q: self.u_q[quad],
            bound: self.bound[quad],
            values: [
                upper[column],
                lower[column],
                upper[column + 1],
                lower[column + 1],
            ],
        }
    }
}

/// One quad's cleanup coding values. Its `MagSgn` values run upper and lower
/// sample of its left column, then of its right column; an insignificant
/// sample's value is never emitted.
#[derive(Clone, Copy)]
pub(super) struct CodedQuad {
    pub(super) index: u16,
    pub(super) e_qmax: u8,
    pub(super) u_q: u8,
    pub(super) bound: u8,
    pub(super) values: [u32; 4],
}

impl CodedQuad {
    pub(super) fn context(self) -> u16 {
        self.index >> 8
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "rho is the four-bit field of the table index"
    )]
    pub(super) fn rho(self) -> u8 {
        (self.index >> 4) as u8 & 0xF
    }
}

/// Per-quad operations that differ between the byte-emitting encoder and the
/// distribution collector; the quad-pair walk itself is shared by
/// [`quad_pair`]. `INITIAL` selects the first row pair's tables.
pub(super) trait QuadSink {
    type Error;

    fn quad<const INITIAL: bool>(&mut self, quad: CodedQuad) -> Result<(), Self::Error>;

    fn initial_uvlc_pair(&mut self, u_q0: i32, u_q1: i32) -> Result<(), Self::Error>;

    fn initial_uvlc_lone(&mut self, u_q0: i32) -> Result<(), Self::Error>;

    fn non_initial_uvlc(&mut self, u_q0: i32, u_q1: i32) -> Result<(), Self::Error>;
}

/// Code the quad pair at column `x` of the first (`INITIAL`) or a later row
/// pair; the pair's second quad exists unless the block ends inside it.
#[expect(
    clippy::inline_always,
    reason = "this normative quad walk stays fused per sink"
)]
#[inline(always)]
pub(super) fn quad_pair<const INITIAL: bool, S: QuadSink>(
    rows: &RowPair,
    width: usize,
    x: usize,
    sink: &mut S,
) -> Result<(), S::Error> {
    let quad = x / 2;
    sink.quad::<INITIAL>(rows.coded_quad(quad))?;
    let u_q0 = i32::from(rows.u_q[quad]);
    if x + 2 < width {
        sink.quad::<INITIAL>(rows.coded_quad(quad + 1))?;
        let u_q1 = i32::from(rows.u_q[quad + 1]);
        if INITIAL {
            sink.initial_uvlc_pair(u_q0, u_q1)
        } else {
            sink.non_initial_uvlc(u_q0, u_q1)
        }
    } else if INITIAL {
        sink.initial_uvlc_lone(u_q0)
    } else {
        sink.non_initial_uvlc(u_q0, 0)
    }
}
