// SPDX-License-Identifier: MIT OR Apache-2.0

use super::cleanup::CleanupCoefficientSource;

/// Samples per row a row pair holds: the widest code block plus the columns
/// a final, partial quad pair reads past the right edge.
const ROW_PAIR_COLUMNS: usize = 1024 + 4;

/// Two code-block rows as sample exponents and `MagSgn` values, zero past the
/// block's right and bottom edges, so the quad walk loads samples without
/// per-sample branches or edge tests.
pub(super) struct RowPair {
    exponents: [[u8; ROW_PAIR_COLUMNS]; 2],
    values: [[u32; ROW_PAIR_COLUMNS]; 2],
}

impl RowPair {
    pub(super) fn new() -> Self {
        Self {
            exponents: [[0; ROW_PAIR_COLUMNS]; 2],
            values: [[0; ROW_PAIR_COLUMNS]; 2],
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
        Ok(())
    }
}

/// Load quad `quad` (0 or 1) of the quad pair at column `x` into slots
/// `4 * quad..4 * quad + 4`: upper and lower sample of its left column, then
/// of its right column. An insignificant sample has exponent 0, and its value
/// is never emitted.
#[expect(
    clippy::inline_always,
    reason = "sample loads are fused into the per-quad cleanup hot path"
)]
#[inline(always)]
fn load_quad(
    rows: &RowPair,
    x: usize,
    quad: usize,
    rho: &mut i32,
    e_q: &mut [i32; 8],
    e_qmax: &mut i32,
    s: &mut [u32; 8],
) {
    for sample in 0..4 {
        let (row, column) = (sample & 1, x + 2 * quad + sample / 2);
        let exponent = i32::from(rows.exponents[row][column]);
        *rho |= i32::from(exponent != 0) << sample;
        e_q[4 * quad + sample] = exponent;
        *e_qmax = (*e_qmax).max(exponent);
        s[4 * quad + sample] = rows.values[row][column];
    }
}

pub(super) struct QuadPairState<'a> {
    pub(super) c_q0: &'a mut usize,
    pub(super) rho: &'a mut [i32; 2],
    pub(super) e_q: &'a mut [i32; 8],
    pub(super) e_qmax: &'a mut [i32; 2],
    pub(super) s: &'a mut [u32; 8],
}

pub(super) struct QuadMarkerRows<'a> {
    pub(super) e_val: &'a mut [u8; 514],
    pub(super) cx_val: &'a mut [u8; 514],
}

pub(super) struct FirstQuadPairRequest<'a> {
    pub(super) rows: &'a RowPair,
    pub(super) width: usize,
    pub(super) x: usize,
    pub(super) markers: QuadMarkerRows<'a>,
    pub(super) state: QuadPairState<'a>,
}

pub(super) struct NonInitialQuadPairRequest<'a> {
    pub(super) rows: &'a RowPair,
    pub(super) width: usize,
    pub(super) x: usize,
    pub(super) markers: QuadMarkerRows<'a>,
    pub(super) lep: &'a mut usize,
    pub(super) lcxp: &'a mut usize,
    pub(super) max_e: &'a mut i32,
    pub(super) state: QuadPairState<'a>,
}

pub(super) struct InitialQuadRow<'a> {
    pub(super) offset: usize,
    pub(super) c_q: usize,
    pub(super) rho: i32,
    pub(super) e_qmax: i32,
    pub(super) e_q: &'a [i32; 8],
    pub(super) s: &'a [u32; 8],
    pub(super) lep: usize,
    pub(super) lcxp: usize,
    pub(super) e_val: &'a mut [u8; 514],
    pub(super) cx_val: &'a mut [u8; 514],
}

pub(super) struct NonInitialQuadRow<'a> {
    pub(super) offset: usize,
    pub(super) c_q: usize,
    pub(super) rho: i32,
    pub(super) e_qmax: i32,
    pub(super) max_e: i32,
    pub(super) e_q: &'a [i32; 8],
    pub(super) s: &'a [u32; 8],
}

/// Per-quad operations that differ between the byte-emitting encoder and the
/// distribution collector; the quad-pair walking logic itself is shared by
/// `first_quad_pair` / `non_initial_quad_pair`.
pub(super) trait QuadSink {
    type Error;

    fn quad_initial(&mut self, row: InitialQuadRow<'_>) -> Result<i32, Self::Error>;

    fn quad_non_initial(&mut self, row: NonInitialQuadRow<'_>) -> Result<i32, Self::Error>;

    fn initial_uvlc_pair(&mut self, u_q0: i32, u_q1: i32) -> Result<(), Self::Error>;

    fn initial_uvlc_lone(&mut self, u_q0: i32) -> Result<(), Self::Error>;

    fn non_initial_uvlc(&mut self, u_q0: i32, u_q1: i32) -> Result<(), Self::Error>;
}

#[expect(
    clippy::cast_sign_loss,
    clippy::inline_always,
    reason = "rho is a nonnegative four-bit mask and this normative quad walk stays fused per sink"
)]
#[inline(always)]
pub(super) fn first_quad_pair<S: QuadSink>(
    request: FirstQuadPairRequest<'_>,
    sink: &mut S,
) -> Result<(), S::Error> {
    let FirstQuadPairRequest {
        rows,
        width,
        x,
        markers,
        state,
    } = request;
    let lep = x / 2;
    let lcxp = x / 2;

    load_quad(
        rows,
        x,
        0,
        &mut state.rho[0],
        state.e_q,
        &mut state.e_qmax[0],
        state.s,
    );
    let u_q0 = sink.quad_initial(InitialQuadRow {
        offset: 0,
        c_q: *state.c_q0,
        rho: state.rho[0],
        e_qmax: state.e_qmax[0],
        e_q: &*state.e_q,
        s: &*state.s,
        lep,
        lcxp,
        e_val: &mut *markers.e_val,
        cx_val: &mut *markers.cx_val,
    })?;

    if x + 2 < width {
        load_quad(
            rows,
            x,
            1,
            &mut state.rho[1],
            state.e_q,
            &mut state.e_qmax[1],
            state.s,
        );

        let c_q1 = ((state.rho[0] >> 1) | (state.rho[0] & 1)) as usize;
        let u_q1 = sink.quad_initial(InitialQuadRow {
            offset: 4,
            c_q: c_q1,
            rho: state.rho[1],
            e_qmax: state.e_qmax[1],
            e_q: &*state.e_q,
            s: &*state.s,
            lep: lep + 1,
            lcxp: lcxp + 1,
            e_val: &mut *markers.e_val,
            cx_val: &mut *markers.cx_val,
        })?;

        sink.initial_uvlc_pair(u_q0, u_q1)?;
        *state.c_q0 = ((state.rho[1] >> 1) | (state.rho[1] & 1)) as usize;
    } else {
        sink.initial_uvlc_lone(u_q0)?;
        *state.c_q0 = 0;
    }

    *state.rho = [0; 2];
    *state.e_q = [0; 8];
    *state.e_qmax = [0; 2];
    *state.s = [0; 8];

    Ok(())
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::inline_always,
    reason = "bounded HT quad fields feed packed marker rows in this normative fused quad walk"
)]
#[inline(always)]
pub(super) fn non_initial_quad_pair<S: QuadSink>(
    request: NonInitialQuadPairRequest<'_>,
    sink: &mut S,
) -> Result<(), S::Error> {
    let NonInitialQuadPairRequest {
        rows,
        width,
        x,
        markers,
        lep,
        lcxp,
        max_e,
        state,
    } = request;
    load_quad(
        rows,
        x,
        0,
        &mut state.rho[0],
        state.e_q,
        &mut state.e_qmax[0],
        state.s,
    );

    let prev_max = *max_e;
    let u_q0 = sink.quad_non_initial(NonInitialQuadRow {
        offset: 0,
        c_q: *state.c_q0,
        rho: state.rho[0],
        e_qmax: state.e_qmax[0],
        max_e: prev_max,
        e_q: &*state.e_q,
        s: &*state.s,
    })?;

    markers.e_val[*lep] = markers.e_val[*lep].max(state.e_q[1] as u8);
    *lep += 1;
    *max_e = i32::from(markers.e_val[*lep].max(markers.e_val[*lep + 1])) - 1;
    markers.e_val[*lep] = state.e_q[3] as u8;
    markers.cx_val[*lcxp] |= ((state.rho[0] & 2) >> 1) as u8;
    *lcxp += 1;
    let c_q1 = usize::from(markers.cx_val[*lcxp]) + (usize::from(markers.cx_val[*lcxp + 1]) << 2);
    markers.cx_val[*lcxp] = ((state.rho[0] & 8) >> 3) as u8;

    let mut u_q1 = 0;
    if x + 2 < width {
        load_quad(
            rows,
            x,
            1,
            &mut state.rho[1],
            state.e_q,
            &mut state.e_qmax[1],
            state.s,
        );

        let mut c_q1_local = c_q1;
        c_q1_local |= ((state.rho[0] & 4) >> 1) as usize;
        c_q1_local |= ((state.rho[0] & 8) >> 2) as usize;

        u_q1 = sink.quad_non_initial(NonInitialQuadRow {
            offset: 4,
            c_q: c_q1_local,
            rho: state.rho[1],
            e_qmax: state.e_qmax[1],
            max_e: *max_e,
            e_q: &*state.e_q,
            s: &*state.s,
        })?;

        markers.e_val[*lep] = markers.e_val[*lep].max(state.e_q[5] as u8);
        *lep += 1;
        *max_e = i32::from(markers.e_val[*lep].max(markers.e_val[*lep + 1])) - 1;
        markers.e_val[*lep] = state.e_q[7] as u8;
        markers.cx_val[*lcxp] |= ((state.rho[1] & 2) >> 1) as u8;
        *lcxp += 1;
        *state.c_q0 =
            usize::from(markers.cx_val[*lcxp]) + (usize::from(markers.cx_val[*lcxp + 1]) << 2);
        markers.cx_val[*lcxp] = ((state.rho[1] & 8) >> 3) as u8;

        *state.c_q0 |= ((state.rho[1] & 4) >> 1) as usize;
        *state.c_q0 |= ((state.rho[1] & 8) >> 2) as usize;
    } else {
        *state.c_q0 = 0;
    }

    sink.non_initial_uvlc(u_q0, u_q1)?;

    *state.rho = [0; 2];
    *state.e_q = [0; 8];
    *state.e_qmax = [0; 2];
    *state.s = [0; 8];

    Ok(())
}
