// SPDX-License-Identifier: MIT OR Apache-2.0

use super::cleanup::CleanupCoefficientSource;

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::inline_always,
    reason = "shift bounds keep the intermediate in 32 bits and its bit width in i32 on the per-sample hot path"
)]
#[inline(always)]
fn process_sample(
    slot: usize,
    value: u32,
    p: u32,
    rho_acc: &mut i32,
    e_q: &mut [i32; 8],
    e_qmax: &mut i32,
    s: &mut [u32; 8],
) {
    let magnitude = value & 0x7FFF_FFFF;
    let mut val = (u64::from(magnitude) << 1) >> p;
    val &= !1u64;
    if val != 0 {
        *rho_acc |= 1 << (slot & 0x3);
        val -= 1;
        let val_u32 = val as u32;
        e_q[slot] = (u32::BITS - val_u32.leading_zeros()) as i32;
        *e_qmax = (*e_qmax).max(e_q[slot]);
        val -= 1;
        s[slot] = (val as u32) + (value >> 31);
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

pub(super) struct FirstQuadPairRequest<'a, C: CleanupCoefficientSource + ?Sized> {
    pub(super) coefficients: &'a C,
    pub(super) stride: usize,
    pub(super) height: usize,
    pub(super) p: u32,
    pub(super) sp: &'a mut usize,
    pub(super) x: usize,
    pub(super) markers: QuadMarkerRows<'a>,
    pub(super) state: QuadPairState<'a>,
}

pub(super) struct NonInitialQuadPairRequest<'a, C: CleanupCoefficientSource + ?Sized> {
    pub(super) coefficients: &'a C,
    pub(super) stride: usize,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) y: usize,
    pub(super) p: u32,
    pub(super) sp: &'a mut usize,
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
    clippy::too_many_lines,
    reason = "rho is a nonnegative four-bit mask and this normative quad walk stays fused per sink"
)]
#[inline(always)]
pub(super) fn first_quad_pair<C, S>(
    request: FirstQuadPairRequest<'_, C>,
    sink: &mut S,
) -> Result<(), S::Error>
where
    C: CleanupCoefficientSource + ?Sized,
    S: QuadSink,
{
    let FirstQuadPairRequest {
        coefficients,
        stride,
        height,
        p,
        sp,
        x,
        markers,
        state,
    } = request;
    let lep = x / 2;
    let lcxp = x / 2;

    process_sample(
        0,
        coefficients.aligned_value(*sp),
        p,
        &mut state.rho[0],
        state.e_q,
        &mut state.e_qmax[0],
        state.s,
    );
    process_sample(
        1,
        if height > 1 {
            coefficients.aligned_value(*sp + stride)
        } else {
            0
        },
        p,
        &mut state.rho[0],
        state.e_q,
        &mut state.e_qmax[0],
        state.s,
    );
    *sp += 1;

    if x + 1 < stride {
        process_sample(
            2,
            coefficients.aligned_value(*sp),
            p,
            &mut state.rho[0],
            state.e_q,
            &mut state.e_qmax[0],
            state.s,
        );
        process_sample(
            3,
            if height > 1 {
                coefficients.aligned_value(*sp + stride)
            } else {
                0
            },
            p,
            &mut state.rho[0],
            state.e_q,
            &mut state.e_qmax[0],
            state.s,
        );
        *sp += 1;
    }

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

    if x + 2 < stride {
        process_sample(
            4,
            coefficients.aligned_value(*sp),
            p,
            &mut state.rho[1],
            state.e_q,
            &mut state.e_qmax[1],
            state.s,
        );
        process_sample(
            5,
            if height > 1 {
                coefficients.aligned_value(*sp + stride)
            } else {
                0
            },
            p,
            &mut state.rho[1],
            state.e_q,
            &mut state.e_qmax[1],
            state.s,
        );
        *sp += 1;

        if x + 3 < stride {
            process_sample(
                6,
                coefficients.aligned_value(*sp),
                p,
                &mut state.rho[1],
                state.e_q,
                &mut state.e_qmax[1],
                state.s,
            );
            process_sample(
                7,
                if height > 1 {
                    coefficients.aligned_value(*sp + stride)
                } else {
                    0
                },
                p,
                &mut state.rho[1],
                state.e_q,
                &mut state.e_qmax[1],
                state.s,
            );
            *sp += 1;
        }

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
    clippy::too_many_lines,
    reason = "bounded HT quad fields feed packed marker rows in this normative fused quad walk"
)]
#[inline(always)]
pub(super) fn non_initial_quad_pair<C, S>(
    request: NonInitialQuadPairRequest<'_, C>,
    sink: &mut S,
) -> Result<(), S::Error>
where
    C: CleanupCoefficientSource + ?Sized,
    S: QuadSink,
{
    let NonInitialQuadPairRequest {
        coefficients,
        stride,
        width,
        height,
        y,
        p,
        sp,
        x,
        markers,
        lep,
        lcxp,
        max_e,
        state,
    } = request;
    process_sample(
        0,
        coefficients.aligned_value(*sp),
        p,
        &mut state.rho[0],
        state.e_q,
        &mut state.e_qmax[0],
        state.s,
    );
    process_sample(
        1,
        if y + 1 < height {
            coefficients.aligned_value(*sp + stride)
        } else {
            0
        },
        p,
        &mut state.rho[0],
        state.e_q,
        &mut state.e_qmax[0],
        state.s,
    );
    *sp += 1;

    if x + 1 < width {
        process_sample(
            2,
            coefficients.aligned_value(*sp),
            p,
            &mut state.rho[0],
            state.e_q,
            &mut state.e_qmax[0],
            state.s,
        );
        process_sample(
            3,
            if y + 1 < height {
                coefficients.aligned_value(*sp + stride)
            } else {
                0
            },
            p,
            &mut state.rho[0],
            state.e_q,
            &mut state.e_qmax[0],
            state.s,
        );
        *sp += 1;
    }

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
        process_sample(
            4,
            coefficients.aligned_value(*sp),
            p,
            &mut state.rho[1],
            state.e_q,
            &mut state.e_qmax[1],
            state.s,
        );
        process_sample(
            5,
            if y + 1 < height {
                coefficients.aligned_value(*sp + stride)
            } else {
                0
            },
            p,
            &mut state.rho[1],
            state.e_q,
            &mut state.e_qmax[1],
            state.s,
        );
        *sp += 1;

        if x + 3 < width {
            process_sample(
                6,
                coefficients.aligned_value(*sp),
                p,
                &mut state.rho[1],
                state.e_q,
                &mut state.e_qmax[1],
                state.s,
            );
            process_sample(
                7,
                if y + 1 < height {
                    coefficients.aligned_value(*sp + stride)
                } else {
                    0
                },
                p,
                &mut state.rho[1],
                state.e_q,
                &mut state.e_qmax[1],
                state.s,
            );
            *sp += 1;
        }

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
