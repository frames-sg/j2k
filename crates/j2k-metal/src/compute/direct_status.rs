// SPDX-License-Identifier: Apache-2.0

use metal::Buffer;

use crate::Error;

use super::{
    J2kClassicStatus, J2kHtStatus, J2kIdwtStatus, J2kMctStatus, J2K_CLASSIC_STATUS_FAIL,
    J2K_CLASSIC_STATUS_OK, J2K_CLASSIC_STATUS_UNSUPPORTED, J2K_HT_STATUS_FAIL, J2K_HT_STATUS_OK,
    J2K_HT_STATUS_UNSUPPORTED, J2K_IDWT_STATUS_FAIL, J2K_IDWT_STATUS_OK, J2K_MCT_STATUS_FAIL,
    J2K_MCT_STATUS_OK,
};

pub(super) enum DirectStatusCheck {
    Classic { buffer: Buffer, len: usize },
    Ht { buffer: Buffer, len: usize },
    Idwt(Buffer),
    Mct(Buffer),
}

pub(super) fn validate_direct_status(status_check: DirectStatusCheck) -> Result<(), Error> {
    match status_check {
        DirectStatusCheck::Classic { buffer, len } => {
            // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
            let statuses = unsafe {
                core::slice::from_raw_parts(buffer.contents().cast::<J2kClassicStatus>(), len)
            };
            if let Some(status) = statuses
                .iter()
                .copied()
                .find(|status| status.code != J2K_CLASSIC_STATUS_OK)
            {
                return Err(decode_classic_status_error(status));
            }
        }
        DirectStatusCheck::Ht { buffer, len } => {
            // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
            let statuses = unsafe {
                core::slice::from_raw_parts(buffer.contents().cast::<J2kHtStatus>(), len)
            };
            if let Some(status) = statuses
                .iter()
                .copied()
                .find(|status| status.code != J2K_HT_STATUS_OK)
            {
                return Err(decode_ht_status_error(status));
            }
        }
        DirectStatusCheck::Idwt(buffer) => {
            // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
            let status = unsafe { buffer.contents().cast::<J2kIdwtStatus>().read() };
            if status.code != J2K_IDWT_STATUS_OK {
                return Err(decode_idwt_status_error(status));
            }
        }
        DirectStatusCheck::Mct(buffer) => {
            // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
            let status = unsafe { buffer.contents().cast::<J2kMctStatus>().read() };
            if status.code != J2K_MCT_STATUS_OK {
                return Err(decode_mct_status_error(status));
            }
        }
    }

    Ok(())
}

pub(super) fn decode_classic_status_error(status: J2kClassicStatus) -> Error {
    let kind = match status.code {
        J2K_CLASSIC_STATUS_FAIL => "decode failure",
        J2K_CLASSIC_STATUS_UNSUPPORTED => "unsupported classic kernel input",
        _ => "unexpected classic kernel status",
    };
    Error::MetalKernel {
        message: format!("classic J2K Metal kernel {kind} (detail={})", status.detail),
    }
}

pub(super) fn decode_idwt_status_error(status: J2kIdwtStatus) -> Error {
    let kind = match status.code {
        J2K_IDWT_STATUS_FAIL => "decode failure",
        _ => "unexpected IDWT kernel status",
    };
    Error::MetalKernel {
        message: format!("J2K Metal IDWT kernel {kind} (detail={})", status.detail),
    }
}

pub(super) fn decode_mct_status_error(status: J2kMctStatus) -> Error {
    let kind = match status.code {
        J2K_MCT_STATUS_FAIL => "decode failure",
        _ => "unexpected inverse MCT kernel status",
    };
    Error::MetalKernel {
        message: format!(
            "J2K Metal inverse MCT kernel {kind} (detail={})",
            status.detail
        ),
    }
}

pub(super) fn decode_ht_status_error(status: J2kHtStatus) -> Error {
    let kind = match status.code {
        J2K_HT_STATUS_FAIL => "decode failure",
        J2K_HT_STATUS_UNSUPPORTED => "unsupported HT kernel input",
        _ => "unexpected HT kernel status",
    };
    Error::MetalKernel {
        message: format!("HTJ2K Metal kernel {kind} (detail={})", status.detail),
    }
}
