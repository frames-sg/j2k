// SPDX-License-Identifier: MIT OR Apache-2.0

use metal::Buffer;

use crate::{Error, MetalDirectFallbackReason};

use super::{
    direct_buffers::{checked_buffer_read, checked_buffer_slice},
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
            let statuses =
                checked_buffer_slice::<J2kClassicStatus>(&buffer, len, "classic direct status")?;
            if let Some(status) = statuses
                .iter()
                .copied()
                .find(|status| status.code != J2K_CLASSIC_STATUS_OK)
            {
                return Err(decode_classic_status_error(status));
            }
        }
        DirectStatusCheck::Ht { buffer, len } => {
            let statuses = checked_buffer_slice::<J2kHtStatus>(&buffer, len, "HT direct status")?;
            if let Some(status) = statuses
                .iter()
                .copied()
                .find(|status| status.code != J2K_HT_STATUS_OK)
            {
                return Err(decode_ht_status_error(status));
            }
        }
        DirectStatusCheck::Idwt(buffer) => {
            let status = checked_buffer_read::<J2kIdwtStatus>(&buffer, "IDWT direct status")?;
            if status.code != J2K_IDWT_STATUS_OK {
                return Err(decode_idwt_status_error(status));
            }
        }
        DirectStatusCheck::Mct(buffer) => {
            let status = checked_buffer_read::<J2kMctStatus>(&buffer, "MCT direct status")?;
            if status.code != J2K_MCT_STATUS_OK {
                return Err(decode_mct_status_error(status));
            }
        }
    }

    Ok(())
}

pub(super) fn decode_classic_status_error(status: J2kClassicStatus) -> Error {
    if status.code == J2K_CLASSIC_STATUS_UNSUPPORTED {
        return Error::MetalDirectFallback {
            message: format!(
                "classic J2K Metal kernel unsupported classic kernel input (detail={})",
                status.detail
            ),
            reason: MetalDirectFallbackReason::UnsupportedRuntimeInput,
        };
    }
    let kind = match status.code {
        J2K_CLASSIC_STATUS_FAIL => "decode failure",
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
    if status.code == J2K_HT_STATUS_UNSUPPORTED {
        return Error::MetalDirectFallback {
            message: format!(
                "HTJ2K Metal kernel unsupported HT kernel input (detail={})",
                status.detail
            ),
            reason: MetalDirectFallbackReason::UnsupportedRuntimeInput,
        };
    }
    let kind = match status.code {
        J2K_HT_STATUS_FAIL => "decode failure",
        _ => "unexpected HT kernel status",
    };
    Error::MetalKernel {
        message: format!("HTJ2K Metal kernel {kind} (detail={})", status.detail),
    }
}
