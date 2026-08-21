// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_cuda_runtime::CudaError;

pub(crate) fn ensure_htj2k_encode_ptx_built() -> Result<(), CudaError> {
    #[cfg(feature = "cuda-oxide-htj2k-encode")]
    {
        if cfg!(j2k_cuda_oxide_htj2k_encode_built) {
            Ok(())
        } else {
            Err(CudaError::InvalidArgument {
                message: "cuda-oxide HTJ2K encode PTX was not built; set J2K_REQUIRE_CUDA_OXIDE_BUILD on a Linux cuda-oxide host to require it".to_string(),
            })
        }
    }

    #[cfg(not(feature = "cuda-oxide-htj2k-encode"))]
    Err(CudaError::InvalidArgument {
        message: "CUDA Oxide PTX was not built for HTJ2K encode; enable j2k-cuda-j2k-engine/cuda-oxide-htj2k-encode or an adapter feature that implies it"
            .to_string(),
    })
}

pub(crate) fn ensure_j2k_encode_ptx_built() -> Result<(), CudaError> {
    #[cfg(feature = "cuda-oxide-j2k-encode")]
    {
        if cfg!(j2k_cuda_oxide_j2k_encode_built) {
            Ok(())
        } else {
            Err(CudaError::InvalidArgument {
                message: "cuda-oxide J2K encode PTX was not built; set J2K_REQUIRE_CUDA_OXIDE_BUILD on a Linux cuda-oxide host to require it".to_string(),
            })
        }
    }

    #[cfg(not(feature = "cuda-oxide-j2k-encode"))]
    Err(CudaError::InvalidArgument {
        message: "CUDA Oxide PTX was not built for J2K encode; enable j2k-cuda-j2k-engine/cuda-oxide-j2k-encode or an adapter feature that implies it"
            .to_string(),
    })
}

pub(crate) fn ensure_j2k_ml_ptx_built() -> Result<(), CudaError> {
    #[cfg(feature = "cuda-oxide-j2k-ml")]
    {
        if cfg!(j2k_cuda_oxide_j2k_ml_built) {
            Ok(())
        } else {
            Err(CudaError::InvalidArgument {
                message: "cuda-oxide j2k-ml PTX was not built; set J2K_REQUIRE_CUDA_OXIDE_BUILD on a Linux cuda-oxide host to require it".to_string(),
            })
        }
    }

    #[cfg(not(feature = "cuda-oxide-j2k-ml"))]
    Err(CudaError::InvalidArgument {
        message: "CUDA Oxide PTX was not built for j2k-ml; enable j2k-cuda-j2k-engine/cuda-oxide-j2k-ml or an adapter feature that implies it"
            .to_string(),
    })
}

pub(crate) fn ensure_j2k_classic_decode_ptx_built() -> Result<(), CudaError> {
    #[cfg(feature = "cuda-oxide-j2k-classic-decode")]
    {
        if cfg!(j2k_cuda_oxide_j2k_classic_decode_built) {
            Ok(())
        } else {
            Err(CudaError::InvalidArgument {
                message: "cuda-oxide classic J2K decode PTX was not built; set J2K_REQUIRE_CUDA_OXIDE_BUILD on a Linux cuda-oxide host to require it".to_string(),
            })
        }
    }

    #[cfg(not(feature = "cuda-oxide-j2k-classic-decode"))]
    Err(CudaError::InvalidArgument {
        message: "CUDA Oxide PTX was not built for classic J2K decode; enable j2k-cuda-j2k-engine/cuda-oxide-j2k-classic-decode or an adapter feature that implies it"
            .to_string(),
    })
}

pub(crate) fn ensure_htj2k_decode_ptx_built() -> Result<(), CudaError> {
    #[cfg(feature = "cuda-oxide-htj2k-decode")]
    {
        if cfg!(j2k_cuda_oxide_htj2k_decode_built) {
            Ok(())
        } else {
            Err(CudaError::InvalidArgument {
                message: "cuda-oxide HTJ2K decode PTX was not built; set J2K_REQUIRE_CUDA_OXIDE_BUILD on a Linux cuda-oxide host to require it".to_string(),
            })
        }
    }

    #[cfg(not(feature = "cuda-oxide-htj2k-decode"))]
    Err(CudaError::InvalidArgument {
        message: "CUDA Oxide PTX was not built for HTJ2K decode; enable j2k-cuda-j2k-engine/cuda-oxide-htj2k-decode or an adapter feature that implies it"
            .to_string(),
    })
}

pub(crate) fn ensure_j2k_dequantize_ptx_built() -> Result<(), CudaError> {
    #[cfg(feature = "cuda-oxide-j2k-dequantize")]
    {
        if cfg!(j2k_cuda_oxide_j2k_dequantize_built) {
            Ok(())
        } else {
            Err(CudaError::InvalidArgument {
                message: "cuda-oxide J2K dequantize PTX was not built; set J2K_REQUIRE_CUDA_OXIDE_BUILD on a Linux cuda-oxide host to require it".to_string(),
            })
        }
    }

    #[cfg(not(feature = "cuda-oxide-j2k-dequantize"))]
    Err(CudaError::InvalidArgument {
        message: "CUDA Oxide PTX was not built for J2K dequantize; enable j2k-cuda-j2k-engine/cuda-oxide-j2k-dequantize or an adapter feature that implies it"
            .to_string(),
    })
}

pub(crate) fn ensure_j2k_decode_store_ptx_built() -> Result<(), CudaError> {
    #[cfg(feature = "cuda-oxide-j2k-decode-store")]
    {
        if cfg!(j2k_cuda_oxide_j2k_decode_store_built) {
            Ok(())
        } else {
            Err(CudaError::InvalidArgument {
                message: "cuda-oxide J2K decode store PTX was not built; set J2K_REQUIRE_CUDA_OXIDE_BUILD on a Linux cuda-oxide host to require it".to_string(),
            })
        }
    }

    #[cfg(not(feature = "cuda-oxide-j2k-decode-store"))]
    Err(CudaError::InvalidArgument {
        message: "CUDA Oxide PTX was not built for J2K decode store; enable j2k-cuda-j2k-engine/cuda-oxide-j2k-decode-store or an adapter feature that implies it"
            .to_string(),
    })
}

pub(crate) fn ensure_j2k_idwt_ptx_built() -> Result<(), CudaError> {
    #[cfg(feature = "cuda-oxide-j2k-idwt")]
    {
        if cfg!(j2k_cuda_oxide_j2k_idwt_built) {
            Ok(())
        } else {
            Err(CudaError::InvalidArgument {
                message: "cuda-oxide J2K IDWT PTX was not built; set J2K_REQUIRE_CUDA_OXIDE_BUILD on a Linux cuda-oxide host to require it".to_string(),
            })
        }
    }

    #[cfg(not(feature = "cuda-oxide-j2k-idwt"))]
    Err(CudaError::InvalidArgument {
        message: "CUDA Oxide PTX was not built for J2K IDWT; enable j2k-cuda-j2k-engine/cuda-oxide-j2k-idwt or an adapter feature that implies it"
            .to_string(),
    })
}

#[cfg(all(
    test,
    feature = "cuda-oxide-htj2k-decode",
    not(j2k_cuda_oxide_htj2k_decode_built)
))]
mod tests {
    #[test]
    fn missing_htj2k_decode_build_error_mentions_strict_gate() {
        let error = super::ensure_htj2k_decode_ptx_built().expect_err("missing HTJ2K PTX");
        let message = error.to_string();
        assert!(message.contains("cuda-oxide HTJ2K decode PTX was not built"));
        assert!(message.contains("J2K_REQUIRE_CUDA_OXIDE_BUILD"));
    }

    #[cfg(all(
        feature = "cuda-oxide-htj2k-encode",
        not(j2k_cuda_oxide_htj2k_encode_built)
    ))]
    #[test]
    fn missing_htj2k_encode_build_error_mentions_strict_gate() {
        let error = super::ensure_htj2k_encode_ptx_built().expect_err("missing HTJ2K PTX");
        let message = error.to_string();
        assert!(message.contains("cuda-oxide HTJ2K encode PTX was not built"));
        assert!(message.contains("J2K_REQUIRE_CUDA_OXIDE_BUILD"));
    }
}
