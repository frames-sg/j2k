// SPDX-License-Identifier: MIT OR Apache-2.0

use metal::Buffer;

use crate::compute::{
    direct_scratch::{take_f32_scratch_buffer, DirectScratchBuffer},
    MetalRuntime,
};
use crate::Error;

pub(super) struct FinalComponentPlane {
    retained: Option<RetainedComponentPlane>,
}

struct RetainedComponentPlane {
    buffer: Buffer,
    dimensions: (u32, u32),
    len: usize,
}

impl FinalComponentPlane {
    pub(super) const fn empty() -> Self {
        Self { retained: None }
    }

    pub(super) fn buffer_for_store(
        &mut self,
        runtime: &MetalRuntime,
        dimensions: (u32, u32),
        len: usize,
        bytes: usize,
        scratch_buffers: &mut Vec<DirectScratchBuffer>,
    ) -> Result<Buffer, Error> {
        if let Some(retained) = &self.retained {
            validate_later_component_store(retained.dimensions, retained.len, dimensions, len)?;
            if retained.buffer.length() < bytes as u64 {
                return Err(Error::MetalStateInvariant {
                    state: "J2K MetalDirect component tile store",
                    reason: "retained final component plane is smaller than the validated store",
                });
            }
            return Ok(retained.buffer.clone());
        }

        let output = take_f32_scratch_buffer(runtime, len)?;
        let buffer = output.buffer.clone();
        scratch_buffers.push(output);
        self.retained = Some(RetainedComponentPlane {
            buffer: buffer.clone(),
            dimensions,
            len,
        });
        Ok(buffer)
    }

    pub(super) fn finish(self) -> Result<Buffer, Error> {
        self.retained
            .map(|retained| retained.buffer)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K MetalDirect component plan did not produce a stored plane"
                    .to_string(),
            })
    }
}

fn validate_later_component_store(
    retained_dimensions: (u32, u32),
    retained_len: usize,
    dimensions: (u32, u32),
    len: usize,
) -> Result<(), Error> {
    if retained_dimensions != dimensions || retained_len != len {
        return Err(Error::MetalStateInvariant {
            state: "J2K MetalDirect component tile store",
            reason: "later tile store changed the final component plane shape",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_later_component_store;
    use crate::Error;

    #[test]
    fn later_component_store_requires_exact_final_plane_geometry() {
        assert!(validate_later_component_store((4, 2), 8, (4, 2), 8).is_ok());

        let changed_shape = validate_later_component_store((4, 2), 8, (2, 4), 8)
            .expect_err("the same element count with different dimensions must be rejected");
        assert!(matches!(
            changed_shape,
            Error::MetalStateInvariant {
                state: "J2K MetalDirect component tile store",
                reason: "later tile store changed the final component plane shape",
            }
        ));

        let changed_len = validate_later_component_store((4, 2), 7, (4, 2), 8)
            .expect_err("matching dimensions with a different exact length must be rejected");
        assert!(matches!(
            changed_len,
            Error::MetalStateInvariant {
                state: "J2K MetalDirect component tile store",
                reason: "later tile store changed the final component plane shape",
            }
        ));

        let smaller_plane = validate_later_component_store((4, 2), 8, (2, 2), 4)
            .expect_err("a smaller later store must not reuse the first final plane");
        assert!(matches!(
            smaller_plane,
            Error::MetalStateInvariant {
                state: "J2K MetalDirect component tile store",
                reason: "later tile store changed the final component plane shape",
            }
        ));
    }
}
