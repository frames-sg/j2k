// SPDX-License-Identifier: MIT OR Apache-2.0

use super::Error;

pub(super) fn classic_status_sources(
    job_count: usize,
    sources: Option<Vec<usize>>,
) -> Result<Vec<usize>, Error> {
    if let Some(sources) = sources {
        if sources.len() != job_count {
            return Err(Error::MetalStateInvariant {
                state: "classic J2K Metal batch status attribution",
                reason: "classic source identity count does not match status count",
            });
        }
        return Ok(sources);
    }

    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("classic J2K Metal status attribution");
    Ok(budget.try_filled(job_count, 0usize, "classic J2K Metal status sources")?)
}

pub(super) fn repeated_classic_status_sources(
    job_count: usize,
    total_job_count: usize,
) -> Result<Vec<usize>, Error> {
    if job_count == 0 || !total_job_count.is_multiple_of(job_count) {
        return Err(Error::MetalStateInvariant {
            state: "classic J2K Metal repeated status attribution",
            reason: "total status count is not a whole number of source images",
        });
    }
    let source_count = total_job_count / job_count;
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "classic J2K Metal repeated status attribution",
    );
    let mut sources =
        budget.try_vec(total_job_count, "classic J2K Metal repeated status sources")?;
    for source_index in 0..source_count {
        sources.extend(core::iter::repeat_n(source_index, job_count));
    }
    Ok(sources)
}

#[cfg(test)]
mod tests {
    use super::repeated_classic_status_sources;

    #[test]
    fn repeated_classic_status_sources_follow_shader_linearization() {
        assert_eq!(
            repeated_classic_status_sources(3, 6).expect("two sources with three jobs each"),
            [0, 0, 0, 1, 1, 1]
        );
        assert!(repeated_classic_status_sources(0, 0).is_err());
        assert!(repeated_classic_status_sources(3, 5).is_err());
    }
}
