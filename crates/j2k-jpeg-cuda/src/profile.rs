// SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) fn emit_optional_gpu_route_fields<const N: usize>(
    operation: &'static str,
    build: impl FnOnce() -> j2k_profile::ProfileResult<[j2k_profile::ProfileField; N]>,
    emit: impl FnOnce([j2k_profile::ProfileField; N]),
) {
    if !j2k_profile::gpu_route_profile_enabled() {
        return;
    }
    match build() {
        Ok(fields) => emit(fields),
        Err(error) => {
            j2k_profile::emit_profile_error(operation, &error);
        }
    }
}
