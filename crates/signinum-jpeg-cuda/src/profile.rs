// SPDX-License-Identifier: Apache-2.0

pub(crate) fn gpu_route_profile_enabled() -> bool {
    signinum_profile::gpu_route_profile_enabled()
}

pub(crate) fn emit_gpu_route_profile<K, V>(codec: &str, op: &str, path: &str, fields: &[(K, V)])
where
    K: AsRef<str>,
    V: AsRef<str>,
{
    debug_assert_eq!(op, "gpu_route");
    signinum_profile::emit_gpu_route_profile(codec, path, fields);
}
