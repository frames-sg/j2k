// SPDX-License-Identifier: MIT OR Apache-2.0

//! Memory available to this process, used to size aggregate batch admission.

use std::sync::OnceLock;

/// Physical memory, lowered by the Linux cgroup memory limit of this process
/// when one is set. `None` when the host reports neither.
///
/// The value is read once per process.
pub(super) fn process_memory_limit_bytes() -> Option<u64> {
    static LIMIT: OnceLock<Option<u64>> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        min_known(
            j2k_core::host_physical_memory_bytes(),
            cgroup_memory_limit_bytes(),
        )
    })
}

fn min_known(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (left, right) => left.or(right),
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn cgroup_memory_limit_bytes() -> Option<u64> {
    const V2_ROOT: &str = "/sys/fs/cgroup";
    const V1_MEMORY_ROOT: &str = "/sys/fs/cgroup/memory";

    let membership = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let v2 = cgroup_v2_path(&membership)
        .and_then(|path| smallest_limit_along(V2_ROOT, path, "memory.max"));
    let v1 = cgroup_v1_memory_path(&membership)
        .and_then(|path| smallest_limit_along(V1_MEMORY_ROOT, path, "memory.limit_in_bytes"));
    min_known(v2, v1)
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
const fn cgroup_memory_limit_bytes() -> Option<u64> {
    None
}

/// Smallest limit set on the cgroup at `path` or any of its ancestors.
///
/// Inside a cgroup namespace the process sees its own cgroup as `/`, so the
/// root file holds the container limit.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn smallest_limit_along(root: &str, path: &str, file: &str) -> Option<u64> {
    cgroup_ancestors(path)
        .filter_map(|dir| std::fs::read_to_string(format!("{root}{dir}/{file}")).ok())
        .filter_map(|contents| parse_cgroup_limit(&contents))
        .min()
}

/// `path` followed by each ancestor, ending with the root as `""`.
#[cfg(any(target_os = "linux", target_os = "android", test))]
fn cgroup_ancestors(path: &str) -> impl Iterator<Item = &str> {
    let mut next = Some(path.trim_end_matches('/'));
    core::iter::from_fn(move || {
        let current = next?;
        next = (!current.is_empty())
            .then(|| current.rsplit_once('/').map_or("", |(parent, _)| parent));
        Some(current)
    })
}

/// A limit file holds a byte count, or `max` when no limit is set.
#[cfg(any(target_os = "linux", target_os = "android", test))]
fn parse_cgroup_limit(contents: &str) -> Option<u64> {
    contents
        .trim()
        .parse::<u64>()
        .ok()
        .filter(|&bytes| bytes > 0)
}

/// The unified (v2) hierarchy appears in `/proc/self/cgroup` as `0::<path>`.
#[cfg(any(target_os = "linux", target_os = "android", test))]
fn cgroup_v2_path(membership: &str) -> Option<&str> {
    membership
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .filter(|path| path.starts_with('/'))
}

/// A v1 hierarchy line is `<id>:<controllers>:<path>`; the memory controller
/// may share a line with others.
#[cfg(any(target_os = "linux", target_os = "android", test))]
fn cgroup_v1_memory_path(membership: &str) -> Option<&str> {
    membership.lines().find_map(|line| {
        let mut fields = line.splitn(3, ':');
        let _id = fields.next()?;
        let controllers = fields.next()?;
        let path = fields.next()?;
        (controllers.split(',').any(|name| name == "memory") && path.starts_with('/'))
            .then_some(path)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cgroup_limit_files_parse_byte_counts_and_reject_unlimited() {
        assert_eq!(parse_cgroup_limit("1073741824\n"), Some(1 << 30));
        assert_eq!(parse_cgroup_limit("max\n"), None);
        assert_eq!(parse_cgroup_limit("0\n"), None);
        assert_eq!(parse_cgroup_limit(""), None);
        assert_eq!(parse_cgroup_limit("-1"), None);
    }

    #[test]
    fn cgroup_membership_paths_are_found_for_v2_and_v1() {
        let v2 = "0::/system.slice/viewer.service\n";
        assert_eq!(cgroup_v2_path(v2), Some("/system.slice/viewer.service"));
        assert_eq!(cgroup_v1_memory_path(v2), None);

        let v1 = "12:cpu,cpuacct:/docker/abc\n7:memory:/docker/abc\n0::/\n";
        assert_eq!(cgroup_v1_memory_path(v1), Some("/docker/abc"));
        assert_eq!(cgroup_v2_path(v1), Some("/"));

        let shared = "4:cpuset,memory:/kubepods/pod1\n";
        assert_eq!(cgroup_v1_memory_path(shared), Some("/kubepods/pod1"));
        assert_eq!(cgroup_v1_memory_path("4:memoryx:/a\n"), None);
    }

    #[test]
    fn cgroup_ancestors_walk_to_the_root() {
        assert_eq!(
            cgroup_ancestors("/kubepods/pod1/ctr").collect::<Vec<_>>(),
            ["/kubepods/pod1/ctr", "/kubepods/pod1", "/kubepods", ""]
        );
        assert_eq!(cgroup_ancestors("/").collect::<Vec<_>>(), [""]);
        assert_eq!(cgroup_ancestors("/a/").collect::<Vec<_>>(), ["/a", ""]);
    }

    #[test]
    fn the_smaller_known_limit_wins() {
        assert_eq!(min_known(Some(8), Some(4)), Some(4));
        assert_eq!(min_known(Some(8), None), Some(8));
        assert_eq!(min_known(None, Some(4)), Some(4));
        assert_eq!(min_known(None, None), None);
    }
}
