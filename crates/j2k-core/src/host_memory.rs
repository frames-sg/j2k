// SPDX-License-Identifier: MIT OR Apache-2.0

//! Physical memory size reported by the host operating system.

/// Total physical memory installed on the host, in bytes.
///
/// Returns `None` on targets without a supported query or when the operating
/// system reports an error. This is the machine total: container, cgroup, or
/// job-object limits that apply to the calling process are not considered.
#[doc(hidden)]
#[must_use]
pub fn host_physical_memory_bytes() -> Option<u64> {
    imp::physical_memory_bytes()
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod imp {
    pub(super) fn physical_memory_bytes() -> Option<u64> {
        let mut value = 0u64;
        let mut len = core::mem::size_of::<u64>();
        let name = b"hw.memsize\0";
        // SAFETY: `name` is NUL-terminated, and `sysctlbyname` writes at most
        // `len` bytes into `value`, a live u64, then stores the written size.
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr().cast(),
                (&raw mut value).cast(),
                &raw mut len,
                core::ptr::null_mut(),
                0,
            )
        };
        (rc == 0 && len == core::mem::size_of::<u64>() && value > 0).then_some(value)
    }
}

#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
mod imp {
    pub(super) fn physical_memory_bytes() -> Option<u64> {
        // SAFETY: `sysconf` only reads a system configuration value; it has no
        // pointer arguments or other memory-safety preconditions.
        let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
        // SAFETY: as above.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let pages = u64::try_from(pages).ok().filter(|&pages| pages > 0)?;
        let page_size = u64::try_from(page_size).ok().filter(|&size| size > 0)?;
        pages.checked_mul(page_size)
    }
}

#[cfg(windows)]
mod imp {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    pub(super) fn physical_memory_bytes() -> Option<u64> {
        let mut status = MEMORYSTATUSEX {
            dwLength: u32::try_from(core::mem::size_of::<MEMORYSTATUSEX>()).ok()?,
            ..MEMORYSTATUSEX::default()
        };
        // SAFETY: `status` is a live, writable `MEMORYSTATUSEX` whose
        // `dwLength` holds the structure size, as the API requires.
        let succeeded = unsafe { GlobalMemoryStatusEx(&raw mut status) };
        (succeeded != 0 && status.ullTotalPhys > 0).then_some(status.ullTotalPhys)
    }
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    windows
)))]
mod imp {
    pub(super) const fn physical_memory_bytes() -> Option<u64> {
        None
    }
}

#[cfg(test)]
mod tests {
    #[cfg(any(target_os = "linux", target_os = "macos", windows))]
    #[test]
    fn supported_desktop_hosts_report_physical_memory() {
        let bytes = super::host_physical_memory_bytes().expect("host physical memory");
        // Any host able to build the workspace has at least 256 MiB.
        assert!(bytes >= 256 * 1024 * 1024, "{bytes}");
    }
}
