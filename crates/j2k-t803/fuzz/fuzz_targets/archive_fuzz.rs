#![no_main]

use std::{io::Cursor, path::PathBuf, sync::OnceLock};

use j2k_t803::runner::{extract_selected_archive, ArchiveLimits};
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 4 * 1024 * 1024;
const LIMITS: ArchiveLimits = ArchiveLimits {
    max_entries: 128,
    max_entry_bytes: 1024 * 1024,
    max_total_bytes: 4 * 1024 * 1024,
};

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    let output = extraction_directory();
    let _ = extract_selected_archive(Cursor::new(data), output, &[], LIMITS);
});

fn extraction_directory() -> &'static PathBuf {
    static OUTPUT: OnceLock<PathBuf> = OnceLock::new();
    OUTPUT.get_or_init(|| {
        let path = std::env::temp_dir().join(format!(
            "j2k-t803-archive-fuzz-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).expect("create bounded archive fuzz output");
        path
    })
}
