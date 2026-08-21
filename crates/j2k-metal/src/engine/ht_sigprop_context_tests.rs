// SPDX-License-Identifier: MIT OR Apache-2.0

//! Golden model for `SigProp` saved-row context at adjacent x-quads.

const HT_CLEANUP_SOURCE: &str = include_str!("../ht_cleanup.metal");

fn save_sigprop_context(prev_row: &mut [u16], index: usize, new_sig: u32, current_sig: u32) {
    let combined = new_sig | (current_sig & 0xffff);
    prev_row[index] = u16::try_from(combined).expect("saved-row sigma fits u16");
}

#[test]
fn sigprop_context_update_does_not_overwrite_next_x_quad() {
    let mut prev_row = [0x1111, 0x2222, 0x3333];
    save_sigprop_context(&mut prev_row, 0, 0x0000_0040, 0xaaaa_0001);

    assert_eq!(prev_row, [0x0041, 0x2222, 0x3333]);
}

#[test]
fn metal_sigprop_context_saves_only_current_quad() {
    assert!(
        HT_CLEANUP_SOURCE.contains("const uint combined_sig = new_sig | (cs & 0xFFFFu);"),
        "SigProp must exclude the next x-quad's lookahead sigma"
    );
    assert!(
        !HT_CLEANUP_SOURCE.contains("prev_row_sig[idx + 1u] = ushort(combined_sig >> 16u);"),
        "SigProp must preserve the next x-quad's saved above-row context"
    );
}
