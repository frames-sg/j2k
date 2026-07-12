// SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "markdown/html.rs"]
mod html;

use html::HtmlBlock;

pub(super) fn content_lines(source: &str) -> Vec<&str> {
    let mut content = Vec::new();
    let mut open_fence = None;
    let mut open_html: Option<HtmlBlock> = None;
    for raw_line in source.lines() {
        if let Some(block) = open_html {
            if block.closes_on(raw_line.trim_end()) {
                open_html = None;
            }
            continue;
        }
        let Some(line) = non_indented_line(raw_line) else {
            continue;
        };
        if let Some((marker, width)) = open_fence {
            if closes_fence(line, marker, width) {
                open_fence = None;
            }
            continue;
        }
        if let Some(fence) = opening_fence(line) {
            open_fence = Some(fence);
            continue;
        }
        if let Some(block) = HtmlBlock::opening(line) {
            if !block.closes_on(line) {
                open_html = Some(block);
            }
            continue;
        }
        content.push(line);
    }
    content
}

fn non_indented_line(raw_line: &str) -> Option<&str> {
    let line = raw_line.trim_end();
    let indent = line.bytes().take_while(|byte| *byte == b' ').count();
    if indent >= 4 || line.as_bytes().get(indent) == Some(&b'\t') {
        None
    } else {
        Some(&line[indent..])
    }
}

fn opening_fence(line: &str) -> Option<(u8, usize)> {
    let marker = *line.as_bytes().first()?;
    if !matches!(marker, b'`' | b'~') {
        return None;
    }
    let width = line.bytes().take_while(|byte| *byte == marker).count();
    if width < 3 || (marker == b'`' && line[width..].contains('`')) {
        None
    } else {
        Some((marker, width))
    }
}

fn closes_fence(line: &str, marker: u8, opening_width: usize) -> bool {
    let width = line.bytes().take_while(|byte| *byte == marker).count();
    width >= opening_width && line[width..].trim().is_empty()
}
