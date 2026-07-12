// SPDX-License-Identifier: MIT OR Apache-2.0

#[derive(Clone, Copy)]
pub(super) enum HtmlBlock {
    Comment,
    RawTag(&'static str),
    BlankTerminated,
}

impl HtmlBlock {
    pub(super) fn opening(line: &str) -> Option<Self> {
        let lowercase = line.to_ascii_lowercase();
        if lowercase.starts_with("<!--") {
            return Some(Self::Comment);
        }
        for tag in ["pre", "script", "style", "textarea"] {
            if starts_open_tag(&lowercase, tag) {
                return Some(Self::RawTag(tag));
            }
        }
        starts_block_tag(&lowercase, "div").then_some(Self::BlankTerminated)
    }

    pub(super) fn closes_on(self, line: &str) -> bool {
        match self {
            Self::Comment => line.contains("-->"),
            Self::RawTag(tag) => line.to_ascii_lowercase().contains(&format!("</{tag}>")),
            Self::BlankTerminated => line.trim().is_empty(),
        }
    }
}

fn starts_open_tag(line: &str, tag: &str) -> bool {
    let Some(rest) = line
        .strip_prefix('<')
        .and_then(|rest| rest.strip_prefix(tag))
    else {
        return false;
    };
    has_tag_boundary(rest)
}

fn starts_block_tag(line: &str, tag: &str) -> bool {
    let Some(rest) = line.strip_prefix('<') else {
        return false;
    };
    let rest = rest.strip_prefix('/').unwrap_or(rest);
    rest.strip_prefix(tag).is_some_and(has_tag_boundary)
}

fn has_tag_boundary(rest: &str) -> bool {
    rest.is_empty()
        || rest
            .as_bytes()
            .first()
            .is_some_and(|byte| byte.is_ascii_whitespace() || matches!(*byte, b'/' | b'>'))
}
