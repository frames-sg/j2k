// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::str::FromStr;

use proc_macro2::{TokenStream, TokenTree};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PanicMacroSite {
    pub(crate) name: &'static str,
    pub(crate) path: String,
    pub(crate) line: usize,
    pub(crate) column: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PanicMacroInventory {
    pub(crate) panic: usize,
    pub(crate) unreachable: usize,
    pub(crate) assert: usize,
    pub(crate) assert_eq: usize,
    pub(crate) assert_ne: usize,
    pub(crate) debug_assert: usize,
    pub(crate) debug_assert_eq: usize,
    pub(crate) debug_assert_ne: usize,
}

impl PanicMacroInventory {
    pub(crate) fn checked_add(self, other: Self) -> Result<Self, String> {
        Ok(Self {
            panic: checked_sum("panic", self.panic, other.panic)?,
            unreachable: checked_sum("unreachable", self.unreachable, other.unreachable)?,
            assert: checked_sum("assert", self.assert, other.assert)?,
            assert_eq: checked_sum("assert_eq", self.assert_eq, other.assert_eq)?,
            assert_ne: checked_sum("assert_ne", self.assert_ne, other.assert_ne)?,
            debug_assert: checked_sum("debug_assert", self.debug_assert, other.debug_assert)?,
            debug_assert_eq: checked_sum(
                "debug_assert_eq",
                self.debug_assert_eq,
                other.debug_assert_eq,
            )?,
            debug_assert_ne: checked_sum(
                "debug_assert_ne",
                self.debug_assert_ne,
                other.debug_assert_ne,
            )?,
        })
    }

    pub(crate) const fn entries(self) -> [(&'static str, usize); 8] {
        [
            ("panic!", self.panic),
            ("unreachable!", self.unreachable),
            ("assert!", self.assert),
            ("assert_eq!", self.assert_eq),
            ("assert_ne!", self.assert_ne),
            ("debug_assert!", self.debug_assert),
            ("debug_assert_eq!", self.debug_assert_eq),
            ("debug_assert_ne!", self.debug_assert_ne),
        ]
    }
}

impl fmt::Display for PanicMacroInventory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let entries = self
            .entries()
            .map(|(name, count)| format!("{name}={count}"))
            .join(", ");
        formatter.write_str(&entries)
    }
}

pub(crate) fn inventory_panic_macro_sites(
    path: &str,
    masked_source: &str,
) -> Result<(PanicMacroInventory, Vec<PanicMacroSite>), String> {
    let tokens = TokenStream::from_str(masked_source)
        .map_err(|error| format!("tokenize production Rust source `{path}`: {error}"))?;
    let mut inventory = PanicMacroInventory::default();
    let mut sites = Vec::new();
    scan_tokens(tokens, path, &mut inventory, &mut sites)?;
    Ok((inventory, sites))
}

fn scan_tokens(
    tokens: TokenStream,
    path: &str,
    inventory: &mut PanicMacroInventory,
    sites: &mut Vec<PanicMacroSite>,
) -> Result<(), String> {
    let mut tokens = tokens.into_iter().peekable();
    while let Some(token) = tokens.next() {
        match token {
            TokenTree::Group(group) => scan_tokens(group.stream(), path, inventory, sites)?,
            TokenTree::Ident(identifier)
                if tokens.peek().is_some_and(
                    |next| matches!(next, TokenTree::Punct(punct) if punct.as_char() == '!'),
                ) =>
            {
                if let Some(name) = increment_macro(inventory, &identifier.to_string())? {
                    let start = identifier.span().start();
                    sites.push(PanicMacroSite {
                        name,
                        path: path.to_string(),
                        line: start.line,
                        column: start.column.saturating_add(1),
                    });
                }
            }
            TokenTree::Ident(_) | TokenTree::Punct(_) | TokenTree::Literal(_) => {}
        }
    }
    Ok(())
}

fn increment_macro(
    inventory: &mut PanicMacroInventory,
    name: &str,
) -> Result<Option<&'static str>, String> {
    let (canonical_name, counter) = match name {
        "panic" => ("panic!", &mut inventory.panic),
        "unreachable" => ("unreachable!", &mut inventory.unreachable),
        "assert" => ("assert!", &mut inventory.assert),
        "assert_eq" => ("assert_eq!", &mut inventory.assert_eq),
        "assert_ne" => ("assert_ne!", &mut inventory.assert_ne),
        "debug_assert" => ("debug_assert!", &mut inventory.debug_assert),
        "debug_assert_eq" => ("debug_assert_eq!", &mut inventory.debug_assert_eq),
        "debug_assert_ne" => ("debug_assert_ne!", &mut inventory.debug_assert_ne),
        _ => return Ok(None),
    };
    *counter = counter
        .checked_add(1)
        .ok_or_else(|| format!("production `{name}!` inventory overflow"))?;
    Ok(Some(canonical_name))
}

fn checked_sum(name: &str, left: usize, right: usize) -> Result<usize, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("production `{name}!` inventory overflow"))
}
