// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::env;

use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::{Attribute, Expr, Lit, Meta, Token};

#[derive(Clone, Debug)]
pub(super) struct CoverageCfgContext {
    target_os: String,
    target_arch: String,
    target_family: String,
    target_pointer_width: String,
    target_endian: String,
    enabled_features: BTreeSet<String>,
    custom_flags: Option<BTreeMap<String, bool>>,
}

impl CoverageCfgContext {
    pub(super) fn for_current_target(
        enabled_features: BTreeSet<String>,
        custom_flags: Option<BTreeMap<String, bool>>,
    ) -> Self {
        Self {
            target_os: env::consts::OS.to_string(),
            target_arch: env::consts::ARCH.to_string(),
            target_family: if cfg!(windows) { "windows" } else { "unix" }.to_string(),
            target_pointer_width: usize::BITS.to_string(),
            target_endian: if cfg!(target_endian = "little") {
                "little"
            } else {
                "big"
            }
            .to_string(),
            enabled_features,
            custom_flags,
        }
    }

    #[cfg(test)]
    pub(super) fn synthetic(custom_flags: impl IntoIterator<Item = (&'static str, bool)>) -> Self {
        Self::for_current_target(
            BTreeSet::new(),
            Some(
                custom_flags
                    .into_iter()
                    .map(|(name, active)| (name.to_string(), active))
                    .collect(),
            ),
        )
    }

    fn path_truth(&self, path: &syn::Path) -> Result<SymbolicTruth, String> {
        let Some(identifier) = path.get_ident() else {
            return Err("compound cfg paths are unsupported by coverage analysis".to_string());
        };
        let name = identifier.to_string();
        match name.as_str() {
            "test" => Ok(SymbolicTruth::False),
            "unix" => Ok(SymbolicTruth::from_bool(self.target_family == "unix")),
            "windows" => Ok(SymbolicTruth::from_bool(self.target_family == "windows")),
            // Custom build-script cfg values are captured when Cargo emitted
            // check-cfg evidence. Unseen cfgs stay unknown so both a predicate
            // and its negation remain conservatively active.
            _ => Ok(self
                .custom_flags
                .as_ref()
                .and_then(|flags| flags.get(&name))
                .copied()
                .map_or(SymbolicTruth::Unknown, SymbolicTruth::from_bool)),
        }
    }

    fn name_value_truth(&self, value: &syn::MetaNameValue) -> Result<SymbolicTruth, String> {
        let Expr::Lit(expression) = &value.value else {
            return Err("cfg name-value predicate must use a literal".to_string());
        };
        let Lit::Str(literal) = &expression.lit else {
            return Err("cfg name-value predicate must use a string literal".to_string());
        };
        let expected = literal.value();
        if value.path.is_ident("target_os") {
            return Ok(SymbolicTruth::from_bool(self.target_os == expected));
        }
        if value.path.is_ident("target_arch") {
            return Ok(SymbolicTruth::from_bool(self.target_arch == expected));
        }
        if value.path.is_ident("target_family") {
            return Ok(SymbolicTruth::from_bool(self.target_family == expected));
        }
        if value.path.is_ident("target_pointer_width") {
            return Ok(SymbolicTruth::from_bool(
                self.target_pointer_width == expected,
            ));
        }
        if value.path.is_ident("target_endian") {
            return Ok(SymbolicTruth::from_bool(self.target_endian == expected));
        }
        if value.path.is_ident("feature") {
            return Ok(SymbolicTruth::from_bool(
                self.enabled_features.contains(&expected),
            ));
        }
        if value.path.get_ident().is_none() {
            return Err("compound cfg name-value paths are unsupported".to_string());
        }
        // `target_feature` and custom name-value cfgs are not reconstructable
        // from Cargo metadata alone. Unknown is fail-closed in either polarity.
        Ok(SymbolicTruth::Unknown)
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct AttributeState {
    pub(super) implies_test: bool,
    pub(super) active: bool,
}

pub(super) fn attributes_state(
    attrs: &[Attribute],
    context: &CoverageCfgContext,
) -> Result<AttributeState, String> {
    let mut state = AttributeState {
        implies_test: attrs.iter().any(is_test_or_bench),
        active: true,
    };
    for attribute in attrs {
        if attribute.path().is_ident("cfg") {
            let predicate = cfg_predicate(attribute)?;
            state.implies_test |= meta_implies_test(&predicate)?;
            state.active &= evaluate_meta(&predicate, context)?.conservatively_active();
        } else if attribute.path().is_ident("cfg_attr") {
            reject_structural_cfg_attr(attribute)?;
        }
    }
    Ok(state)
}

fn is_test_or_bench(attribute: &Attribute) -> bool {
    attribute.path().is_ident("test") || attribute.path().is_ident("bench")
}

fn cfg_predicate(attribute: &Attribute) -> Result<Meta, String> {
    let Meta::List(list) = &attribute.meta else {
        return Err("cfg attribute is not a predicate list".to_string());
    };
    let mut predicates = Punctuated::<Meta, Token![,]>::parse_terminated
        .parse2(list.tokens.clone())
        .map_err(|error| format!("malformed cfg predicate: {error}"))?
        .into_iter();
    let predicate = predicates
        .next()
        .ok_or_else(|| "cfg attribute must contain exactly one predicate".to_string())?;
    if predicates.next().is_some() {
        return Err("cfg attribute must contain exactly one predicate".to_string());
    }
    Ok(predicate)
}

fn reject_structural_cfg_attr(attribute: &Attribute) -> Result<(), String> {
    let Meta::List(list) = &attribute.meta else {
        return Err("cfg_attr attribute is not a predicate list".to_string());
    };
    let mut entries = nested_meta(list)?.into_iter();
    entries
        .next()
        .ok_or_else(|| "cfg_attr must contain a condition and attribute".to_string())?;
    let mut saw_attribute = false;
    for applied in entries {
        saw_attribute = true;
        if applied.path().is_ident("cfg")
            || applied.path().is_ident("cfg_attr")
            || applied.path().is_ident("path")
            || applied.path().is_ident("test")
            || applied.path().is_ident("bench")
        {
            return Err(
                "coverage source analysis does not support structural cfg_attr".to_string(),
            );
        }
    }
    if !saw_attribute {
        return Err("cfg_attr must contain a condition and attribute".to_string());
    }
    Ok(())
}

fn nested_meta(list: &syn::MetaList) -> Result<Vec<Meta>, String> {
    Punctuated::<Meta, Token![,]>::parse_terminated
        .parse2(list.tokens.clone())
        .map(|items| items.into_iter().collect())
        .map_err(|error| format!("malformed cfg list: {error}"))
}

fn meta_implies_test(meta: &Meta) -> Result<bool, String> {
    Ok(test_false_truth(meta)? == SymbolicTruth::False)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SymbolicTruth {
    True,
    False,
    Unknown,
}

impl SymbolicTruth {
    const fn from_bool(value: bool) -> Self {
        if value {
            Self::True
        } else {
            Self::False
        }
    }

    const fn conservatively_active(self) -> bool {
        !matches!(self, Self::False)
    }

    const fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::False, _) | (_, Self::False) => Self::False,
            (Self::True, Self::True) => Self::True,
            _ => Self::Unknown,
        }
    }

    const fn or(self, other: Self) -> Self {
        match (self, other) {
            (Self::True, _) | (_, Self::True) => Self::True,
            (Self::False, Self::False) => Self::False,
            _ => Self::Unknown,
        }
    }

    const fn not(self) -> Self {
        match self {
            Self::True => Self::False,
            Self::False => Self::True,
            Self::Unknown => Self::Unknown,
        }
    }
}

fn test_false_truth(meta: &Meta) -> Result<SymbolicTruth, String> {
    match meta {
        Meta::Path(path) if path.is_ident("test") => Ok(SymbolicTruth::False),
        Meta::List(list) if list.path.is_ident("all") => nested_meta(list)?
            .iter()
            .try_fold(SymbolicTruth::True, |state, item| {
                test_false_truth(item).map(|value| state.and(value))
            }),
        Meta::List(list) if list.path.is_ident("any") => nested_meta(list)?
            .iter()
            .try_fold(SymbolicTruth::False, |state, item| {
                test_false_truth(item).map(|value| state.or(value))
            }),
        Meta::List(list) if list.path.is_ident("not") => {
            let items = nested_meta(list)?;
            let [item] = items.as_slice() else {
                return Err("cfg(not(...)) must contain exactly one predicate".to_string());
            };
            Ok(test_false_truth(item)?.not())
        }
        Meta::Path(_) | Meta::NameValue(_) | Meta::List(_) => Ok(SymbolicTruth::Unknown),
    }
}

fn evaluate_meta(meta: &Meta, context: &CoverageCfgContext) -> Result<SymbolicTruth, String> {
    match meta {
        Meta::Path(path) => context.path_truth(path),
        Meta::NameValue(value) => context.name_value_truth(value),
        Meta::List(list) if list.path.is_ident("all") => {
            nested_meta(list)?
                .iter()
                .try_fold(SymbolicTruth::True, |state, item| {
                    let active = evaluate_meta(item, context)?;
                    Ok(state.and(active))
                })
        }
        Meta::List(list) if list.path.is_ident("any") => {
            nested_meta(list)?
                .iter()
                .try_fold(SymbolicTruth::False, |state, item| {
                    let active = evaluate_meta(item, context)?;
                    Ok(state.or(active))
                })
        }
        Meta::List(list) if list.path.is_ident("not") => {
            let items = nested_meta(list)?;
            let [item] = items.as_slice() else {
                return Err("cfg(not(...)) must contain exactly one predicate".to_string());
            };
            Ok(evaluate_meta(item, context)?.not())
        }
        Meta::List(list) => Err(format!(
            "unknown cfg predicate function `{}`",
            list.path
                .get_ident()
                .map_or_else(|| "<compound>".to_string(), ToString::to_string)
        )),
    }
}
