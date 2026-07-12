// SPDX-License-Identifier: MIT OR Apache-2.0

use syn::{Attribute, Expr, FnArg, ForeignItem, GenericParam, ImplItem, Item, Pat, TraitItem};

pub(super) fn item(item: &Item) -> &[Attribute] {
    match item {
        Item::Const(value) => &value.attrs,
        Item::Enum(value) => &value.attrs,
        Item::ExternCrate(value) => &value.attrs,
        Item::Fn(value) => &value.attrs,
        Item::ForeignMod(value) => &value.attrs,
        Item::Impl(value) => &value.attrs,
        Item::Macro(value) => &value.attrs,
        Item::Mod(value) => &value.attrs,
        Item::Static(value) => &value.attrs,
        Item::Struct(value) => &value.attrs,
        Item::Trait(value) => &value.attrs,
        Item::TraitAlias(value) => &value.attrs,
        Item::Type(value) => &value.attrs,
        Item::Union(value) => &value.attrs,
        Item::Use(value) => &value.attrs,
        _ => &[],
    }
}

pub(super) fn impl_item(item: &ImplItem) -> &[Attribute] {
    match item {
        ImplItem::Const(value) => &value.attrs,
        ImplItem::Fn(value) => &value.attrs,
        ImplItem::Macro(value) => &value.attrs,
        ImplItem::Type(value) => &value.attrs,
        _ => &[],
    }
}

pub(super) fn trait_item(item: &TraitItem) -> &[Attribute] {
    match item {
        TraitItem::Const(value) => &value.attrs,
        TraitItem::Fn(value) => &value.attrs,
        TraitItem::Macro(value) => &value.attrs,
        TraitItem::Type(value) => &value.attrs,
        _ => &[],
    }
}

pub(super) fn expression(expr: &Expr) -> Result<&[Attribute], String> {
    match expr {
        Expr::Array(value) => Ok(&value.attrs),
        Expr::Assign(value) => Ok(&value.attrs),
        Expr::Async(value) => Ok(&value.attrs),
        Expr::Await(value) => Ok(&value.attrs),
        Expr::Binary(value) => Ok(&value.attrs),
        Expr::Block(value) => Ok(&value.attrs),
        Expr::Break(value) => Ok(&value.attrs),
        Expr::Call(value) => Ok(&value.attrs),
        Expr::Cast(value) => Ok(&value.attrs),
        Expr::Closure(value) => Ok(&value.attrs),
        Expr::Const(value) => Ok(&value.attrs),
        Expr::Continue(value) => Ok(&value.attrs),
        Expr::Field(value) => Ok(&value.attrs),
        Expr::ForLoop(value) => Ok(&value.attrs),
        Expr::Group(value) => Ok(&value.attrs),
        Expr::If(value) => Ok(&value.attrs),
        Expr::Index(value) => Ok(&value.attrs),
        Expr::Infer(value) => Ok(&value.attrs),
        Expr::Let(value) => Ok(&value.attrs),
        Expr::Lit(value) => Ok(&value.attrs),
        Expr::Loop(value) => Ok(&value.attrs),
        Expr::Macro(value) => Ok(&value.attrs),
        Expr::Match(value) => Ok(&value.attrs),
        Expr::MethodCall(value) => Ok(&value.attrs),
        Expr::Paren(value) => Ok(&value.attrs),
        Expr::Path(value) => Ok(&value.attrs),
        Expr::Range(value) => Ok(&value.attrs),
        Expr::RawAddr(value) => Ok(&value.attrs),
        Expr::Reference(value) => Ok(&value.attrs),
        Expr::Repeat(value) => Ok(&value.attrs),
        Expr::Return(value) => Ok(&value.attrs),
        Expr::Struct(value) => Ok(&value.attrs),
        Expr::Try(value) => Ok(&value.attrs),
        Expr::TryBlock(value) => Ok(&value.attrs),
        Expr::Tuple(value) => Ok(&value.attrs),
        Expr::Unary(value) => Ok(&value.attrs),
        Expr::Unsafe(value) => Ok(&value.attrs),
        Expr::While(value) => Ok(&value.attrs),
        Expr::Yield(value) => Ok(&value.attrs),
        Expr::Verbatim(_) => Err("unclassified verbatim expression".to_string()),
        _ => Err("unclassified non-exhaustive expression variant".to_string()),
    }
}

pub(super) fn foreign_item(item: &ForeignItem) -> Result<&[Attribute], String> {
    match item {
        ForeignItem::Fn(value) => Ok(&value.attrs),
        ForeignItem::Macro(value) => Ok(&value.attrs),
        ForeignItem::Static(value) => Ok(&value.attrs),
        ForeignItem::Type(value) => Ok(&value.attrs),
        ForeignItem::Verbatim(_) => Err("unclassified verbatim foreign item".to_string()),
        _ => Err("unclassified non-exhaustive foreign item variant".to_string()),
    }
}

pub(super) fn function_argument(argument: &FnArg) -> &[Attribute] {
    match argument {
        FnArg::Receiver(receiver) => &receiver.attrs,
        FnArg::Typed(argument) => &argument.attrs,
    }
}

pub(super) fn generic_parameter(parameter: &GenericParam) -> &[Attribute] {
    match parameter {
        GenericParam::Lifetime(parameter) => &parameter.attrs,
        GenericParam::Type(parameter) => &parameter.attrs,
        GenericParam::Const(parameter) => &parameter.attrs,
    }
}

pub(super) fn pattern(pattern: &Pat) -> Result<&[Attribute], String> {
    match pattern {
        Pat::Const(value) => Ok(&value.attrs),
        Pat::Ident(value) => Ok(&value.attrs),
        Pat::Lit(value) => Ok(&value.attrs),
        Pat::Macro(value) => Ok(&value.attrs),
        Pat::Or(value) => Ok(&value.attrs),
        Pat::Paren(value) => Ok(&value.attrs),
        Pat::Path(value) => Ok(&value.attrs),
        Pat::Range(value) => Ok(&value.attrs),
        Pat::Reference(value) => Ok(&value.attrs),
        Pat::Rest(value) => Ok(&value.attrs),
        Pat::Slice(value) => Ok(&value.attrs),
        Pat::Struct(value) => Ok(&value.attrs),
        Pat::Tuple(value) => Ok(&value.attrs),
        Pat::TupleStruct(value) => Ok(&value.attrs),
        Pat::Type(value) => Ok(&value.attrs),
        Pat::Wild(value) => Ok(&value.attrs),
        Pat::Verbatim(_) => Err("unclassified verbatim pattern".to_string()),
        _ => Err("unclassified non-exhaustive pattern variant".to_string()),
    }
}
