//! Message handling system — port of `ojph_message.h/cpp`.
//!
//! The C++ library uses a global mutable message handler with three severity
//! levels.  This module provides a similar mechanism using a trait object
//! behind a [`std::sync::Mutex`].

use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Severity level
// ---------------------------------------------------------------------------

/// Message severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsgLevel {
    /// Informational message.
    Info = 0,
    /// Warning — non-fatal but potentially problematic.
    Warn = 1,
    /// Error — the operation cannot continue.
    Error = 2,
}

// ---------------------------------------------------------------------------
// Handler trait
// ---------------------------------------------------------------------------

/// Trait for receiving diagnostic messages from the codec.
///
/// Implement this to redirect messages to a custom logging framework.
pub trait MessageHandler: Send + Sync {
    /// Handle a diagnostic message.
    fn handle(&self, level: MsgLevel, code: u32, msg: &str);
}

/// Default handler that writes to stderr.
struct StderrHandler;

impl MessageHandler for StderrHandler {
    fn handle(&self, level: MsgLevel, code: u32, msg: &str) {
        let tag = match level {
            MsgLevel::Info => "INFO",
            MsgLevel::Warn => "WARN",
            MsgLevel::Error => "ERROR",
        };
        eprintln!("[openjph {}] (0x{:08x}) {}", tag, code, msg);
    }
}

// ---------------------------------------------------------------------------
// Global handler
// ---------------------------------------------------------------------------

static HANDLER: Mutex<Option<Box<dyn MessageHandler>>> = Mutex::new(None);

/// Dispatches a message to the currently installed handler.
pub fn dispatch_message(level: MsgLevel, code: u32, msg: &str) {
    let guard = HANDLER.lock().unwrap();
    match guard.as_ref() {
        Some(h) => h.handle(level, code, msg),
        None => {
            // Fallback: use the default stderr handler.
            StderrHandler.handle(level, code, msg);
        }
    }
}

/// Replaces the global message handler.
///
/// Pass `None` to restore the default stderr handler.
pub fn set_message_handler(handler: Option<Box<dyn MessageHandler>>) {
    let mut guard = HANDLER.lock().unwrap();
    *guard = handler;
}

// ---------------------------------------------------------------------------
// Convenience macros
// ---------------------------------------------------------------------------

/// Emit an informational message.
#[macro_export]
macro_rules! ojph_info {
    ($code:expr, $($arg:tt)*) => {
        $crate::message::dispatch_message(
            $crate::message::MsgLevel::Info,
            $code,
            &format!($($arg)*),
        )
    };
}

/// Emit a warning message.
#[macro_export]
macro_rules! ojph_warn {
    ($code:expr, $($arg:tt)*) => {
        $crate::message::dispatch_message(
            $crate::message::MsgLevel::Warn,
            $code,
            &format!($($arg)*),
        )
    };
}

/// Emit an error message and return an `OjphError::Codec`.
#[macro_export]
macro_rules! ojph_error {
    ($code:expr, $($arg:tt)*) => {{
        let msg = format!($($arg)*);
        $crate::message::dispatch_message(
            $crate::message::MsgLevel::Error,
            $code,
            &msg,
        );
        $crate::error::OjphError::Codec { code: $code, message: msg }
    }};
}
