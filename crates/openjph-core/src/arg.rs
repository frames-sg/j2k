//! CLI argument interpreter — port of `ojph_arg.h`.
//!
//! A minimal argument parser that tracks which arguments have been consumed.
//! The full CLI crate uses `clap`; this module exists for library-level
//! argument handling and C++ API compatibility.

use crate::error::{OjphError, Result};

/// A simple command-line argument interpreter.
///
/// Arguments are stored alongside a "consumed" flag so that callers can
/// verify every argument has been handled.
pub struct CliInterpreter {
    args: Vec<String>,
    available: Vec<bool>,
}

impl CliInterpreter {
    /// Creates a new interpreter from a list of argument strings.
    pub fn init(args: Vec<String>) -> Self {
        let len = args.len();
        Self {
            args,
            available: vec![true; len],
        }
    }

    /// Searches for `name` (e.g. `"-o"`) and returns its index, or `None`.
    pub fn find_argument(&self, name: &str) -> Option<usize> {
        self.args
            .iter()
            .enumerate()
            .find(|(i, a)| *a == name && self.available[*i])
            .map(|(i, _)| i)
    }

    /// Returns the value immediately following the argument at `index`,
    /// marking both as consumed.
    pub fn get_next_value(&mut self, index: usize) -> Result<&str> {
        let next = index + 1;
        if next >= self.args.len() || !self.available[next] {
            return Err(OjphError::InvalidParam(format!(
                "expected value after argument '{}'",
                self.args.get(index).unwrap_or(&String::new()),
            )));
        }
        self.available[index] = false;
        self.available[next] = false;
        Ok(&self.args[next])
    }

    /// Marks the argument at `index` as consumed.
    pub fn release_argument(&mut self, index: usize) {
        if index < self.available.len() {
            self.available[index] = false;
        }
    }

    /// Returns `true` when every argument has been consumed.
    pub fn is_exhausted(&self) -> bool {
        self.available.iter().all(|&a| !a)
    }

    /// Returns the first un-consumed argument, if any.
    pub fn first_unconsumed(&self) -> Option<&str> {
        self.args
            .iter()
            .enumerate()
            .find(|(i, _)| self.available[*i])
            .map(|(_, a)| a.as_str())
    }

    // ------------------------------------------------------------------
    // Type-safe re-interpretation helpers
    // ------------------------------------------------------------------

    /// Parses the value at `index + 1` as `i32`.
    pub fn reinterpret_i32(&mut self, index: usize) -> Result<i32> {
        let s = self.get_next_value(index)?;
        s.parse::<i32>()
            .map_err(|_| OjphError::InvalidParam(format!("cannot parse '{}' as i32", s)))
    }

    /// Parses the value at `index + 1` as `u32`.
    pub fn reinterpret_u32(&mut self, index: usize) -> Result<u32> {
        let s = self.get_next_value(index)?;
        s.parse::<u32>()
            .map_err(|_| OjphError::InvalidParam(format!("cannot parse '{}' as u32", s)))
    }

    /// Parses the value at `index + 1` as `f32`.
    pub fn reinterpret_f32(&mut self, index: usize) -> Result<f32> {
        let s = self.get_next_value(index)?;
        s.parse::<f32>()
            .map_err(|_| OjphError::InvalidParam(format!("cannot parse '{}' as f32", s)))
    }

    /// Parses the value at `index + 1` as `bool` (accepts `true`/`false`,
    /// `1`/`0`, `yes`/`no`).
    pub fn reinterpret_bool(&mut self, index: usize) -> Result<bool> {
        let s = self.get_next_value(index)?;
        match s.to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" => Ok(true),
            "false" | "0" | "no" => Ok(false),
            _ => Err(OjphError::InvalidParam(format!(
                "cannot parse '{}' as bool",
                s
            ))),
        }
    }

    /// Returns the value at `index + 1` as a `String`.
    pub fn reinterpret_string(&mut self, index: usize) -> Result<String> {
        self.get_next_value(index).map(|s| s.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_and_consume() {
        let args = vec![
            "-i".to_string(),
            "input.j2c".to_string(),
            "-o".to_string(),
            "output.ppm".to_string(),
        ];
        let mut cli = CliInterpreter::init(args);

        let idx = cli.find_argument("-i").unwrap();
        let val = cli.reinterpret_string(idx).unwrap();
        assert_eq!(val, "input.j2c");

        let idx = cli.find_argument("-o").unwrap();
        let val = cli.reinterpret_string(idx).unwrap();
        assert_eq!(val, "output.ppm");

        assert!(cli.is_exhausted());
    }
}
