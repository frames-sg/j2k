// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared command-line policy for fail-closed GPU validation modes.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ValidationMode {
    Quick,
    Full,
}

impl ValidationMode {
    pub(crate) fn parse(mut args: impl Iterator<Item = String>) -> Result<Self, String> {
        let Some(argument) = args.next() else {
            return Ok(Self::Full);
        };
        if argument != "--mode" {
            return Err(format!(
                "unknown GPU validation argument `{argument}`; expected --mode quick|full"
            ));
        }
        let value = args
            .next()
            .ok_or_else(|| "--mode requires quick or full".to_string())?;
        if let Some(extra) = args.next() {
            return Err(format!(
                "unexpected GPU validation argument `{extra}`; expected only --mode quick|full"
            ));
        }
        match value.as_str() {
            "quick" => Ok(Self::Quick),
            "full" => Ok(Self::Full),
            _ => Err(format!(
                "unknown GPU validation mode `{value}`; expected quick or full"
            )),
        }
    }

    pub(crate) const fn cargo_profile_args(self) -> &'static [&'static str] {
        match self {
            Self::Quick => &["--profile", "gpu-quick"],
            Self::Full => &["--release"],
        }
    }
}
