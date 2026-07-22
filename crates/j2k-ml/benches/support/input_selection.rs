// SPDX-License-Identifier: MIT OR Apache-2.0

const INPUT_MODE_ENV: &str = "J2K_ML_BATCH_INPUT_MODE";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InputMode {
    Distinct,
    Repeated,
}

impl InputMode {
    pub(crate) fn from_env() -> Result<Self, String> {
        match std::env::var(INPUT_MODE_ENV) {
            Ok(value) => Self::parse(Some(&value)),
            Err(std::env::VarError::NotPresent) => Self::parse(None),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err(format!("{INPUT_MODE_ENV} must be valid UTF-8"))
            }
        }
    }

    pub(crate) fn parse(value: Option<&str>) -> Result<Self, String> {
        match value {
            None | Some("distinct") => Ok(Self::Distinct),
            Some("repeated") => Ok(Self::Repeated),
            Some(value) => Err(format!(
                "unsupported {INPUT_MODE_ENV}={value:?}; expected distinct or repeated"
            )),
        }
    }

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Distinct => "distinct",
            Self::Repeated => "repeated",
        }
    }
}
