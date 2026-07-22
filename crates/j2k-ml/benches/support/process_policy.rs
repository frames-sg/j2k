// SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) const PROCESS_MODE_ENV: &str = "J2K_ML_BATCH_PROCESS_MODE";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProcessMode {
    Criterion,
    Profile,
}

impl ProcessMode {
    pub(crate) fn from_env() -> Result<Self, String> {
        match std::env::var(PROCESS_MODE_ENV) {
            Ok(value) => match value.as_str() {
                "criterion" => Ok(Self::Criterion),
                "profile" => Ok(Self::Profile),
                _ => Err(format!(
                    "unsupported {PROCESS_MODE_ENV}={value:?}; expected criterion or profile"
                )),
            },
            Err(std::env::VarError::NotPresent) => Ok(Self::Criterion),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err(format!("{PROCESS_MODE_ENV} must be valid UTF-8"))
            }
        }
    }
}
