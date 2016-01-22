use std::fmt;
use std::error::Error;

/// An error to describe invalid stride states
#[derive(Clone, Debug, PartialEq)]
pub enum StrideError {
    /// stride leads to out of bounds indexing
    OutOfBounds,
    /// stride leads to aliasing array elements
    Aliasing,
}

impl Error for StrideError {
    fn description(&self) -> &str {
        match *self {
            StrideError::OutOfBounds =>
                "stride leads to out of bounds indexing",
            StrideError::Aliasing =>
                "stride leads to aliasing array elements",
        }
    }
}

impl fmt::Display for StrideError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.description().fmt(f)
    }
}
