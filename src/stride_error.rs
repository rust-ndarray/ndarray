use std::fmt;
use std::error::Error;

/// An error to describe invalid stride states
#[derive(Clone, Debug, PartialEq)]
pub enum StrideError {
    /// stride leads to out of bounds indexing
    OutOfBoundsStride,
    /// stride leads to aliasing array elements
    AliasingStride,
    /// negative strides are unsafe in constructors
    NegativeStride,
}

impl Error for StrideError {
    fn description(&self) -> &str {
        match *self {
            StrideError::OutOfBoundsStride =>
                "stride leads to out of bounds indexing",
            StrideError::AliasingStride =>
                "stride leads to aliasing array elements",
            StrideError::NegativeStride =>
                "negative strides are unsafe in constructors",
        }
    }
}

impl fmt::Display for StrideError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.description().fmt(f)
    }
}
