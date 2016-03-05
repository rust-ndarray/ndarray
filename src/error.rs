use std::fmt;
use std::error::Error;
use super::{
    Dimension,
};

/// An error related to array shape or layout.
#[derive(Clone)]
pub struct ShapeError {
    // we want to be able to change this representation later
    repr: ErrorKind,
}

impl ShapeError {
    /// Return the `ErrorKind` of this error.
    #[inline]
    pub fn kind(&self) -> ErrorKind {
        self.repr
    }
}

/// Error code for an error related to array shape or layout.
///
/// This enumeration is not exhaustive. The representation of the enum
/// is not guaranteed.
#[derive(Copy, Clone, Debug)]
pub enum ErrorKind {
    /// incompatible shape
    IncompatibleShape,
    /// incompatible memory layout
    IncompatibleLayout,
    /// the shape does not fit inside type limits
    RangeLimited,
    /// out of bounds indexing
    OutOfBounds,
    /// aliasing array elements
    Unsupported,
    #[doc(hidden)]
    __Incomplete,
}

#[inline(always)]
pub fn from_kind(k: ErrorKind) -> ShapeError {
    ShapeError {
        repr: k
    }
}

impl PartialEq for ErrorKind {
    #[inline(always)]
    fn eq(&self, rhs: &Self) -> bool {
        *self as u8 == *rhs as u8
    }
}

impl PartialEq for ShapeError {
    #[inline(always)]
    fn eq(&self, rhs: &Self) -> bool {
        self.repr == rhs.repr
    }
}

impl Error for ShapeError {
    fn description(&self) -> &str {
        match self.kind() {
            ErrorKind::IncompatibleShape => "incompatible shapes",
            ErrorKind::IncompatibleLayout => "incompatible memory layout",
            ErrorKind::RangeLimited => "the shape does not fit in type limits",
            ErrorKind::OutOfBounds => "out of bounds indexing",
            ErrorKind::Unsupported => "unsupported operation",
            ErrorKind::__Incomplete => "this error variant is not in use",
        }
    }
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.description().fmt(f)
    }
}

impl fmt::Debug for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ShapeError {:?}: {}", self.kind(), self.description())
    }
}

pub fn incompatible_shapes<D, E>(_a: &D, _b: &E) -> ShapeError
    where D: Dimension,
          E: Dimension
{
    from_kind(ErrorKind::IncompatibleShape)
}
