use std::fmt;
use std::error::Error;
use super::{
    Dimension,
};

/// An error related to array shape or layout.
#[derive(Clone, Debug)]
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
#[repr(u64)]
pub enum ErrorKind {
    /// incompatible shapes
    IncompatibleShapes,
    /// incompatible layout: not contiguous
    IncompatibleLayout,
    /// dimension too large (shape)
    DimensionTooLarge,
    /// stride leads to out of bounds indexing
    OutOfBounds,
    /// stride leads to aliasing array elements
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
        *self as u64 == *rhs as u64
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
            ErrorKind::IncompatibleShapes => "incompatible shapes",
            ErrorKind::IncompatibleLayout => "incompatible layout (not contiguous)",
            ErrorKind::DimensionTooLarge => "dimension too large",
            ErrorKind::OutOfBounds => "stride leads to out of bounds indexing",
            ErrorKind::Unsupported => "stride leads to aliasing array elements",
            ErrorKind::__Incomplete => "this error variant is not in use",
        }
    }
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.description().fmt(f)
    }
}

pub fn incompatible_shapes<D, E>(_a: &D, _b: &E) -> ShapeError
    where D: Dimension,
          E: Dimension
{
    from_kind(ErrorKind::IncompatibleShapes)
}
