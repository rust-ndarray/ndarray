// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
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
    msg: String
}

impl ShapeError {
    /// Return the `ErrorKind` of this error.
    #[inline]
    pub fn kind(&self) -> ErrorKind {
        self.repr
    }

    /// Create a new `ShapeError`
    pub fn from_kind(error: ErrorKind, msg: String) -> Self {
        from_kind(error, msg)
    }
}

/// Error code for an error related to array shape or layout.
///
/// This enumeration is not exhaustive. The representation of the enum
/// is not guaranteed.
#[derive(Copy, Clone, Debug)]
pub enum ErrorKind {
    /// incompatible shape
    IncompatibleShape = 1,
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
pub fn from_kind(k: ErrorKind, msg: String) -> ShapeError {
    let prefix = match k {
        ErrorKind::IncompatibleShape => { "incompatible shapes" },
        ErrorKind::IncompatibleLayout => "incompatible memory layout",
        ErrorKind::RangeLimited => "the shape does not fit in type limits",
        ErrorKind::OutOfBounds => "out of bounds indexing",
        ErrorKind::Unsupported => "unsupported operation",
        ErrorKind::__Incomplete => "this error variant is not in use",
    };

    let mut prefixed_msg = String::with_capacity(prefix.len() + ": ".len() + msg.len());
    prefixed_msg.push_str(prefix);
    prefixed_msg.push_str(": ");
    prefixed_msg.push_str(&msg);

    ShapeError {
        repr: k,
        msg: prefixed_msg
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
        self.repr as u8 == rhs.repr as u8
            && self.msg == rhs.msg
    }
}

impl Error for ShapeError {
    fn description(&self) -> &str {
        &self.msg
    }
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ShapeError/{:?}: {}", self.kind(), self.description())
    }
}

impl fmt::Debug for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ShapeError/{:?}: {}", self.kind(), self.description())
    }
}

pub fn incompatible_shapes<D, E>(a: &D, b: &E) -> ShapeError
    where D: Dimension,
          E: Dimension
{
    from_kind(ErrorKind::IncompatibleShape,format!("{:?} != {:?}", a, b))
}
