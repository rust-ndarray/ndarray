// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use super::Dimension;
use std::error::Error;
use std::fmt;

use failure::{Context, Fail, Backtrace};

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

    /// Create a new `ShapeError`
    pub fn from_kind(error: ErrorKind) -> Self {
        from_kind(error)
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
    /// overflow when computing offset, length, etc.
    Overflow,
    #[doc(hidden)]
    __Incomplete,
}

#[inline(always)]
pub fn from_kind(k: ErrorKind) -> ShapeError {
    ShapeError { repr: k }
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
            ErrorKind::Overflow => "arithmetic overflow",
            ErrorKind::__Incomplete => "this error variant is not in use",
        }
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

pub fn incompatible_shapes<D, E>(_a: &D, _b: &E) -> ShapeError
where
    D: Dimension,
    E: Dimension,
{
    from_kind(ErrorKind::IncompatibleShape)
}




/// An error related to array shape or layout.
#[derive(Debug)]
pub struct MyError {
    inner: Context<MyErrorKind>,
}

/// Error code for an error related to array shape or layout.
///
/// This enumeration is not exhaustive. The representation of the enum
/// is not guaranteed.
#[derive(Clone, PartialEq, Debug, Fail)]
pub enum MyErrorKind {
    #[fail(display = "Incompatible shape. Method '{}' returned message error: '{:?}'", method_name, message)]
    IncompatibleShape{
        method_name: &'static str,
        message: String
    },
    #[fail(display = "Incompatible memory layout.")]
    IncompatibleLayout,
    #[fail(display = "The shape does not fit inside type limits.")]
    RangeLimited,
    #[fail(display = "Out of bounds indexing.")]
    OutOfBounds,
    #[fail(display = "Aliasing array elements.")]
    Unsupported,
    #[fail(display = "Overflow when computing offset, length, etc.")]
    Overflow,
    #[fail(display = "Incomplete")]
    #[doc(hidden)]
    __Incomplete,
}

impl Fail for MyError {
    fn cause(&self) -> Option<&Fail> {
        self.inner.cause()
    }

    fn backtrace(&self) -> Option<&Backtrace> {
        self.inner.backtrace()
    }
}

impl fmt::Display for MyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.inner, f)
    }
}

impl MyError {
    pub fn kind(&self) -> &MyErrorKind {
        self.inner.get_context()
    }
}

impl From<MyErrorKind> for MyError {
    fn from(kind: MyErrorKind) -> MyError {
        MyError { inner: Context::new(kind) }
    }
}

impl From<Context<MyErrorKind>> for MyError {
    fn from(inner: Context<MyErrorKind>) -> MyError {
        MyError { inner }
    }
}