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
    #[fail(display = "Incompatible shape.")]
    IncompatibleShape,
    #[fail(display = "Incompatible layout.")]
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

impl PartialEq for MyError {
    #[inline(always)]
    fn eq(&self, rhs: &Self) -> bool {
        self.kind() == rhs.kind()
    }
}

pub fn incompatible_shapes<D, E>(_a: &D, _b: &E) -> MyError
where
    D: Dimension,
    E: Dimension,
{
    MyError::from(MyErrorKind::IncompatibleShape)
}