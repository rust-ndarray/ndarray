// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use super::Dimension;
use std::fmt;

use failure::{Context, Fail, Backtrace};

/// An error related to array shape or layout.
#[derive(Debug)]
pub struct ShapeError {
    inner: Context<ShapeErrorKind>,
}

/// Error code for an error related to array shape or layout.
///
/// This enumeration is not exhaustive. The representation of the enum
/// is not guaranteed.
#[derive(Clone, PartialEq, Debug, Fail)]
pub enum ShapeErrorKind {
    #[fail(display = "Incompatible shape.")]
    IncompatibleShape,
    #[fail(display = "Incompatible layout. {}", message)]
    IncompatibleLayout {
        message: String
    },
    #[fail(display = "Out of bounds indexing. {}", message)]
    OutOfBounds {
        message: String
    },
    #[fail(display = "Aliasing array elements. {}", message)]
    Unsupported {
        message: String
    },
    #[fail(display = "Overflow dimensions. {}", message)]
    Overflow {
        message: String
    },
    #[fail(display = "The shape does not fit inside type limits.")]
    RangeLimited,
    #[fail(display = "Incomplete")]
    #[doc(hidden)]
    __Incomplete,
}

impl Fail for ShapeError {
    fn cause(&self) -> Option<&Fail> {
        self.inner.cause()
    }

    fn backtrace(&self) -> Option<&Backtrace> {
        self.inner.backtrace()
    }
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.inner, f)
    }
}

impl ShapeError {
    pub fn kind(&self) -> &ShapeErrorKind {
        self.inner.get_context()
    }
}

impl From<ShapeErrorKind> for ShapeError {
    fn from(kind: ShapeErrorKind) -> ShapeError {
        ShapeError { inner: Context::new(kind) }
    }
}

impl From<Context<ShapeErrorKind>> for ShapeError {
    fn from(inner: Context<ShapeErrorKind>) -> ShapeError {
        ShapeError { inner }
    }
}

impl PartialEq for ShapeError {
    #[inline(always)]
    fn eq(&self, rhs: &Self) -> bool {
        self.kind() == rhs.kind()
    }
}