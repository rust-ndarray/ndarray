// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![allow(clippy::identity_op)]
use super::Dimension;
use crate::itertools::enumerate;

#[cfg(feature = "std")]
use std::error::Error;
use std::fmt;
use std::mem::size_of;

/// An error related to array shape or layout.
///
/// The shape error encodes and shows expected/actual indices and shapes in some cases, which
/// is visible in the Display/Debug representation. Since this is done without allocation, it is
/// space-limited and bigger indices and shapes may not be representable.
#[derive(Clone)]
pub struct ShapeError {
    /// Error category
    repr: ErrorKind,
    /// Additional info
    info: InfoType,
}

impl ShapeError {
    /// Return the `ErrorKind` of this error.
    #[inline]
    pub fn kind(&self) -> ErrorKind {
        self.repr
    }

    /// Create a new `ShapeError` from the given kind
    pub fn from_kind(error: ErrorKind) -> Self {
        Self::from_kind_info(error, info_default())
    }

    fn from_kind_info(repr: ErrorKind, info: InfoType) -> Self {
        ShapeError { repr, info }
    }

    pub(crate) fn invalid_axis(expected: usize, actual: usize) -> Self {
        // TODO: OutOfBounds for compatibility reasons, should be more specific
        Self::from_kind_info(ErrorKind::OutOfBounds, encode_indices(expected, actual))
    }

    pub(crate) fn shape_length_exceeds_data_length(expected: usize, actual: usize) -> Self {
        // TODO: OutOfBounds for compatibility reasons, should be more specific
        Self::from_kind_info(ErrorKind::OutOfBounds, encode_indices(expected, actual))
    }

    pub(crate) fn incompatible_layout(expected: ExpectedLayout) -> Self {
        Self::from_kind_info(ErrorKind::IncompatibleLayout, encode_indices(expected as usize, 0))
    }

    pub(crate) fn incompatible_shapes<D, E>(expected: &D, actual: &E) -> ShapeError
    where
        D: Dimension,
        E: Dimension,
    {
        Self::from_kind_info(ErrorKind::IncompatibleShape, encode_shapes(expected, actual))
    }

    #[cfg(test)]
    fn info_expected_index(&self) -> Option<usize> {
        let (exp, _) = decode_indices(self.info);
        exp
    }

    #[cfg(test)]
    fn info_actual_index(&self) -> Option<usize> {
        let (_, actual) = decode_indices(self.info);
        actual
    }

    #[cfg(test)]
    fn decode_shapes(&self) -> (Option<DecodedShape>, Option<DecodedShape>) {
        decode_shapes(self.info)
    }
}

/// Error code for an error related to array shape or layout.
///
/// This enumeration is not exhaustive. The representation of the enum
/// is not guaranteed.
#[non_exhaustive]
#[derive(Copy, Clone, Debug)]
pub enum ErrorKind {
    /// incompatible shape
    // encodes info: expected and actual shape
    IncompatibleShape = 1,
    /// incompatible memory layout
    // encodes info: expected layout
    IncompatibleLayout,
    /// the shape does not fit inside type limits
    RangeLimited,
    /// out of bounds indexing
    // encodes info: expected and actual index
    OutOfBounds,
    /// aliasing array elements
    Unsupported,
    /// overflow when computing offset, length, etc.
    Overflow,
}

#[inline(always)]
pub fn from_kind(error: ErrorKind) -> ShapeError {
    ShapeError::from_kind(error)
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

#[cfg(feature = "std")]
impl Error for ShapeError {}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let description = match self.kind() {
            ErrorKind::IncompatibleShape => "incompatible shapes",
            ErrorKind::IncompatibleLayout => "incompatible memory layout",
            ErrorKind::RangeLimited => "the shape does not fit in type limits",
            ErrorKind::OutOfBounds => "out of bounds indexing",
            ErrorKind::Unsupported => "unsupported operation",
            ErrorKind::Overflow => "arithmetic overflow",
        };
        write!(f, "ShapeError/{:?}: {}{}", self.kind(), description, ExtendedInfo(&self))
    }
}

impl fmt::Debug for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

pub(crate) enum ExpectedLayout {
    ContiguousCF = 1,
    Unused,
}

impl From<Option<usize>> for ExpectedLayout {
    #[inline]
    fn from(x: Option<usize>) -> Self {
        match x {
            Some(1) => ExpectedLayout::ContiguousCF,
            _ => ExpectedLayout::Unused,
        }
    }
}

pub(crate) fn incompatible_shapes<D, E>(_a: &D, _b: &E) -> ShapeError
where
    D: Dimension,
    E: Dimension,
{
    ShapeError::incompatible_shapes(_a, _b)
}

/// The InfoType encodes extra information per error kind, for example expected/actual axis for a
/// given error site, or expected layout for a layout error.
///
/// It uses a custom and fixed-width (very limited) encoding; in some cases it skips filling in
/// information because it doesn't fit.
///
/// Two bits in the first byte are reserved for EncodedInformationType, the rest is used
/// for situation-specific encoding of extra info.
/// If the first byte is zero, it shows that there is no encoded info.
type InfoType = [u8; INFO_TYPE_LEN];

const INFO_TYPE_LEN: usize = 15;
const INFO_BYTES: usize = INFO_TYPE_LEN - 1;

const fn info_default() -> InfoType { [0; INFO_TYPE_LEN] }

#[repr(u8)]
// 2 bits
enum EncodedInformationType {
    Nothing = 0,
    Expected = 0b1,
    Actual = 0b10,
}

const IXBYTES: usize = INFO_BYTES / 2;

fn encode_index(x: usize) -> Option<[u8; IXBYTES]> {
    let bits = size_of::<usize>() * 8;
    let used_bits = bits - x.leading_zeros() as usize;
    if used_bits > IXBYTES * 8 {
        None
    } else {
        let bytes = x.to_le_bytes();
        let mut result = [0; IXBYTES];
        let len = bytes.len().min(result.len());
        result[..len].copy_from_slice(&bytes[..len]);
        Some(result)
    }
}

fn decode_index(x: &[u8]) -> usize {
    let mut bytes = 0usize.to_le_bytes();
    let len = x.len().min(bytes.len());
    bytes[..len].copy_from_slice(&x[..len]);
    usize::from_le_bytes(bytes)
}

fn encode_indices(expected: usize, actual: usize) -> InfoType {
    let eexp = encode_index(expected);
    let eact = encode_index(actual);
    let mut info = info_default();
    let mut info_type = EncodedInformationType::Nothing as u8;
    if eexp.is_some() {
        info_type |= EncodedInformationType::Expected as u8;
    }
    let (ebytes, abytes) = info[1..].split_at_mut(IXBYTES);
    if let Some(exp) = eexp {
        ebytes.copy_from_slice(&exp);
    }
    if eact.is_some() {
        info_type |= EncodedInformationType::Actual as u8;
    }
    if let Some(act) = eact {
        abytes.copy_from_slice(&act);
    }
    info[0] = info_type;
    info
}

fn decode_indices(info: InfoType) -> (Option<usize>, Option<usize>) {
    let (ebytes, abytes) = info[1..].split_at(IXBYTES);
    (
        if info[0] & (EncodedInformationType::Expected as u8) != 0 {
            Some(decode_index(ebytes))
        } else { None },
        if info[0] & (EncodedInformationType::Actual as u8) != 0 {
            Some(decode_index(abytes))
        } else { None },
    )

}

fn encode_shapes<D, E>(expected: &D, actual: &E) -> InfoType
where
    D: Dimension,
    E: Dimension,
{
    encode_shapes_impl(expected.slice(), actual.slice())
}

// Shape encoding
//
// 15 bytes to use
// 1 byte:
//   1 bit expected shape is two-byte encoded yes/no (a)
//   3 bits expected shape len (len 0..=7) (b)
//   1 bit actual shape is two-byte encoded yes/no (c)
//   3 bits actual shape len (len 0..=7) (d)
// 14 bytes encoding of expected shape and actual shape:
//   X bytes for expected: X = (a + 1) * (b) bytes
//  then directly following:
//   Y bytes for actual:   Y = (c + 1) * (d) bytes

const SHAPE_MAX_LEN: usize = (1 << 3) - 1;

struct ShapeEncoding {
    len: usize,
    element_width: usize,
    data: [u8; INFO_BYTES - 1],
}

#[derive(Copy, Clone)]
enum EncodingWidth {
    One = 1,
    Two = 2,
}

fn encode_shape(shape: &[usize], use_width: EncodingWidth) -> ShapeEncoding {
    let mut info = [0; INFO_BYTES - 1];
    match use_width {
        EncodingWidth::One => {
            for (i, &d) in enumerate(shape) {
                debug_assert!(d < 256);
                info[i] = d as u8;
            }
            ShapeEncoding {
                len: shape.len(),
                element_width: 1,
                data: info,
            }
        }
        EncodingWidth::Two => {
            for (i, &d) in enumerate(shape) {
                debug_assert!(d < 256 * 256);
                let dbytes = d.to_le_bytes();
                info[2 * i] = dbytes[0];
                info[2 * i + 1] = dbytes[1];
            }
            ShapeEncoding {
                len: shape.len(),
                element_width: 2,
                data: info,
            }
        }
    }
}

fn encode_shapes_impl(expected: &[usize], actual: &[usize]) -> InfoType {
    let exp_onebyte = expected.iter().all(|&i| i < 256);
    let exp_fit = exp_onebyte && expected.len() <= SHAPE_MAX_LEN ||
                   expected.iter().all(|&i| i < 256 * 256) && expected.len() <= (INFO_BYTES - 1) / 2;
    let act_onebyte = actual.iter().all(|&i| i < 256);

    let mut info = info_default();
    let mut info_type = EncodedInformationType::Nothing as u8;
    let mut shape_header = 0;

    let mut remaining_len = INFO_BYTES - 1;
    if exp_fit {
        info_type |= EncodedInformationType::Expected as u8;
        let eexp = encode_shape(expected, if exp_onebyte { EncodingWidth::One } else { EncodingWidth::Two });
        shape_header |= (!exp_onebyte as u8) << 0;

        info[2..].copy_from_slice(&eexp.data[..]);
        remaining_len -= eexp.len * eexp.element_width;
        shape_header |= (eexp.len as u8) << 1;
    }

    if remaining_len > 0 {
        if (act_onebyte && remaining_len >= actual.len()) ||
            remaining_len / 2 >= actual.len()
        {
            info_type |= EncodedInformationType::Actual as u8;
            let eact = encode_shape(actual, if act_onebyte { EncodingWidth::One } else { EncodingWidth::Two });
            shape_header |= (!act_onebyte as u8) << 4;
            let data_start = INFO_BYTES - 1 - remaining_len;

            info[2 + data_start..].copy_from_slice(&eact.data[..remaining_len]);
            shape_header |= (eact.len as u8) << 5;
        } else {
            // skip encoding
        }
    }
    info[0] = info_type;
    info[1] = shape_header;
    info
}

#[derive(Default)]
#[cfg_attr(test, derive(Debug))]
struct DecodedShape {
    len: usize,
    shape: [usize; 8],
}

impl DecodedShape {
    fn as_slice(&self) -> &[usize] {
        &self.shape[..self.len]
    }
}

fn decode_shape(data: &[u8], len: usize, width: EncodingWidth) -> DecodedShape {
    debug_assert!(len * (width as usize) <= data.len(),
        "Too short data when decoding shape");
    let mut shape = DecodedShape { len, ..<_>::default() };
    match width {
        EncodingWidth::One => {
            for (i, &d) in (0..len).zip(data) {
                shape.shape[i] = d as usize;
            }
        }
        EncodingWidth::Two => {
            for i in 0..len {
                let mut bytes = 0usize.to_le_bytes();
                bytes[0] = data[2 * i];
                bytes[1] = data[2 * i + 1];
                shape.shape[i] = usize::from_le_bytes(bytes);
            }
        }
    }
    shape
}

fn decode_shapes(info: InfoType) -> (Option<DecodedShape>, Option<DecodedShape>) {
    let exp_present = info[0] & (EncodedInformationType::Expected as u8) != 0;
    let act_present = info[0] & (EncodedInformationType::Actual as u8) != 0;
    let exp_twobyte = ((info[1] >> 0) & 0b1) != 0;
    let act_twobyte = ((info[1] >> 4) & 0b1) != 0;
    let exp_len_mask = if !act_present { !0u8 } else { (1u8 << 3) - 1 };
    let exp_len = ((info[1] >> 1) & exp_len_mask) as usize;
    let act_len = ((info[1] >> 5) & 0b111) as usize;
    let mut start = 2;
    let exp = if exp_present {
        let width = if !exp_twobyte { EncodingWidth::One } else { EncodingWidth::Two };
        let exp = decode_shape(&info[start..], exp_len, width);
        start += exp_len * width as usize;
        Some(exp)
    } else { None };
    let act = if act_present {
        let width = if !act_twobyte { EncodingWidth::One } else { EncodingWidth::Two };
        let act = decode_shape(&info[start..], act_len, width);
        Some(act)
    } else { None };
    (exp, act)
}

#[derive(Copy, Clone)]
struct ExtendedInfo<'a>(&'a ShapeError);

impl<'a> ExtendedInfo<'a> {
    fn has_info(&self) -> bool {
        self.0.info[0] != EncodedInformationType::Nothing as u8
    }
}

impl<'a> fmt::Display for ExtendedInfo<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.has_info() {
            return Ok(());
        }
        // Use the wording of "expected: X, but got: Y" for
        // expected and actual part of the error exented info.
        match self.0.kind() {
            ErrorKind::IncompatibleLayout => {
                let (expected, _) = decode_indices(self.0.info);
                match ExpectedLayout::from(expected) {
                    ExpectedLayout::ContiguousCF => {
                        write!(f, "; expected c- or f-contiguous input")?;
                    }
                    ExpectedLayout::Unused => {}
                }
            }
            ErrorKind::IncompatibleShape => {
                let (expected, actual) = decode_shapes(self.0.info);
                write!(f, "; expected compatible: ")?;
                if let Some(value) = expected {
                    write!(f, "{:?}", value.as_slice())?;
                } else {
                    write!(f, "unknown")?;
                }
                if let Some(value) = actual {
                    write!(f, ", but got: {:?}", value.as_slice())?;
                } else {
                    write!(f, "unknown")?;
                }
            }
            _otherwise => {
                let (expected, actual) = decode_indices(self.0.info);
                write!(f, "; expected: ")?;
                if let Some(value) = expected {
                    write!(f, "{}", value)?;
                } else {
                    write!(f, "unknown")?;
                }

                write!(f, ", but got: ")?;
                if let Some(value) = actual {
                    write!(f, "{}", value)?;
                } else {
                    write!(f, "unknown")?;
                }
            }
        }
        Ok(())
    }
}


#[cfg(test)]
use matches::assert_matches;
#[cfg(test)]
use crate::IntoDimension;

#[test]
fn test_sizes() {
    assert!(size_of::<ErrorKind>() <= size_of::<u8>());
    assert!(size_of::<ShapeError>() <= 16);

    assert_eq!(size_of::<Result<(), ShapeError>>(), size_of::<ShapeError>());
}

#[test]
fn test_encode_decode_format() {
    use alloc::string::ToString;

    assert_eq!(
        ShapeError::invalid_axis(1, 0).to_string(),
        "ShapeError/OutOfBounds: out of bounds indexing; expected: 1, but got: 0");

    if size_of::<usize>() > 4 {
        assert_eq!(
            ShapeError::invalid_axis(usize::MAX, usize::MAX).to_string(),
            "ShapeError/OutOfBounds: out of bounds indexing");
    }

    assert_eq!(
        ShapeError::incompatible_shapes(&(1, 2, 3).into_dimension(), &(2, 3).into_dimension())
            .to_string(),
        "ShapeError/IncompatibleShape: \
        incompatible shapes; expected compatible: [1, 2, 3], but got: [2, 3]");
}

#[test]
fn test_encode_decode() {
    for &i in [0, 1, 2, 3, 10, 32, 256, 1736, 16300].iter() {
        let err = ShapeError::invalid_axis(i, 0);
        assert_eq!(err.info_expected_index(), Some(i));
        let err = ShapeError::invalid_axis(0, i);
        assert_eq!(err.info_actual_index(), Some(i));
    }

    let err = ShapeError::invalid_axis(1 << 24, (1 << 24) + 1);
    assert_eq!(err.info_expected_index(), Some(1 << 24));
    assert_eq!(err.info_actual_index(), Some((1 << 24) + 1));

    if size_of::<usize>() > 4 {
        // use .wrapping_shl(_) for portability
        let err = ShapeError::invalid_axis(1usize.wrapping_shl(56) - 1, 0);
        assert_eq!(err.info_expected_index(), Some(1usize.wrapping_shl(56) - 1));
        assert_eq!(err.info_actual_index(), Some(0));

        let err = ShapeError::invalid_axis(1usize.wrapping_shl(56), 1usize.wrapping_shl(56));
        assert_eq!(err.info_expected_index(), None);
        assert_eq!(err.info_actual_index(), None);

        let err = ShapeError::invalid_axis(usize::MAX, usize::MAX);
        assert_eq!(err.info_expected_index(), None);
        assert_eq!(err.info_actual_index(), None);
    }
}

#[test]
fn test_encode_decode_shape() {
    let err = ShapeError::incompatible_shapes(&(1, 2).into_dimension(), &(4, 5).into_dimension());
    let (exp, act) = err.decode_shapes();
    assert_eq!(exp.unwrap().as_slice(), &[1, 2]);
    assert_eq!(act.unwrap().as_slice(), &[4, 5]);

    let err = ShapeError::incompatible_shapes(&(1, 2, 3).into_dimension(), &(4, 5, 6).into_dimension());
    let (exp, act) = err.decode_shapes();
    assert_eq!(exp.unwrap().as_slice(), &[1, 2, 3]);
    assert_eq!(act.unwrap().as_slice(), &[4, 5, 6]);

    let err = ShapeError::incompatible_shapes(&().into_dimension(), &().into_dimension());
    let (exp, act) = err.decode_shapes();
    assert_eq!(exp.unwrap().as_slice(), &[]);
    assert_eq!(act.unwrap().as_slice(), &[]);

    let (m, n) = (256, 768);
    let err = ShapeError::incompatible_shapes(&(m, n).into_dimension(), &(m + 1, n + 1).into_dimension());
    let (exp, act) = err.decode_shapes();
    assert_eq!(exp.unwrap().as_slice(), &[m, n]);
    assert_eq!(act.unwrap().as_slice(), &[m + 1, n + 1]);
    //assert!(act.is_none());

    let (m, n) = (256, 768);
    let err = ShapeError::incompatible_shapes(&(m, n).into_dimension(), &(m + 1).into_dimension());
    let (exp, act) = err.decode_shapes();
    assert_eq!(exp.unwrap().as_slice(), &[m, n]);
    assert_eq!(act.unwrap().as_slice(), &[m + 1]);

    let (m, n) = (256, 768);
    let err = ShapeError::incompatible_shapes(&m.into_dimension(), &(m + 1, n + 1).into_dimension());
    let (exp, act) = err.decode_shapes();
    assert_eq!(exp.unwrap().as_slice(), &[m]);
    assert_eq!(act.unwrap().as_slice(), &[m + 1, n + 1]);

    let err = ShapeError::incompatible_shapes(&(768, 2, 1024).into_dimension(), &(4, 500, 6).into_dimension());
    let (exp, act) = err.decode_shapes();
    assert_eq!(exp.unwrap().as_slice(), &[768, 2, 1024]);
    assert_eq!(act.unwrap().as_slice(), &[4, 500, 6]);

    let err = ShapeError::incompatible_shapes(&(768, 2, 1024, 3, 300).into_dimension(), &(4, 500, 6).into_dimension());
    let (exp, act) = err.decode_shapes();
    assert_eq!(exp.unwrap().as_slice(), &[768, 2, 1024, 3, 300]);
    assert_matches!(act, None);

    let err = ShapeError::incompatible_shapes(&(768, 2, 1024, 3, 300).into_dimension(), &(4, 6).into_dimension());
    let (exp, act) = err.decode_shapes();
    assert_eq!(exp.unwrap().as_slice(), &[768, 2, 1024, 3, 300]);
    assert_eq!(act.unwrap().as_slice(), &[4, 6]);

    let err = ShapeError::incompatible_shapes(&().into_dimension(), &(768, 2, 1024).into_dimension());
    let (exp, act) = err.decode_shapes();
    assert_eq!(exp.unwrap().as_slice(), &[]);
    assert_eq!(act.unwrap().as_slice(), &[768, 2, 1024]);

    let err = ShapeError::incompatible_shapes(&[1, 2, 3, 4, 5, 6, 7, 8].into_dimension(), &().into_dimension());
    let (exp, act) = err.decode_shapes();
    assert_matches!(exp, None);
    assert_eq!(act.unwrap().as_slice(), &[]);

    let err = ShapeError::incompatible_shapes(&[1, 2, 3, 4, 5, 6, 7].into_dimension(), &(1, 2).into_dimension());
    let (exp, act) = err.decode_shapes();
    assert_eq!(exp.unwrap().as_slice(), &[1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(act.unwrap().as_slice(), &[1, 2]);
}
