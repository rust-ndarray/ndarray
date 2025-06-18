// Copyright 2014-2025 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use bincode::{
    de::{BorrowDecoder, Decoder},
    enc::Encoder,
    error::{DecodeError, EncodeError},
    BorrowDecode, Decode, Encode,
};

use alloc::format;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::{imp_prelude::*, IntoDimension, IxDynImpl, ShapeError};

use super::arraytraits::ARRAY_FORMAT_VERSION;
use crate::iterators::Iter;

/// Verifies that the version of the deserialized array matches the current
/// `ARRAY_FORMAT_VERSION`.
pub fn verify_version(v: u8) -> Result<(), DecodeError> {
    if v != ARRAY_FORMAT_VERSION {
        let err_msg = format!("unknown array version: {}", v);
        Err(DecodeError::OtherString(err_msg))
    } else {
        Ok(())
    }
}

/// **Requires crate feature `"bincode"`**
impl<I> Encode for Dim<I>
where
    I: Encode,
{
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&self.ix(), encoder)
    }
}

/// **Requires crate feature `"bincode"`**
impl<Context, I> Decode<Context> for Dim<I>
where
    I: Decode<Context>,
{
    fn decode<D: Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(Dim::new(Decode::decode(decoder)?))
    }
}

/// **Requires crate feature `"bincode"`**
impl<'de, Context, I> BorrowDecode<'de, Context> for Dim<I>
where
    I: BorrowDecode<'de, Context>,
{
    fn borrow_decode<D: BorrowDecoder<'de, Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(Dim::new(BorrowDecode::borrow_decode(decoder)?))
    }
}

/// **Requires crate feature `"bincode"`**
impl Encode for IxDyn {
    fn encode<E: bincode::enc::Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        let ix: &IxDynImpl = self.ix();
        Encode::encode(&ix.len(), encoder)?;
        for ix in ix.into_iter() {
            Encode::encode(ix, encoder)?;
        }
        Ok(())
    }
}

/// **Requires crate feature `"bincode"`**
impl<Context> Decode<Context> for IxDynImpl {
    fn decode<D: Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let len: usize = Decode::decode(decoder)?;
        let vals = (0..len)
            .map(|_| Decode::decode(decoder))
            .collect::<Result<Vec<_>, DecodeError>>()?;
        Ok(IxDynImpl::from(vals))
    }
}

/// **Requires crate feature `"bincode"`**
impl<'de, Context> bincode::BorrowDecode<'de, Context> for IxDynImpl {
    fn borrow_decode<D: BorrowDecoder<'de, Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let len: usize = BorrowDecode::borrow_decode(decoder)?;
        let vals = (0..len)
            .map(|_| BorrowDecode::borrow_decode(decoder))
            .collect::<Result<Vec<_>, DecodeError>>()?;
        Ok(IxDynImpl::from(vals))
    }
}

/// **Requires crate feature `"serde"`**
impl<A, D, S> Encode for ArrayBase<S, D>
where
    A: Encode,
    D: Dimension + Encode,
    S: Data<Elem = A>,
{
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&ARRAY_FORMAT_VERSION, encoder)?;
        Encode::encode(&self.raw_dim(), encoder)?;
        let iter = self.iter();
        Encode::encode(&iter.len(), encoder)?;
        for elt in iter.clone() {
            Encode::encode(elt, encoder)?;
        }
        Ok(())
    }
}

/// **Requires crate feature `"bincode"`**
impl<A, D, S, Context> Decode<Context> for ArrayBase<S, D>
where
    A: Decode<Context>,
    D: Dimension + Decode<Context>,
    S: DataOwned<Elem = A>,
{
    fn decode<De: Decoder<Context = Context>>(decoder: &mut De) -> Result<Self, DecodeError> {
        let data_version: u8 = Decode::decode(decoder)?;
        (data_version == ARRAY_FORMAT_VERSION)
            .then_some(())
            .ok_or(DecodeError::Other("ARRAY_FORMAT_VERSION not match!"))?;
        let dim: D = Decode::decode(decoder)?;
        let data_len: usize = Decode::decode(decoder)?;
        let data = (0..data_len)
            .map(|_| Decode::decode(decoder))
            .collect::<Result<Vec<_>, DecodeError>>()?;
        let expected_size = dim.size();
        ArrayBase::from_shape_vec(dim, data).map_err(|_err: ShapeError| DecodeError::ArrayLengthMismatch {
            required: expected_size,
            found: data_len,
        })
    }
}
