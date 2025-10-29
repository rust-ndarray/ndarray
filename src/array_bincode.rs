// Copyright 2014-2025 MiyakoMeow and ndarray developers.
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

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::{imp_prelude::*, IxDynImpl, ShapeError};

use super::arraytraits::ARRAY_FORMAT_VERSION;

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
        Decode::decode(decoder).map(Dim::new)
    }
}

/// **Requires crate feature `"bincode"`**
impl<'de, Context, I> BorrowDecode<'de, Context> for Dim<I>
where
    I: BorrowDecode<'de, Context>,
{
    fn borrow_decode<D: BorrowDecoder<'de, Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        BorrowDecode::borrow_decode(decoder).map(Dim::new)
    }
}

/// **Requires crate feature `"bincode"`**
impl Encode for IxDyn {
    fn encode<E: bincode::enc::Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        let ix: &IxDynImpl = self.ix();
        Encode::encode(&ix.len(), encoder)?;
        ix.into_iter()
            .try_for_each(|ix| Encode::encode(ix, encoder))
    }
}

/// **Requires crate feature `"bincode"`**
impl<Context> Decode<Context> for IxDynImpl {
    fn decode<D: Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let len: usize = Decode::decode(decoder)?;
        (0..len)
            .map(|_| Decode::decode(decoder))
            .collect::<Result<Vec<_>, DecodeError>>()
            .map(IxDynImpl::from)
    }
}

/// **Requires crate feature `"bincode"`**
impl<'de, Context> bincode::BorrowDecode<'de, Context> for IxDynImpl {
    fn borrow_decode<D: BorrowDecoder<'de, Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let len: usize = BorrowDecode::borrow_decode(decoder)?;
        (0..len)
            .map(|_| BorrowDecode::borrow_decode(decoder))
            .collect::<Result<Vec<_>, DecodeError>>()
            .map(IxDynImpl::from)
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
        iter.into_iter()
            .try_for_each(|elt| Encode::encode(elt, encoder))
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
        let data: Vec<_> = (0..data_len)
            .map(|_| Decode::decode(decoder))
            .collect::<Result<Vec<_>, DecodeError>>()?;
        let expected_size = dim.size();
        ArrayBase::from_shape_vec(dim, data).map_err(|_err: ShapeError| DecodeError::ArrayLengthMismatch {
            required: expected_size,
            found: data_len,
        })
    }
}
