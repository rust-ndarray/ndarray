// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use serialize::{Encodable, Encoder, Decodable, Decoder};
use super::arraytraits::ARRAY_FORMAT_VERSION;

use imp_prelude::*;

/// **Requires crate feature `"rustc-serialize"`**
impl<A, S, D> Encodable for ArrayBase<S, D>
    where A: Encodable,
          D: Dimension + Encodable,
          S: Data<Elem = A>
{
    fn encode<E: Encoder>(&self, s: &mut E) -> Result<(), E::Error> {
        s.emit_struct("Array", 3, |e| {
            try!(e.emit_struct_field("v", 0, |e| ARRAY_FORMAT_VERSION.encode(e)));
            // FIXME: Write self.dim as a slice (self.shape)
            // The problem is decoding it.
            try!(e.emit_struct_field("dim", 1, |e| self.dim.encode(e)));
            try!(e.emit_struct_field("data", 2, |e| {
                let sz = self.dim.size();
                e.emit_seq(sz, |e| {
                    for (i, elt) in self.iter().enumerate() {
                        try!(e.emit_seq_elt(i, |e| elt.encode(e)))
                    }
                    Ok(())
                })
            }));
            Ok(())
        })
    }
}

/// **Requires crate feature `"rustc-serialize"`**
impl<A, S, D> Decodable for ArrayBase<S, D>
    where A: Decodable,
          D: Dimension + Decodable,
          S: DataOwned<Elem = A>
{
    fn decode<E: Decoder>(d: &mut E) -> Result<ArrayBase<S, D>, E::Error> {
        d.read_struct("Array", 3, |d| {
            let version: u8 = try!(d.read_struct_field("v", 0, Decodable::decode));
            if version > ARRAY_FORMAT_VERSION {
                return Err(d.error("unknown array version"))
            }
            let dim: D = try!(d.read_struct_field("dim", 1, |d| {
                Decodable::decode(d)
            }));

            let elements = try!(
                d.read_struct_field("data", 2, |d| {
                    d.read_seq(|d, len| {
                        if len != dim.size() {
                            Err(d.error("data and dimension must match in size"))
                        } else {
                            let mut elements = Vec::with_capacity(len);
                            for i in 0..len {
                                elements.push(try!(d.read_seq_elt::<A, _>(i, Decodable::decode)))
                            }
                            Ok(elements)
                        }
                    })
            }));
            unsafe {
                Ok(ArrayBase::from_shape_vec_unchecked(dim, elements))
            }
        })
    }
}

