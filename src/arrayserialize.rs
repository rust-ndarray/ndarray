// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use serde::{self, Serialize};

use imp_prelude::*;

use super::arraytraits::ARRAY_FORMAT_VERSION;
use super::Elements;

impl<A, D, S> Serialize for ArrayBase<S, D>
    where A: Serialize,
          D: Dimension + Serialize,
          S: Data<Elem = A>

{
    fn serialize<Se>(&self, serializer: &mut Se) -> Result<(), Se::Error>
        where Se: serde::Serializer
    {
        let mut struct_state = try!(serializer.serialize_struct("Array", 3));
        try!(serializer.serialize_struct_elt(&mut struct_state, "v", ARRAY_FORMAT_VERSION));
        try!(serializer.serialize_struct_elt(&mut struct_state, "dim", self.dim()));
        try!(serializer.serialize_struct_elt(&mut struct_state, "data", self.iter()));
        serializer.serialize_struct_end(struct_state)
    }
}

impl<'a, A, D> Serialize for Elements<'a, A, D>
    where A: Serialize,
          D: Dimension + Serialize
{
    fn serialize<S>(&self, serializer: &mut S) -> Result<(), S::Error>
        where S: serde::Serializer
    {
        let mut seq_state = try!(serializer.serialize_seq(Some(self.len())));
        for elt in self.clone() {
            try!(serializer.serialize_seq_elt(&mut seq_state, elt));
        }
        serializer.serialize_seq_end(seq_state)
    }
}
