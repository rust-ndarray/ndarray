// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use serde::ser::impls::SeqIteratorVisitor;
use serde::{self, Serialize};

use imp_prelude::*;

use super::arraytraits::ARRAY_FORMAT_VERSION;
use super::Elements;

struct AVisitor<'a, D: 'a, S: 'a>
    where S: Data
{
    arr: &'a ArrayBase<S, D>,
    state: u32,
}

impl<A, D, S> Serialize for ArrayBase<S, D>
    where A: Serialize,
          D: Dimension + Serialize,
          S: DataOwned<Elem = A>

{
    fn serialize<Se>(&self, serializer: &mut Se) -> Result<(), Se::Error>
        where Se: serde::Serializer
    {
        serializer.serialize_struct("Array",
            AVisitor {
                arr: self,
                state: 0,
        })
    }
}

impl<'a, A, D> Serialize for Elements<'a, A, D>
    where A: Serialize,
          D: Dimension + Serialize
{
    fn serialize<S>(&self, serializer: &mut S) -> Result<(), S::Error>
        where S: serde::Serializer
    {
        serializer.serialize_seq(SeqIteratorVisitor::new(
            self.clone(),
            None,
        ))
    }
}

impl<'a, A, D, S> serde::ser::MapVisitor for AVisitor<'a, D, S>
    where A: Serialize,
          D: Serialize + Dimension,
          S: DataOwned<Elem = A>
{
    fn visit<Se>(&mut self, serializer: &mut Se) -> Result<Option<()>, Se::Error>
        where Se: serde::Serializer
    {
        match self.state {
            0 => {
                self.state +=1;
                Ok(Some(try!(serializer.serialize_map_elt("v", ARRAY_FORMAT_VERSION))))
            },
            1 => {
                self.state += 1;
                Ok(Some(try!(serializer.serialize_struct_elt("dim", self.arr.dim()))))
            },
            2 => {
                self.state += 1;
                Ok(Some(try!(serializer.serialize_struct_elt("data", self.arr.iter()))))
            },
            _ => Ok(None),
        }
    }
}
