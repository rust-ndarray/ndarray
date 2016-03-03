// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use serde::{self, Deserialize};

use std::marker::PhantomData;

use imp_prelude::*;

use super::arraytraits::ARRAY_FORMAT_VERSION;

struct ArrayVisitor<S,Di> {
    _marker_a: PhantomData<S>,
    _marker_b: PhantomData<Di>,
}

enum ArrayField {
    Version,
    Dim,
    Data,
}

impl<S,Di> ArrayVisitor<S,Di> {
    pub fn new() -> Self {
        ArrayVisitor { _marker_a: PhantomData, _marker_b: PhantomData, }
    }
}

impl<A, Di, S> Deserialize for ArrayBase<S, Di>
    where A: Deserialize,
          Di: Deserialize + Dimension,
          S: DataOwned<Elem = A>
{
    fn deserialize<D>(deserializer: &mut D) -> Result<ArrayBase<S,Di>, D::Error>
        where D: serde::de::Deserializer
    {
        static FIELDS: &'static [&'static str] = &["v", "dim", "data"];

        deserializer.deserialize_struct("Array", FIELDS, ArrayVisitor::new())
    }
}

impl serde::de::Deserialize for ArrayField {
    fn deserialize<D>(deserializer: &mut D) -> Result<ArrayField, D::Error>
        where D: serde::de::Deserializer
    {
        struct ArrayFieldVisitor;

        impl serde::de::Visitor for ArrayFieldVisitor {
            type Value = ArrayField;

            fn visit_str<E>(&mut self, value: &str) -> Result<ArrayField, E>
                where E: serde::de::Error
            {
                match value {
                    "v" => Ok(ArrayField::Version),
                    "data" => Ok(ArrayField::Data),
                    "dim" => Ok(ArrayField::Dim),
                    _ => Err(serde::de::Error::custom("expected v, data, or dim")),
                }
            }
        }

        deserializer.deserialize(ArrayFieldVisitor)
    }
}

impl<A, Di, S> serde::de::Visitor for ArrayVisitor<S,Di>
    where A: Deserialize,
          Di: Deserialize + Dimension,
          S: DataOwned<Elem = A>
{
    type Value = ArrayBase<S, Di>;

    fn visit_map<V>(&mut self, mut visitor: V) -> Result<ArrayBase<S,Di>, V::Error>
        where V: serde::de::MapVisitor,
    {
        let mut v: Option<u8> = None;
        let mut data: Option<Vec<A>> = None;
        let mut dim: Option<Di> = None;

        loop {
            match try!(visitor.visit_key()) {
                Some(ArrayField::Version) => { v = Some(try!(visitor.visit_value())); },
                Some(ArrayField::Data) => { data = Some(try!(visitor.visit_value())); },
                Some(ArrayField::Dim) => { dim = Some(try!(visitor.visit_value())); },
                None => { break; },
            }
        }

        let v = match v {
            Some(v) => v,
            None => try!(visitor.missing_field("v")),
        };

        if v != ARRAY_FORMAT_VERSION {
            try!(Err(serde::de::Error::custom(format!("unknown array version: {}", v))));
        }

        let data = match data {
            Some(data) => data,
            None => try!(visitor.missing_field("data")),
        };

        let dim = match dim {
            Some(dim) => dim,
            None => try!(visitor.missing_field("dim")),
        };

        if data.len() != dim.size() {
            try!(Err(serde::de::Error::custom("data and dimension must match in size")));
        }

        try!(visitor.end());

        unsafe {
            Ok(ArrayBase::from_shape_vec_unchecked(dim, data))
        }
    }
}
