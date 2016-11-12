// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use serde::{self, Serialize, Deserialize};

use std::marker::PhantomData;

use imp_prelude::*;

use super::arraytraits::ARRAY_FORMAT_VERSION;
use super::Iter;
use Dim;
use dimension::DimPrivate;

/// **Requires crate feature `"serde"`**
impl<I> Serialize for Dim<I>
    where I: Serialize,
{
    fn serialize<Se>(&self, serializer: &mut Se) -> Result<(), Se::Error>
        where Se: serde::Serializer
    {
        self.ix().serialize(serializer)
    }
}

/// **Requires crate feature `"serde"`**
impl<I> Deserialize for Dim<I>
    where I: Deserialize,
{
    fn deserialize<D>(deserializer: &mut D) -> Result<Self, D::Error>
        where D: serde::de::Deserializer
    {
        I::deserialize(deserializer).map(Dim::new)
    }
}

/// **Requires crate feature `"serde"`**
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
        try!(serializer.serialize_struct_elt(&mut struct_state, "data", Sequence(self.iter())));
        serializer.serialize_struct_end(struct_state)
    }
}

// private iterator wrapper
struct Sequence<'a, A: 'a, D>(Iter<'a, A, D>);

impl<'a, A, D> Serialize for Sequence<'a, A, D>
    where A: Serialize,
          D: Dimension + Serialize
{
    fn serialize<S>(&self, serializer: &mut S) -> Result<(), S::Error>
        where S: serde::Serializer
    {
        let iter = &self.0;
        let mut seq_state = try!(serializer.serialize_seq(Some(iter.len())));
        for elt in iter.clone() {
            try!(serializer.serialize_seq_elt(&mut seq_state, elt));
        }
        serializer.serialize_seq_end(seq_state)
    }
}

struct ArrayVisitor<S, Di> {
    _marker_a: PhantomData<S>,
    _marker_b: PhantomData<Di>,
}

enum ArrayField {
    Version,
    Dim,
    Data,
}

impl<S, Di> ArrayVisitor<S, Di> {
    pub fn new() -> Self {
        ArrayVisitor { _marker_a: PhantomData, _marker_b: PhantomData, }
    }
}

/// **Requires crate feature `"serde"`**
impl<A, Di, S> Deserialize for ArrayBase<S, Di>
    where A: Deserialize,
          Di: Deserialize + Dimension,
          S: DataOwned<Elem = A>
{
    fn deserialize<D>(deserializer: &mut D) -> Result<ArrayBase<S, Di>, D::Error>
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

    fn visit_map<V>(&mut self, mut visitor: V) -> Result<ArrayBase<S, Di>, V::Error>
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

        if let Ok(array) = ArrayBase::from_shape_vec(dim, data) {
            try!(visitor.end());
            Ok(array)
        } else {
            Err(serde::de::Error::custom("data and dimension must match in size"))
        }
    }
}
