// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use serde::{Serialize, Serializer, Deserialize, Deserializer};
use serde::de::{self, Visitor, SeqAccess, MapAccess};
use serde::ser::{SerializeSeq, SerializeStruct};

use std::fmt;
use std::marker::PhantomData;

use imp_prelude::*;

use super::arraytraits::ARRAY_FORMAT_VERSION;
use super::Iter;

/// Verifies that the version of the deserialized array matches the current
/// `ARRAY_FORMAT_VERSION`.
pub fn verify_version<E>(v: u8) -> Result<(), E>
        where E: de::Error
{
    if v != ARRAY_FORMAT_VERSION {
        let err_msg = format!("unknown array version: {}", v);
        try!(Err(de::Error::custom(err_msg)));
    }

    Ok(())
}

/// **Requires crate feature `"serde-1"`**
impl<I> Serialize for Dim<I>
    where I: Serialize,
{
    fn serialize<Se>(&self, serializer: Se) -> Result<Se::Ok, Se::Error>
        where Se: Serializer
    {
        self.ix().serialize(serializer)
    }
}

/// **Requires crate feature `"serde-1"`**
impl<'de, I> Deserialize<'de> for Dim<I>
    where I: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de>
    {
        I::deserialize(deserializer).map(Dim::new)
    }
}

/// **Requires crate feature `"serde-1"`**
impl<A, D, S> Serialize for ArrayBase<S, D>
    where A: Serialize,
          D: Dimension + Serialize,
          S: Data<Elem = A>

{
    fn serialize<Se>(&self, serializer: Se) -> Result<Se::Ok, Se::Error>
        where Se: Serializer
    {
        let mut state = try!(serializer.serialize_struct("Array", 3));
        try!(state.serialize_field("v", &ARRAY_FORMAT_VERSION));
        try!(state.serialize_field("dim", &self.raw_dim()));
        try!(state.serialize_field("data", &Sequence(self.iter())));
        state.end()
    }
}

// private iterator wrapper
struct Sequence<'a, A: 'a, D>(Iter<'a, A, D>);

impl<'a, A, D> Serialize for Sequence<'a, A, D>
    where A: Serialize,
          D: Dimension + Serialize
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        let iter = &self.0;
        let mut seq = try!(serializer.serialize_seq(Some(iter.len())));
        for elt in iter.clone() {
            try!(seq.serialize_element(elt));
        }
        seq.end()
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

static ARRAY_FIELDS: &'static [&'static str] = &["v", "dim", "data"];

/// **Requires crate feature `"serde-1"`**
impl<'de, A, Di, S> Deserialize<'de> for ArrayBase<S, Di>
    where A: Deserialize<'de>,
          Di: Deserialize<'de> + Dimension,
          S: DataOwned<Elem = A>
{
    fn deserialize<D>(deserializer: D) -> Result<ArrayBase<S, Di>, D::Error>
        where D: Deserializer<'de>
    {
        deserializer.deserialize_struct("Array", ARRAY_FIELDS, ArrayVisitor::new())
    }
}

impl<'de> Deserialize<'de> for ArrayField {
    fn deserialize<D>(deserializer: D) -> Result<ArrayField, D::Error>
        where D: Deserializer<'de>
    {
        struct ArrayFieldVisitor;

        impl<'de> Visitor<'de> for ArrayFieldVisitor {
            type Value = ArrayField;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str(r#""v", "dim", or "data""#)
            }

            fn visit_str<E>(self, value: &str) -> Result<ArrayField, E>
                where E: de::Error
            {
                match value {
                    "v" => Ok(ArrayField::Version),
                    "dim" => Ok(ArrayField::Dim),
                    "data" => Ok(ArrayField::Data),
                    other => Err(de::Error::unknown_field(other, ARRAY_FIELDS)),
                }
            }

            fn visit_bytes<E>(self, value: &[u8]) -> Result<ArrayField, E>
                where E: de::Error
            {
                match value {
                    b"v" => Ok(ArrayField::Version),
                    b"dim" => Ok(ArrayField::Dim),
                    b"data" => Ok(ArrayField::Data),
                    other => Err(de::Error::unknown_field(&format!("{:?}",other), ARRAY_FIELDS)),
                }
            }
        }

        deserializer.deserialize_identifier(ArrayFieldVisitor)
    }
}

impl<'de, A, Di, S> Visitor<'de> for ArrayVisitor<S,Di>
    where A: Deserialize<'de>,
          Di: Deserialize<'de> + Dimension,
          S: DataOwned<Elem = A>
{
    type Value = ArrayBase<S, Di>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("ndarray representation")
    }

    fn visit_seq<V>(self, mut visitor: V) -> Result<ArrayBase<S, Di>, V::Error>
        where V: SeqAccess<'de>,
    {
        let v: u8 = match try!(visitor.next_element()) {
            Some(value) => value,
            None => {
                return Err(de::Error::invalid_length(0, &self));
            }
        };

        try!(verify_version(v));

        let dim: Di = match try!(visitor.next_element()) {
            Some(value) => value,
            None => {
                return Err(de::Error::invalid_length(1, &self));
            }
        };

        let data: Vec<A> = match try!(visitor.next_element()) {
            Some(value) => value,
            None => {
                return Err(de::Error::invalid_length(2, &self));
            }
        };

        if let Ok(array) = ArrayBase::from_shape_vec(dim, data) {
            Ok(array)
        } else {
            Err(de::Error::custom("data and dimension must match in size"))
        }
    }

    fn visit_map<V>(self, mut visitor: V) -> Result<ArrayBase<S, Di>, V::Error>
        where V: MapAccess<'de>,
    {
        let mut v: Option<u8> = None;
        let mut data: Option<Vec<A>> = None;
        let mut dim: Option<Di> = None;

        while let Some(key) = try!(visitor.next_key()) {
            match key {
                ArrayField::Version => {
                    let val = try!(visitor.next_value());
                    try!(verify_version(val));
                    v = Some(val);
                },
                ArrayField::Data => {
                    data = Some(try!(visitor.next_value()));
                },
                ArrayField::Dim => {
                    dim = Some(try!(visitor.next_value()));
                },
            }
        }

        let _v = match v {
            Some(v) => v,
            None => try!(Err(de::Error::missing_field("v"))),
        };

        let data = match data {
            Some(data) => data,
            None => try!(Err(de::Error::missing_field("data"))),
        };

        let dim = match dim {
            Some(dim) => dim,
            None => try!(Err(de::Error::missing_field("dim"))),
        };

        if let Ok(array) = ArrayBase::from_shape_vec(dim, data) {
            Ok(array)
        } else {
            Err(de::Error::custom("data and dimension must match in size"))
        }
    }
}
