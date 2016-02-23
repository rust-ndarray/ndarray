use serde::ser::impls::SeqIteratorVisitor;
use serde::{self, Serialize, Deserialize};

use super::{
    Array,
    Dimension,
    Ix,
    Elements,
};

struct AVisitor<'a, A: 'a, D: 'a> {
    arr: &'a RcArray<A, D>,
    state: u32,
}

impl<A: Serialize, D: Serialize> Serialize for RcArray<A, D>
    where D: Dimension
{
    fn serialize<S>(&self, serializer: &mut S) -> Result<(), S::Error>
        where S: serde::Serializer
    {
        serializer.visit_named_map("Array",
                                   AVisitor {
                                       arr: self,
                                       state: 0,
                                   })
    }
}

impl<'a, A: Serialize, D: Serialize> Serialize for Elements<'a, A, D>
    where D: Dimension
{
    fn serialize<S>(&self, serializer: &mut S) -> Result<(), S::Error>
        where S: serde::Serializer
    {
        serializer.visit_seq(SeqIteratorVisitor::new(self.clone(), None))
    }
}

impl<'a, A, D> serde::ser::MapVisitor for AVisitor<'a, A, D>
    where A: Serialize,
          D: Serialize + Dimension
{
    fn visit<S>(&mut self, serializer: &mut S) -> Result<Option<()>, S::Error>
        where S: serde::Serializer
    {
        match self.state {
            0 => {
                self.state += 1;
                Ok(Some(try!(serializer.visit_map_elt("shape", self.arr.dim()))))
            }
            1 => {
                self.state += 1;
                Ok(Some(try!(serializer.visit_map_elt("data", self.arr.iter()))))
            }
            _ => Ok(None),
        }

    }
}
