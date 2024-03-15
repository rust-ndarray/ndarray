use crate::imp_prelude::*;
use crate::IntoDimension;
use alloc::vec::Vec;
use borsh::{BorshDeserialize, BorshSerialize};
use core::ops::Deref;

/// **Requires crate feature `"borsh"`**
impl<I> BorshSerialize for Dim<I>
where
    I: BorshSerialize,
{
    fn serialize<W: borsh::io::Write>(&self, writer: &mut W) -> borsh::io::Result<()> {
        <I as BorshSerialize>::serialize(&self.ix(), writer)
    }
}

/// **Requires crate feature `"borsh"`**
impl<I> BorshDeserialize for Dim<I>
where
    I: BorshDeserialize,
{
    fn deserialize_reader<R: borsh::io::Read>(reader: &mut R) -> borsh::io::Result<Self> {
        <I as BorshDeserialize>::deserialize_reader(reader).map(Dim::new)
    }
}

/// **Requires crate feature `"borsh"`**
impl BorshSerialize for IxDyn {
    fn serialize<W: borsh::io::Write>(&self, writer: &mut W) -> borsh::io::Result<()> {
        let elts = self.ix().deref();
        // Output length of dimensions.
        <usize as BorshSerialize>::serialize(&elts.len(), writer)?;
        // Followed by actual data.
        for elt in elts {
            <Ix as BorshSerialize>::serialize(elt, writer)?;
        }
        Ok(())
    }
}

/// **Requires crate feature `"borsh"`**
impl BorshDeserialize for IxDyn {
    fn deserialize_reader<R: borsh::io::Read>(reader: &mut R) -> borsh::io::Result<Self> {
        // Deserialize the length.
        let len = <usize as BorshDeserialize>::deserialize_reader(reader)?;
        // Deserialize the given number of elements. We assume the source is
        // trusted so we use a capacity hint...
        let mut elts = Vec::with_capacity(len);
        for _ix in 0..len {
            elts.push(<Ix as BorshDeserialize>::deserialize_reader(reader)?);
        }
        Ok(elts.into_dimension())
    }
}

/// **Requires crate feature `"borsh"`**
impl<A, D, S> BorshSerialize for ArrayBase<S, D>
where
    A: BorshSerialize,
    D: Dimension + BorshSerialize,
    S: Data<Elem = A>,
{
    fn serialize<W: borsh::io::Write>(&self, writer: &mut W) -> borsh::io::Result<()> {
        // Dimensions
        <D as BorshSerialize>::serialize(&self.raw_dim(), writer)?;
        // Followed by length of data
        let iter = self.iter();
        <usize as BorshSerialize>::serialize(&iter.len(), writer)?;
        // Followed by data itself.
        for elt in iter {
            <A as BorshSerialize>::serialize(elt, writer)?;
        }
        Ok(())
    }
}

/// **Requires crate feature `"borsh"`**
impl<A, D, S> BorshDeserialize for ArrayBase<S, D>
where
    A: BorshDeserialize,
    D: BorshDeserialize + Dimension,
    S: DataOwned<Elem = A>,
{
    fn deserialize_reader<R: borsh::io::Read>(reader: &mut R) -> borsh::io::Result<Self> {
        // Dimensions
        let dim = <D as BorshDeserialize>::deserialize_reader(reader)?;
        // Followed by length of data
        let len = <usize as BorshDeserialize>::deserialize_reader(reader)?;
        // Followed by data itself.
        let mut data = Vec::with_capacity(len);
        for _ix in 0..len {
            data.push(<A as BorshDeserialize>::deserialize_reader(reader)?);
        }
        ArrayBase::from_shape_vec(dim, data).map_err(|_shape_err| {
            borsh::io::Error::new(
                borsh::io::ErrorKind::InvalidData,
                "data and dimensions must match in size",
            )
        })
    }
}
