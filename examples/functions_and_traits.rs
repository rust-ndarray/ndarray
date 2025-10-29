//! Examples of how to write functions and traits that operate on `ndarray` types.
//!
//! `ndarray` has four kinds of array types that users may interact with:
//!     1. [`ArrayBase`], the owner of the layout that describes an array in memory;
//!         this includes [`ndarray::Array`], [`ndarray::ArcArray`], [`ndarray::ArrayView`],
//!         [`ndarray::RawArrayView`], and other variants.
//!     2. [`ArrayRef`], which represents a read-safe, uniquely-owned look at an array.
//!     3. [`RawRef`], which represents a read-unsafe, possibly-shared look at an array.
//!     4. [`LayoutRef`], which represents a look at an array's underlying structure,
//!         but does not allow data reading of any kind.
//!
//! Below, we illustrate how to write functions and traits for most variants of these types.

use ndarray::{ArrayBase, ArrayRef, Data, DataMut, Dimension, LayoutRef, RawData, RawDataMut, RawRef};

/// Take an array with the most basic requirements.
///
/// This function takes its data as owning. It is very rare that a user will need to specifically
/// take a reference to an `ArrayBase`, rather than to one of the other four types.
#[rustfmt::skip]
fn takes_base_raw<S: RawData, D>(arr: ArrayBase<S, D>) -> ArrayBase<S, D>
{
    // These skip from a possibly-raw array to `RawRef` and `LayoutRef`, and so must go through `AsRef`
    takes_rawref(arr.as_ref()); // Caller uses `.as_ref`
    takes_rawref_asref(&arr);   // Implementor uses `.as_ref`
    takes_layout(arr.as_ref()); // Caller uses `.as_ref`
    takes_layout_asref(&arr);   // Implementor uses `.as_ref`

    arr
}

/// Similar to above, but allow us to read the underlying data.
#[rustfmt::skip]
fn takes_base_raw_mut<S: RawDataMut, D>(mut arr: ArrayBase<S, D>) -> ArrayBase<S, D>
{
    // These skip from a possibly-raw array to `RawRef` and `LayoutRef`, and so must go through `AsMut`
    takes_rawref_mut(arr.as_mut()); // Caller uses `.as_mut`
    takes_rawref_asmut(&mut arr);   // Implementor uses `.as_mut`
    takes_layout_mut(arr.as_mut()); // Caller uses `.as_mut`
    takes_layout_asmut(&mut arr);   // Implementor uses `.as_mut`

    arr
}

/// Now take an array whose data is safe to read.
#[allow(dead_code)]
fn takes_base<S: Data, D>(mut arr: ArrayBase<S, D>) -> ArrayBase<S, D>
{
    // Raw call
    arr = takes_base_raw(arr);

    // No need for AsRef, since data is safe
    takes_arrref(&arr);
    takes_rawref(&arr);
    takes_rawref_asref(&arr);
    takes_layout(&arr);
    takes_layout_asref(&arr);

    arr
}

/// Now, an array whose data is safe to read and that we can mutate.
///
/// Notice that we include now a trait bound on `D: Dimension`; this is necessary in order
/// for the `ArrayBase` to dereference to an `ArrayRef` (or to any of the other types).
#[allow(dead_code)]
fn takes_base_mut<S: DataMut, D: Dimension>(mut arr: ArrayBase<S, D>) -> ArrayBase<S, D>
{
    // Raw call
    arr = takes_base_raw_mut(arr);

    // No need for AsMut, since data is safe
    takes_arrref_mut(&mut arr);
    takes_rawref_mut(&mut arr);
    takes_rawref_asmut(&mut arr);
    takes_layout_mut(&mut arr);
    takes_layout_asmut(&mut arr);

    arr
}

/// Now for new stuff: we want to read (but not alter) any array whose data is safe to read.
///
/// This is probably the most common functionality that one would want to write.
/// As we'll see below, calling this function is very simple for `ArrayBase<S: Data, D>`.
fn takes_arrref<A, D>(arr: &ArrayRef<A, D>)
{
    // No need for AsRef, since data is safe
    takes_rawref(arr);
    takes_rawref_asref(arr);
    takes_layout(arr);
    takes_layout_asref(arr);
}

/// Now we want any array whose data is safe to mutate.
///
/// **Importantly**, any array passed to this function is guaranteed to uniquely point to its data.
/// As a result, passing a shared array to this function will **silently** un-share the array.
#[allow(dead_code)]
fn takes_arrref_mut<A, D>(arr: &mut ArrayRef<A, D>)
{
    // Immutable call
    takes_arrref(arr);

    // No need for AsMut, since data is safe
    takes_rawref_mut(arr);
    takes_rawref_asmut(arr);
    takes_layout_mut(arr);
    takes_rawref_asmut(arr);
}

/// Now, we no longer care about whether we can safely read data.
///
/// This is probably the rarest type to deal with, since `LayoutRef` can access and modify an array's
/// shape and strides, and even do in-place slicing. As a result, `RawRef` is only for functionality
/// that requires unsafe data access, something that `LayoutRef` can't do.
///
/// Writing functions and traits that deal with `RawRef`s and `LayoutRef`s can be done two ways:
///     1. Directly on the types; calling these functions on arrays whose data are not known to be safe
///         to dereference (i.e., raw array views or `ArrayBase<S: RawData, D>`) must explicitly call `.as_ref()`.
///     2. Via a generic with `: AsRef<RawRef<A, D>>`; doing this will allow direct calling for all `ArrayBase` and
///         `ArrayRef` instances.
/// We'll demonstrate #1 here for both immutable and mutable references, then #2 directly below.
#[allow(dead_code)]
fn takes_rawref<A, D>(arr: &RawRef<A, D>)
{
    takes_layout(arr);
    takes_layout_asref(arr);
}

/// Mutable, directly take `RawRef`
#[allow(dead_code)]
fn takes_rawref_mut<A, D>(arr: &mut RawRef<A, D>)
{
    takes_layout(arr);
    takes_layout_asmut(arr);
}

/// Immutable, take a generic that implements `AsRef` to `RawRef`
#[allow(dead_code)]
fn takes_rawref_asref<T, A, D>(_arr: &T)
where T: AsRef<RawRef<A, D>> + ?Sized
{
    takes_layout(_arr.as_ref());
    takes_layout_asref(_arr.as_ref());
}

/// Mutable, take a generic that implements `AsMut` to `RawRef`
#[allow(dead_code)]
fn takes_rawref_asmut<T, A, D>(_arr: &mut T)
where T: AsMut<RawRef<A, D>> + ?Sized
{
    takes_layout_mut(_arr.as_mut());
    takes_layout_asmut(_arr.as_mut());
}

/// Finally, there's `LayoutRef`: this type provides read and write access to an array's *structure*, but not its *data*.
///
/// Practically, this means that functions that only read/modify an array's shape or strides,
/// such as checking dimensionality or slicing, should take `LayoutRef`.
///
/// Like `RawRef`, functions can be written either directly on `LayoutRef` or as generics with `: AsRef<LayoutRef<A, D>>>`.
#[allow(dead_code)]
fn takes_layout<A, D>(_arr: &LayoutRef<A, D>) {}

/// Mutable, directly take `LayoutRef`
#[allow(dead_code)]
fn takes_layout_mut<A, D>(_arr: &mut LayoutRef<A, D>) {}

/// Immutable, take a generic that implements `AsRef` to `LayoutRef`
#[allow(dead_code)]
fn takes_layout_asref<T, A, D>(_arr: &T)
where T: AsRef<LayoutRef<A, D>> + ?Sized
{
}

/// Mutable, take a generic that implements `AsMut` to `LayoutRef`
#[allow(dead_code)]
fn takes_layout_asmut<T, A, D>(_arr: &mut T)
where T: AsMut<LayoutRef<A, D>> + ?Sized
{
}

fn main() {}
