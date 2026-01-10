//! Examples of how to write functions and traits that operate on `ndarray` types.
//!
//! `ndarray` has four kinds of array types that users may interact with:
//!     1. [`ArrayBase`], the owner of the layout that describes an array in memory;
//!         this includes [`ndarray::Array`], [`ndarray::ArcArray`], [`ndarray::ArrayView`],
//!         [`ndarray::RawArrayView`], and other variants.
//!     2. [`ArrayRef`], which represents a read-safe, uniquely-owned look at an array.
//!     3. [`RawRef`], which represents a read-unsafe, possibly-shared look at an array.
//!     4. [`LayoutRef`], which represents a look at an array's underlying structure,
//!         but does not allow reading data of any kind.
//!
//! Below, we illustrate how to write functions and traits for most variants of these types.

use ndarray::{ArrayBase, ArrayRef, Data, DataMut, Dimension, LayoutRef, RawRef};

/// First, the newest pattern: this function accepts arrays whose data are safe to
/// dereference and uniquely held.
///
/// This is probably the most common pattern for users.
/// Once we have an array reference, we can go to [`RawRef`] and [`LayoutRef`] very easily.
fn takes_arrref<A, D>(arr: &ArrayRef<A, D>)
{
    // Since `ArrayRef` implements `Deref` to `RawRef`, we can pass `arr` directly to a function
    // that takes `RawRef`. Similarly, since `RawRef` implements `Deref` to `LayoutRef`, we can pass
    // `arr` directly to a function that takes `LayoutRef`.
    takes_rawref(arr); // &ArrayRef -> &RawRef
    takes_layout(arr); // &ArrayRef -> &RawRef -> &LayoutRef

    // We can also pass `arr` to functions that accept `RawRef` and `LayoutRef` via `AsRef`.
    // These alternative function signatures are important for other types, but we see that when
    // we have an `ArrayRef`, we can call them very simply.
    takes_rawref_asref(arr); // &ArrayRef -> &RawRef
    takes_layout_asref(arr); // &ArrayRef -> &LayoutRef
}

/// Now we want any array whose data is safe to mutate.
///
/// Importantly, any array passed to this function is guaranteed to uniquely point to its data.
/// As a result, passing a shared array to this function will silently un-share the array.
/// So, ***users should only accept `&mut ArrayRef` when they want to mutate data***.
/// If they just want to mutate shape and strides, use `&mut LayoutRef` or `&AsMut<LayoutRef>`.
#[allow(dead_code)]
fn takes_arrref_mut<A, D>(arr: &mut ArrayRef<A, D>)
{
    // We can do everything we did with a `&ArrayRef`
    takes_arrref(arr);

    // Similarly, we can pass this to functions that accept mutable references
    // to our other array reference types. These first two happen via `Deref`...
    takes_rawref_mut(arr);
    takes_layout_mut(arr);

    // ... and these two happen via `AsRef`.
    takes_rawref_asmut(arr);
    takes_rawref_asmut(arr);
}

/// Now let's go back and look at the way to write functions prior to 0.17: using `ArrayBase`.
///
/// This function signature says three things:
/// 1. Let me take a read only reference (that's the `&`)
/// 2. Of an array whose data is safe to dereference (that's the `S: Data`)
/// 3. And whose data is read-only (also `S: Data`)
///
/// Let's see what we can do with this array:
#[allow(dead_code)]
fn takes_base<S: Data, D>(arr: &ArrayBase<S, D>)
{
    // First off: we can pass it to functions that accept `&ArrayRef`.
    //
    // This is always "cheap", in the sense that even if `arr` is an
    // `ArcArray` that shares its data, using this call will not un-share that data.
    takes_arrref(arr);

    // We can also pass it to functions that accept `RawRef` and `LayoutRef`
    // in the usual two ways:
    takes_rawref(arr);
    takes_layout(arr);
    //
    takes_rawref_asref(&arr);
    takes_layout_asref(&arr);
}

/// Now, let's take a mutable reference to an `ArrayBase` - but let's keep `S: Data`, such
/// that we are allowed to change the _layout_ of the array, but not its data.
fn takes_base_mut<S: Data, D>(arr: &mut ArrayBase<S, D>)
{
    // Of course we can call everything we did with a immutable reference:
    takes_base(arr);

    // However, we _can't_ call a function that takes `&mut ArrayRef`:
    // this would require mutable data access, which `S: Data` does not provide.
    //
    // takes_arrref_mut(arr);
    // rustc: cannot borrow data in dereference of `ArrayBase<S, D>` as mutable
    //
    // Nor can we call a function that takes `&mut RawRef`
    // takes_rawref_mut(arr);

    // We can, however, call functions that take `AsMut<LayoutRef>`,
    // since `LayoutRef` does not provide read access to the data:
    takes_layout_mut(arr.as_layout_ref_mut());
    //
    takes_layout_asmut(arr);
}

/// Finally, let's look at a mutable reference to an `ArrayBase` with `S: DataMut`.
///
/// Note that we require a constraint of `D: Dimension` to dereference to `&mut ArrayRef`.
#[allow(dead_code)]
fn takes_base_data_mut<S: DataMut, D: Dimension>(arr: &mut ArrayBase<S, D>)
{
    // Of course, everything we can do with just `S: Data`:
    takes_base_mut(arr);

    // But also, we can now call functions that take `&mut ArrayRef`.
    //
    // NOTE: If `arr` is actually an `ArcArray` with shared data, this
    // will un-share the data. This can be a potentially costly operation.
    takes_arrref_mut(arr);
}

/// Let's now look at writing functions for the new `LayoutRef` type. We'll do this for both
/// immutable and mutable references, and we'll see how there are two different ways to accept
/// these types.
///
/// These functions can only read/modify an array's shape or strides,
/// such as checking dimensionality or slicing, should take `LayoutRef`.
///
/// Our first way is to accept an immutable reference to `LayoutRef`:
#[allow(dead_code)]
fn takes_layout<A, D>(_arr: &LayoutRef<A, D>) {}

/// We can also directly take a mutable reference to `LayoutRef`.
#[allow(dead_code)]
fn takes_layout_mut<A, D>(_arr: &mut LayoutRef<A, D>) {}

/// However, the preferred way to write these functions is by accepting
/// generics using `AsRef`.
///
/// For immutable access, writing with `AsRef` has the same benefit as usual:
/// callers have nicer ergonomics, since they can just pass any type
/// without having to call `.as_ref` or `.as_layout_ref`.
#[allow(dead_code)]
fn takes_layout_asref<T, A, D>(_arr: &T)
where T: AsRef<LayoutRef<A, D>> + ?Sized
{
}

/// For mutable access, there is an additional reason to write with `AsMut`:
/// it prevents callers who are passing in `ArcArray` or other shared array types
/// from accidentally unsharing the data through a deref chain:
/// `&mut ArcArray --(unshare)--> &mut ArrayRef -> &mut RawRef -> &mut LayoutRef`.
#[allow(dead_code)]
fn takes_layout_asmut<T, A, D>(_arr: &mut T)
where T: AsMut<LayoutRef<A, D>> + ?Sized
{
}

/// Finally, we have `RawRef`, where we can access and mutate the array's data, but only unsafely.
/// This is important for, e.g., dealing with [`std::mem::MaybeUninit`].
///
/// This is probably the rarest type to deal with, since `LayoutRef` can access and modify an array's
/// shape and strides, and even do in-place slicing. As a result, `RawRef` is only for functionality
/// that requires unsafe data access, something that `LayoutRef` can't do.
///
/// Like `LayoutRef`, writing functions with `RawRef` can be done in a few ways.
/// We start with a direct, immutable reference
#[allow(dead_code)]
fn takes_rawref<A, D>(arr: &RawRef<A, D>)
{
    takes_layout(arr);
    takes_layout_asref(arr);
}

/// We can also directly take a mutable reference.
#[allow(dead_code)]
fn takes_rawref_mut<A, D>(arr: &mut RawRef<A, D>)
{
    takes_layout(arr);
    takes_layout_asmut(arr);
}

/// However, like before, the preferred way is to write with `AsRef`,
/// for the same reasons as for `LayoutRef`:
#[allow(dead_code)]
fn takes_rawref_asref<T, A, D>(_arr: &T)
where T: AsRef<RawRef<A, D>> + ?Sized
{
    takes_layout(_arr.as_ref());
    takes_layout_asref(_arr.as_ref());
}

/// Finally, mutably:
#[allow(dead_code)]
fn takes_rawref_asmut<T, A, D>(_arr: &mut T)
where T: AsMut<RawRef<A, D>> + ?Sized
{
    takes_layout_mut(_arr.as_mut());
    takes_layout_asmut(_arr.as_mut());
}

fn main() {}
