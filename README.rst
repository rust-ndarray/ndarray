ndarray
=========

The ``ndarray`` crate provides an N-dimensional container for general elements
and for numerics.  Requires Rust 1.13.

Please read the API documentation here: `(0.8)`__, `(0.7)`__, `(0.6)`__,
`(0.5)`__, `(0.4)`__, `(0.3)`__, `(0.2)`__

__ http://bluss.github.io/rust-ndarray/
__ http://bluss.github.io/rust-ndarray/0.7/
__ http://bluss.github.io/rust-ndarray/0.6/
__ http://bluss.github.io/rust-ndarray/0.5/
__ http://bluss.github.io/rust-ndarray/0.4/
__ http://bluss.github.io/rust-ndarray/0.3/
__ http://bluss.github.io/rust-ndarray/0.2/

|build_status|_ |crates|_

.. |build_status| image:: https://travis-ci.org/bluss/rust-ndarray.svg?branch=master
.. _build_status: https://travis-ci.org/bluss/rust-ndarray

.. |crates| image:: http://meritbadge.herokuapp.com/ndarray
.. _crates: https://crates.io/crates/ndarray

Highlights
----------

- Generic N-dimensional array
- Owned arrays and array views
- Slicing, also with arbitrary step size, and negative indices to mean
  elements from the end of the axis.
- Views and subviews of arrays; iterators that yield subviews.

Status and Lookout
------------------

- Still iterating on and evolving the crate

  + The crate is continuously developing, and breaking changes are expected
    during evolution from version to version. We adopt the newest stable
    rust features if we need them.

- Performance:

  + Prefer higher order methods and arithmetic operations on arrays first,
    then iteration, and as a last priority using indexed algorithms.
  + Efficient floating point matrix multiplication even for very large
    matrices; can optionally use BLAS to improve it further.

Crate Feature Flags
-------------------

The following crate feature flags are available. They are configured in
your `Cargo.toml`.

- ``rustc-serialize``

  - Optional, compatible with Rust stable
  - Enables serialization support for rustc-serialize 0.3

- ``serde``

  - Optional, compatible with Rust stable
  - Enables serialization support for serde 0.9

- ``blas``

  - Optional and experimental, compatible with Rust stable
  - Enable transparent BLAS support for matrix multiplication.
    Uses ``blas-sys`` for pluggable backend, which needs to be configured
    separately.

How to use with cargo::

    [dependencies]
    ndarray = "0.8"

Recent Changes (ndarray)
------------------------

- 0.8

  - Update serde dependency to 0.9
  - Remove deprecated type alias ``OwnedArray`` (use ``Array``)
  - Remove deprecated ``.assign_scalar()`` (use ``fill``)

- 0.7.3

  - Add macro `array![]` for creating one-, two-, or three-dimensional arrays
    (with ownership semantics like `vec![]`)
  - `Array` now implements `Clone::clone_from()` specifically, so that its
    allocation is (possibly) reused.
  - Add `.to_vec()` for one-dimensional arrays
  - Add `RcArray::into_owned(self) -> Array`.
  - Add crate categories

- 0.7.2

  - Add array methods ``.remove_axis()``, ``.merge_axes()`` and ``.invert_axis()``
  - Rename ``Axis``’ accessor ``axis`` to ``index``, old name is deprecated.

- 0.7.1

  - Fix two bugs in ``Array::clone()``; it did not support zero-size elements
    like ``()``, and for some negatively strided arrays it did not update the
    first element offset correctly.
  - Add ``.axes()`` which is an iterator over the axes of an array, yielding
    its index, length and stride.
  - Add method ``.max_stride_axis()``.

- 0.6.10

  - Fix two bugs in ``Array::clone()``; it did not support zero-size elements
    like ``()``, and for some negatively strided arrays it did not update the
    first element offset correctly.

- 0.7.0

  - Big overhaul of dimensions: Add type ``Dim`` with aliases
    ``Ix1, Ix2, Ix3, ...`` etc for specific dimensionalities.
    Instead of ``Ix`` for dimension use ``Ix1``, instead of ``(Ix, Ix)`` use
    ``Ix2``, and so on.
  - The dimension type ``Dim`` supports indexing and arithmetic. See
    ``Dimension`` trait for new methods and inherited traits.
  - Constructors and methods that take tuples for array sizes, like ``Array::zeros,``
    ``Array::from_shape_vec``, ``.into_shape()`` and so on will continue to work
    with tuples.
  - The array method ``.raw_dim()`` returns the shape description
    ``D`` as it is. ``.dim()`` continues to return the dimension as a tuple.
  - Renamed iterators for consistency (each iterator is named for the
    method that creates it, for example ``.iter()`` returns ``Iter``).
  - The index iterator is now created with free functions ``indices`` or
    ``indices_of``.
  - Expanded the ``ndarray::prelude`` module with the dimensionality-specific
    type aliases, and some other items
  - ``LinalgScalar`` and related features no longer need to use ``Any`` for
    static type dispatch.
  - Serialization with ``serde`` now supports binary encoders like bincode
    and others.
  - ``.assign_scalar()`` was deprecated and replaced by ``.fill()``, which
    takes an element by value.
  - Require Rust 1.13

- 0.6.9

  - Implement ``ExactSizeIterator`` for the indexed iterators

- 0.6.8

  - Fix a bug in a partially consumed elements iterator's ``.fold()``.
    (**Note** that users are recommended to not use the elements iterator,
    but the higher level functions which are the maps, folds and other methods
    of the array types themselves.)

- 0.6.7

  - Improve performance of a lot of basic operations for arrays where
    the innermost dimension is not contiguous (``.fold(), .map(),
    .to_owned()``, arithmetic operations with scalars).
  - Require Rust 1.11

- 0.6.6

  - Add dimensionality specific type aliases: ``Array0, Array1, Array2, ...``
    and so on (there are many), also ``Ix0, Ix1, Ix2, ...``.
  - Add constructor ``Array::from_shape_fn(D, |D| -> A)``.
  - Improve performance of ``Array::default``, and ``.fold()`` for noncontiguous
    array iterators.

- 0.6.5

  - Add method ``.into_raw_vec()`` to turn an ``Array`` into the its
    underlying element storage vector, in whatever element order it is using.

- 0.6.4

  - Add method ``.map_axis()`` which is used to flatten an array along
    one axis by mapping it to a scalar.

- 0.6.3

  - Work around compilation issues in nightly (issue #217)
  - Add ``Default`` implementations for owned arrays

- 0.6.2

  - Add serialization support for serde 0.8, under the crate feature name ``serde``

- 0.6.1

  - Add ``unsafe`` array view constructors ``ArrayView::from_shape_ptr``
    for read-only and read-write array views. These make it easier to
    create views from raw pointers.

- 0.6.0

  - Rename ``OwnedArray`` to ``Array``. The old name is deprecated.
  - Remove deprecated constructor methods. Use zeros, from_elem, from_shape_vec
    or from_shape_vec_unchecked instead.
  - Remove deprecated in place arithmetic methods like iadd et.c. Use += et.c.
    instead.
  - Remove deprecated method mat_mul, use dot instead.
  - Require Rust 1.9

- 0.5.2

  - Use num-traits, num-complex instead of num.

- 0.5.1

  - Fix theoretical well-formedness issue with Data trait

- 0.5.0

  - Require Rust 1.8 and enable +=, -=, and the other assign operators.
    All ``iadd, iadd_scalar`` and similar methods are now deprecated.
  - ndarray now has a prelude: ``use ndarray::prelude::*;``.
  - Constructors from_elem, zeros, from_shape_vec now all support passing a custom
    memory layout. A lot of specific constructors were deprecated.
  - Add method ``.select(Axis, &[Ix]) -> OwnedArray``, to create an array
    from a non-contiguous pick of subviews along an axis.
  - Rename ``.mat_mul()`` to just ``.dot()`` and add a function ``general_mat_mul``
    for matrix multiplication with scaling into an existing array.
  - **Change .fold() to use arbitrary order.**
  - See below for more details

- 0.5.0-alpha.2

  - Fix a namespace bug in the stack![] macro.
  - Add method .select() that can pick an arbitrary set of rows (for example)
    into a new array.

- 0.4.9

  - Fix a namespace bug in the stack![] macro.
  - Add deprecation messages to .iadd() and similar methods (use += instead).

- 0.5.0-alpha.1

  - Add .swap(i, j) for swapping two elements
  - Add a prelude module ``use ndarray::prelude::*;``
  - Add ndarray::linalg::general_mat_mul which computes *C ← α A B + β C*,
    i.e matrix multiplication into an existing array, with optional scaling.
  - Add .fold_axis(Axis, folder)
  - Implement .into_shape() for f-order arrays

- 0.5.0-alpha.0

  - Requires Rust 1.8. Compound assignment operators are now enabled by default.
  - Rename ``.mat_mul()`` to ``.dot()``. The same method name now handles
    dot product and matrix multiplication.
  - Remove deprecated items: raw_data, raw_data_mut, allclose, zeros, Array.
    Docs for 0.4. lists the replacements.
  - Remove deprecated crate features: rblas, assign_ops
  - A few consuming arithmetic ops with ArrayViewMut were removed (this
    was missed in the last version).
  - **Change .fold() to use arbitrary order.** Its specification and
    implementation has changed, to pick the most appropriate element traversal
    order depending on memory layout.

- 0.4.8

  - Fix an error in ``.dot()`` when using BLAS and arrays with negative stride.

- 0.4.7

  - Add dependency matrixmultiply to handle matrix multiplication
    for floating point elements. It supports matrices of general stride
    and is a great improvement for performance. See PR #175.

- 0.4.6

  - Fix bug with crate feature blas; it would not compute matrix
    multiplication correctly for arrays with negative or zero stride.
  - Update blas-sys version (optional dependency).

- 0.4.5

  - Add ``.all_close()`` which replaces the now deprecated ``.allclose()``.
    The new method has a stricter protocol: it panics if the array
    shapes are not compatible. We don't want errors to pass silently.
  - Add a new illustration to the doc for ``.axis_iter()``.
  - Rename ``OuterIter, OuterIterMut`` to ``AxisIter, AxisIterMut``.
    The old name is now deprecated.

- 0.4.4

  - Add mapping methods ``.mapv(), .mapv_into(), .map_inplace(),``
    ``.mapv_inplace(), .visit()``. The ``mapv`` versions
    have the transformation function receive the element by value (hence *v*).
  - Add method ``.scaled_add()`` (a.k.a axpy) and constructor ``from_vec_dim_f``.
  - Add 2d array methods ``.rows(), .cols()``.
  - Deprecate method ``.fold()`` because it dictates a specific visit order.

- 0.4.3

  - Add array method ``.t()`` as a shorthand to create a transposed view.
  - Fix ``mat_mul`` so that it accepts arguments of different array kind
  - Fix a bug in ``mat_mul`` when using BLAS and multiplying with a column
    matrix (#154)

- 0.4.2

  - Add new BLAS integration used by matrix multiplication
    (selected with crate feature ``blas``). Uses pluggable backend.
  - Deprecate module ``ndarray::blas`` and crate feature ``rblas``. This module
    was moved to the crate ``ndarray-rblas``.
  - Add array methods ``as_slice_memory_order, as_slice_memory_order_mut, as_ptr,
    as_mut_ptr``.
  - Deprecate ``raw_data, raw_data_mut``.
  - Add ``Send + Sync`` to ``NdFloat``.
  - Arrays now show shape & stride in their debug formatter.
  - Fix a bug where ``from_vec_dim_stride`` did not accept arrays with unitary axes.
  - Performance improvements for contiguous arrays in non-c order when using
    methods ``to_owned, map, scalar_sum, assign_scalar``,
    and arithmetic operations between array and scalar.
  - Some methods now return arrays in the same memory order of the input
    if the input is contiguous: ``to_owned, map, mat_mul`` (matrix multiplication
    only if both inputs are the same memory order), and arithmetic operations
    that allocate a new result.
  - Slight performance improvements in ``dot, mat_mul`` due to more efficient
    glue code for calling BLAS.
  - Performance improvements in ``.assign_scalar``.

- 0.4.1

  - Mark iterators ``Send + Sync`` when possible.

- **0.4.0** `Release Announcement`__

  - New array splitting via ``.split_at(Axis, Ix)`` and ``.axis_chunks_iter()``
  - Added traits ``NdFloat``, ``AsArray`` and ``From for ArrayView`` which
    improve generic programming.
  - Array constructors panic when attempting to create an array whose element
    count overflows ``usize``. (Would be a debug assertion for overflow before.)
  - Performance improvements for ``.map()``.
  - Added ``stack`` and macro ``stack![axis, arrays..]`` to concatenate arrays.
  - Added constructor ``OwnedArray::range(start, end, step)``.
  - The type alias ``Array`` was renamed to ``RcArray`` (and the old name deprecated).
  - Binary operators are not defined when consuming a mutable array view as
    the left hand side argument anymore.
  - Remove methods and items deprecated since 0.3 or earlier; deprecated methods
    have notes about replacements in 0.3 docs.
  - See below for full changelog through alphas.

__ http://bluss.github.io/rust/2016/03/06/ndarray-0.4/

- 0.4.0-alpha.8

  - In debug mode, indexing an array out of bounds now has a detailed
    message about index and shape. (In release mode it does not.)
  - Enable assign_ops feature automatically when it is supported (Rust 1.8 beta
    or later).
  - Add trait ``NdFloat`` which makes it easy to be generic over ``f32, f64``.
  - Add ``From`` implementations that convert slices or references to arrays
    into array views. This replaces ``from_slice`` from a previous alpha.
  - Add ``AsArray`` trait, which is simply based on those ``From`` implementations.
  - Improve ``.map()`` so that it can autovectorize.
  - Use ``Axis`` argument in ``RemoveAxis`` too.
  - Require ``DataOwned`` in the raw data methods.
  - Merged error types into a single ``ShapeError``, which uses no allocated data.

- 0.4.0-alpha.7

  - Fix too strict lifetime bound in arithmetic operations like ``&a @ &b``.
  - Rename trait Scalar to ScalarOperand (and improve its docs).
  - Implement <<= and >>= for arrays.

- 0.4.0-alpha.6

  - All axis arguments must now be wrapped in newtype ``Axis``.
  - Add method ``.split_at(Axis, Ix)`` to read-only and read-write array views.
  - Add constructors ``ArrayView{,Mut}::from_slice`` and array view methods
    are now visible in the docs.

- 0.4.0-alpha.5

  - Use new trait ``LinalgScalar`` for operations where we want type-based specialization.
    This shrinks the set of types that allow dot product, matrix multiply, mean.
  - Use BLAS acceleration transparently in ``.dot()`` (this is the first step).
  - Only OwnedArray and RcArray and not ArrayViewMut can now be used as consumed
    left hand operand for arithmetic operators. `See arithmetic operations docs!`__
  - Remove deprecated module ``linalg`` (it was already mostly empty)
  - Deprecate free function ``zeros`` in favour of static method ``zeros``.

__ https://bluss.github.io/rust-ndarray/master/ndarray/struct.ArrayBase.html#arithmetic-operations

- 0.4.0-alpha.4

  - Rename ``Array`` to ``RcArray``. Old name is deprecated.
  - Add methods ``OuterIter::split_at``, ``OuterIterMut::split_at``
  - Change ``arr0, arr1, arr2, arr3`` to return ``OwnedArray``.
    Add ``rcarr1, rcarr2, rcarr3`` that return ``RcArray``.

- 0.4.0-alpha.3

  - Improve arithmetic operations where the RHS is a broadcast 0-dimensional
    array.
  - Add read-only and read-write array views to the ``rblas`` integration.
    Added methods ``AsBlas::{blas_view_checked, blas_view_mut_checked, bv, bvm}``.
  - Use hash_slice in ``Hash`` impl for arrays.

- 0.4.0-alpha.2

  - Add ``ArrayBase::reversed_axes`` which transposes an array.

- 0.4.0-alpha.1

  - Add checked and unchecked constructor methods for creating arrays
    from a vector and explicit dimension and stride, or with
    fortran (column major) memory order (marked ``f``):
    
    + ``ArrayBase::from_vec_dim``, ``from_vec_dim_stride``,
      ``from_vec_dim_stride_unchecked``,
    + ``from_vec_dim_unchecked_f``, ``from_elem_f``, ``zeros_f``
    + View constructors ``ArrayView::from_slice_dim_stride``,
      ``ArrayViewMut::from_slice_dim_stride``.
    + Rename old ``ArrayBase::from_vec_dim`` to ``from_vec_dim_unchecked``.

  - Check better for wraparound when computing the number of elements in a shape;
    this adds error cases that **panic** in ``from_elem``, ``zeros`` etc,
    however *the new check will only ever panic in cases that would
    trigger debug assertions for overflow in the previous versions*!.
  - Add an array chunks iterator ``.axis_chunks_iter()`` and mutable version;
    it allows traversing the array in for example chunks of *n* rows at a time.
  - Remove methods and items deprecated since 0.3 or earlier; deprecated methods
    have notes about replacements in 0.3 docs.

- 0.3.1

  - Add ``.row_mut()``, ``.column_mut()``
  - Add ``.axis_iter()``, ``.axis_iter_mut()``

- **0.3.0**

  - Second round of API & consistency update is done
  - 0.3.0 highlight: **Index type** ``Ix`` **changed to** ``usize``.
  - 0.3.0 highlight: Operator overloading for scalar and array arithmetic.
  - 0.3.0 highlight: Indexing with ``a[[i, j, k]]`` syntax.
  - Add ``ArrayBase::eye(n)``
  - See below for more info

- 0.3.0-alpha.4

  - Shrink array view structs by removing their redundant slice field (see #45).
    Changed the definition of the view ``type`` aliases.
  - ``.mat_mul()`` and ``.mat_mul_col()`` now return ``OwnedArray``.
    Use ``.into_shared()`` if you need an ``Array``.
  - impl ExactSizeIterator where possible for iterators.
  - impl DoubleEndedIterator for ``.outer_iter()`` (and _mut).

- 0.3.0-alpha.3

  - ``.subview()`` changed to return an array view, also added ``into_subview()``.
  - Add ``.outer_iter()`` and ``.outer_iter_mut()`` for iteration along the
    greatest axis of the array. Views also implement ``into_outer_iter()`` for
    “lifetime preserving” iterators.

- 0.3.0-alpha.2

  - Improve the strided last dimension case in ``zip_mut_with`` slightly
    (affects all binary operations).
  - Add ``.row(i), .column(i)`` for 2D arrays.
  - Deprecate ``.row_iter(), .col_iter()``.
  - Add method ``.dot()`` for computing the dot product between two 1D arrays.


- 0.3.0-alpha.1

  - **Index type** ``Ix`` **changed to** ``usize`` (#9). Gives better iterator codegen
    and 64-bit size arrays.
  - Support scalar operands with arithmetic operators.
  - Change ``.slice()`` and ``.diag()`` to return array views, add ``.into_diag()``.
  - Add ability to use fixed size arrays for array indexing, enabling syntax
    like ``a[[i, j]]`` for indexing.
  - Add ``.ndim()``

- **0.2.0**

  - First chapter of API and performance evolution is done \\o/
  - 0.2.0 highlight: Vectorized (efficient) arithmetic operations
  - 0.2.0 highlight: Easier slicing using `s![]`
  - 0.2.0 highlight: Nicer API using views
  - 0.2.0 highlight: Bridging to BLAS functions.
  - See below for more info

- 0.2.0-alpha.9

  - Support strided matrices in ``rblas`` bridge, and fix a bug with
    non square matrices.
  - Deprecated all of module ``linalg``.

- 0.2.0-alpha.8

  - **Note:** PACKAGE NAME CHANGED TO ``ndarray``. Having package != crate ran
    into many quirks of various tools. Changing the package name is easier for
    everyone involved!
  - Optimized ``scalar_sum()`` so that it will vectorize for the floating point
    element case too.

- 0.2.0-alpha.7

  - Optimized arithmetic operations!

    - For c-contiguous arrays or arrays with c-contiguous lowest dimension
      they optimize very well, and can vectorize!

  - Add ``.inner_iter()``, ``.inner_iter_mut()``
  - Add ``.fold()``, ``.zip_mut_with()``
  - Add ``.scalar_sum()``
  - Add example ``examples/life.rs``

- 0.2.0-alpha.6

  - Add ``#[deprecated]`` attributes (enabled with new enough nightly)
  - Add ``ArrayBase::linspace``, deprecate constructor ``range``.

- 0.2.0-alpha.5

  - Add ``s![...]``, a slice argument macro.
  - Add ``aview_mut1()``, ``zeros()``
  - Add ``.diag_mut()`` and deprecate ``.diag_iter_mut()``, ``.sub_iter_mut()``
  - Add ``.uget()``, ``.uget_mut()`` for unchecked indexing and deprecate the
    old names.
  - Improve ``ArrayBase::from_elem``
  - Removed ``SliceRange``, replaced by ``From`` impls for ``Si``.

- 0.2.0-alpha.4

  - Slicing methods like ``.slice()`` now take a fixed size array of ``Si``
    as the slice description. This allows more type checking to verify that the
    number of axes is correct.
  - Add experimental ``rblas`` integration.
  - Add ``into_shape()`` which allows reshaping any array or view kind.

- 0.2.0-alpha.3

  - Add and edit a lot of documentation

- 0.2.0-alpha.2

  - Improve performance for iterators when the array data is in the default
    memory layout. The iterator then wraps the default slice iterator and
    loops will autovectorize.
  - Remove method ``.indexed()`` on iterators. Changed ``Indexed`` and added
    ``ÌndexedMut``.
  - Added ``.as_slice(), .as_mut_slice()``
  - Support rustc-serialize


- 0.2.0-alpha

  - Alpha release!
  - Introduce ``ArrayBase``, ``OwnedArray``, ``ArrayView``, ``ArrayViewMut``
  - All arithmetic operations should accept any array type
  - ``Array`` continues to refer to the default reference counted copy on write
    array
  - Add ``.view()``, ``.view_mut()``, ``.to_owned()``, ``.into_shared()``
  - Add ``.slice_mut()``, ``.subview_mut()``
  - Some operations now return ``OwnedArray``:

    - ``.map()``
    - ``.sum()``
    - ``.mean()``

  - Add ``get``, ``get_mut`` to replace the now deprecated ``at``, ``at_mut``.
  - Fix bug in assign_scalar

- 0.1.1

  - Add Array::default
  - Fix bug in raw_data_mut

- 0.1.0

  - First release on crates.io
  - Starting point for evolution to come

License
=======

Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0 or the MIT license
http://opensource.org/licenses/MIT, at your
option. This file may not be copied, modified, or distributed
except according to those terms.


