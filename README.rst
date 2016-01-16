ndarray
=========

The ``ndarray`` crate provides an N-dimensional container similar to numpy’s
ndarray. Requires Rust 1.5.

Please read the `API documentation here (master)`__, `(0.3)`__, `(0.2)`__

__ http://bluss.github.io/rust-ndarray/
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
- Owned arrays and views

  - ``ArrayBase``:
    The N-dimensional array type itself.
  - ``Array``:
    An array where the data is shared and copy on write, it
    can act as both an owner of the data as well as a lightweight view.
  - ``OwnedArray``:
    An array where the data is owned uniquely.
  - ``ArrayView``, ``ArrayViewMut``:
    Lightweight array views.

- Slicing, also with arbitrary step size, and negative indices to mean
  elements from the end of the axis.
- Iteration and most operations are efficient on arrays with contiguous
  innermost dimension.
- Array views can be used to slice and mutate any ``[T]`` data.

Status and Lookout
------------------

- Still iterating on the API
- Performance status:

  + Arithmetic involving arrays of contiguous inner dimension optimizes very well.
  + ``.fold()`` and ``.zip_mut_with()`` are the most efficient ways to
    perform single traversal and lock step traversal respectively.
  + ``.iter()`` and ``.iter_mut()`` are efficient for contiguous arrays.

- There is experimental bridging to the linear algebra package ``rblas``.

Crate Feature Flags
-------------------

- ``assign_ops``

  - Optional, requires nightly
  - Enables the compound assignment operators

- ``rustc-serialize``

  - Optional, stable
  - Enables serialization support

- ``rblas``

  - Optional, stable
  - Enables ``rblas`` integration

How to use with cargo::

    [dependencies]
    ndarray = "0.3"

Recent Changes
--------------

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


