rendarray
=========

The ``ndarray`` crate provides an N-dimensional container similar to numpy’s
ndarray.

Please read the `API documentation here`__

__ http://bluss.github.io/rust-ndarray/


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
- Iteration and most operations are very efficient on contiguous c-order arrays
  (the default layout, without any transposition or discontiguous subslicing).
  and on arrays where the lowest dimension is contiguous.
- Array views can be used to slice and mutate any ``[T]`` data.

Status and Lookout
------------------

- Still iterating on the API
- Performance status:

  + Arithmetic involving contiguous c-order arrays and contiguous lowest
    dimension arrays optimizes very well.
  + `.fold()` and `.zip_with_mut()` are the most efficient ways to
    perform single traversal and lock step traversal respectively.
  + Transposed arrays where the lowest dimension is not c-contiguous
    is still a pain point.

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

|build_status|_ |crates|_

.. |build_status| image:: https://travis-ci.org/bluss/rust-ndarray.svg?branch=master
.. _build_status: https://travis-ci.org/bluss/rust-ndarray

.. |crates| image:: http://meritbadge.herokuapp.com/rendarray
.. _crates: https://crates.io/crates/rendarray

How to use with cargo::

    [dependencies]
    rendarray = "0.2"

Recent Changes
--------------

- **Note:** At some point in a future release, the indexing type ``Ix`` will
  change to ``usize``

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


