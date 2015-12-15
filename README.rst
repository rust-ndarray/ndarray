rendarray
=========

An arbitrary dimension N-dimensional array container of elements of
arbitrary types.

Features

- ``Array`` for a reference counted copy on write array
- ``OwnedArray`` for a uniquely owned array
- Array views and slices, including lightweight transposition
- Broadcast array dimensions
- Good support for numerics, but lacking optimization.

Please read the `API documentation here`__

__ http://bluss.github.io/rust-ndarray/

|build_status|_ |crates|_

.. |build_status| image:: https://travis-ci.org/bluss/rust-ndarray.svg?branch=master
.. _build_status: https://travis-ci.org/bluss/rust-ndarray

.. |crates| image:: http://meritbadge.herokuapp.com/rendarray
.. _crates: https://crates.io/crates/rendarray

How to use with cargo::

    [dependencies]
    rendarray = "0.1"

Recent Changes
--------------

- 0.2.0-alpha

  - Alpha release!
  - Introduce ``ArrayBase``, ``OwnedArray``, ``ArrayView``, ``ArrayViewMut``
  - All arithmetic operations should accept any array type
  - ``Array`` continues to refer to the default reference counted copy on write
    array
  - Some operations now return ``OwnedArray``:

    - ``.map()``
    - ``.sum()``
    - ``.mean()``

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


