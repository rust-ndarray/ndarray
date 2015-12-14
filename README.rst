rendarray
=========

An arbitrary dimension N-dimensional array container of elements of
arbitrary types.

Features

- Array views and slices, including lightweight transposition
- Broadcast array dimensions
- Good support for numerics, but lacking optimization.

Please read the `API documentation here`__

__ http://bluss.github.io/rust-ndarray/

|build_status|_

.. |build_status| image:: https://travis-ci.org/bluss/rust-ndarray.svg?branch=master
.. _build_status: https://travis-ci.org/bluss/rust-ndarray

How to use with cargo::

    [dependencies]
    rendarray = "0.1"

Recent Changes
--------------

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


