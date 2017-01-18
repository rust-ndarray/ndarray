ndarray-parallel
================

``ndarray-parallel`` integrates ndarray with rayon__ for simple parallelization.

__ https://github.com/nikomatsakis/rayon
Please read the `API documentation here`__

__ http://docs.rs/ndarray-parallel/

|build_status|_ |crates|_

.. |build_status| image:: https://travis-ci.org/bluss/rust-ndarray.svg?branch=master
.. _build_status: https://travis-ci.org/bluss/rust-ndarray

.. |crates| image:: http://meritbadge.herokuapp.com/ndarray-parallel
.. _crates: https://crates.io/crates/ndarray-parallel

Highlights
----------

- Parallel elementwise (no order) and axis iterators

Status and Lookout
------------------

- Still iterating on and evolving the crate

  + A separate crate is less convenient (doesn't use rayon IntoParallelIterator
    trait, but a separate crate) but allows rapid iteration and we can follow
    the evolution of rayon's internals.
  + This crate is double pace: For every ndarray or rayon major version, this
    crate goes up one major version.

- Performance:

  + TBD. Tell me about your experience.
  + You'll need a big chunk of data (or an expensive operation per data point)
    to gain from parallelization.

How to use with cargo::

    [dependencies]
    ndarray-parallel = "0.1"

Recent Changes (ndarray-parallel)
---------------------------------

- *

  - Not yet released

License
=======

Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0 or the MIT license
http://opensource.org/licenses/MIT, at your
option. This file may not be copied, modified, or distributed
except according to those terms.


