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

- Parallel `Zip`
- Parallel elementwise (no order) iterator
- Parallel `.axis_iter()` (and `_mut`)
- `.par_map_inplace()` for arrays.

Status and Lookout
------------------

- Still iterating on and evolving the crate

  + A separate crate is less convenient (doesn't use rayon IntoParallelIterator
    trait, but a separate trait) but allows rapid iteration and we can follow
    the evolution of rayon's internals.
    This crate is double pace: For every ndarray or rayon major version, this
    crate goes up one major version.

- Performance:

  + TBD. Tell me about your experience.
  + You'll need a big chunk of data (or an expensive operation per data point)
    to gain from parallelization.

How to use with cargo::

    [dependencies]
    ndarray-parallel = "0.3"

Recent Changes (ndarray-parallel)
---------------------------------

- 0.3.0

  - ParallelIterator for Zip, including ``.par_apply``.
  - ``.par_map_inplace`` and ``.par_mav_inplace`` for arrays
  - Require ndarray 0.9 and rayon 0.7
  - Fix bug with array ``par_iter()``'s ``.collect()``

- 0.2.0

  - Require for ndarray 0.8

- 0.1.1

  - Clarify docs
  - Add categories

- 0.1.0

  - Initial release
  - Elementwise parallel iterator (no order) and parallel axis iterators.

License
=======

Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0 or the MIT license
http://opensource.org/licenses/MIT, at your
option. This file may not be copied, modified, or distributed
except according to those terms.


