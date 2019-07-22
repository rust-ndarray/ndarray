ndarray-rand
============

Dependencies
------------

ndarray-rand depends on rand 0.7. If you use any other items from rand,
you need to specify a compatible version of rand in your Cargo.toml. If
you want to use a RNG or distribution from another crate with
ndarray-rand, you need to make sure that crate also depends on the
correct version of rand. Otherwise, the compiler will return errors
saying that the items are not compatible (e.g. that a type doesn't
implement a necessary trait).


Recent Changes
--------------

- 0.10.0

  - Require rand 0.7

- 0.9.0

  - Require rand 0.6

- 0.8.0

  - Require ndarray 0.12
  - Require rand 0.5

- 0.7.0

  - Require ndarray 0.11
  - Require rand 0.4

- 0.6.1

  - Clean up implementation of ``Array::random`` by @v-shmyhlo

- 0.6.0

  - Require ndarray 0.10.0

- 0.5.0

  - Require ndarray 0.9

- 0.4.0

  - Require ndarray 0.8

- 0.3.0

  - Require ndarray 0.7

- 0.2.0

  - Require ndarray 0.6

- 0.1.0

  - Initial release

License
=======

Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0 or the MIT license
http://opensource.org/licenses/MIT, at your
option. This file may not be copied, modified, or distributed
except according to those terms.


