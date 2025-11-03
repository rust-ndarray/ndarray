Recent Changes
--------------

- 0.16.0

  - Require ndarray 0.17.1
  - Bump `rand` to 0.9.0 and `rand_distr` to 0.5.0
  - Add an additional extension trait, `RandomRefExt`, to allow sampling from `ndarray::ArrayRef` instances

- 0.15.0

  - Require ndarray 0.16
  - Remove deprecated F32 by [@bluss](https://github.com/bluss) [#1409](https://github.com/rust-ndarray/ndarray/pull/1409)

- 0.14.0

  - Require ndarray 0.15
  - Require rand 0.8 (unchanged from previous version)
  - The F32 wrapper is now deprecated, it's redundant

- 0.13.0

  - Require ndarray 0.14 (unchanged from previous version)
  - Require rand 0.8
  - Require rand_distr 0.4
  - Fix methods `sample_axis` and `sample_axis_using` so that they can be used on array views too.

- 0.12.0

  - Require ndarray 0.14
  - Require rand 0.7 (unchanged from previous version)
  - Require rand_distr 0.3

- 0.11.0

  - Require ndarray 0.13
  - Require rand 0.7 (unchanged from previous version)

- 0.10.0

  - Require `rand` 0.7
  - Require Rust 1.32 or later
  - Re-export `rand` as a submodule, `ndarray_rand::rand`
  - Re-export `rand-distr` as a submodule, `ndarray_rand::rand_distr`

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
