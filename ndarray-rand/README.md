ndarray-rand
============

Constructors for randomized arrays: `rand`'s integration with `ndarray`.

Example
=======

Generate a 2-dimensional array with shape `(2,5)` and elements drawn from a uniform distribution
over the `(0., 10.)` interval:

```rust
use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

fn main() {
    let a = Array::random((2, 5), Uniform::new(0., 10.));
    println!("{:8.4}", a);
    // Example Output:
    // [[  8.6900,   6.9824,   3.8922,   6.5861,   2.4890],
    //  [  0.0914,   5.5186,   5.8135,   5.2361,   3.1879]]
}
```

Dependencies
============

``ndarray-rand`` depends on ``rand``.

[`rand`](https://docs.rs/rand/) and [`rand-distr`](https://docs.rs/rand_distr/) are 
re-exported as sub-modules, `ndarray_rand::rand` and `ndarray_rand::rand_distr` respectively. 
Please rely on these submodules for guaranteed version compatibility.

If you want to use a random number generator or distribution from another crate
with `ndarray-rand`, you need to make sure that the other crate also depends on the
same version of `rand`. Otherwise, the compiler may return errors saying
that the items are not compatible (e.g. that a type doesn't implement a
necessary trait).

Recent changes
==============

Check _[RELEASES.md](https://github.com/rust-ndarray/ndarray/blob/master/ndarray-rand/RELEASES.md)_ to see 
the changes introduced in previous releases.


License
=======

Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0 or the MIT license
http://opensource.org/licenses/MIT, at your
option. This file may not be copied, modified, or distributed
except according to those terms.
