% ndarray

<br><br>
presented by Ulrik Sverdrup (bluss)

22 March 2016

<small>https://github.com/bluss/rust-ndarray</small>

# What is ndarray?

+ A Rust library, compatible with stable Rust releases
+ Current version as of this talk is [`ndarray 0.4.4`][crate]

[crate]: https://crates.io/crates/ndarray/

# What is ndarray?

*A multidimensional container for general elements and for numerics*

It’s an array with multiple dimensions.

```rust
# extern crate ndarray;
use ndarray::OwnedArray;

# fn main() {
let mut array = OwnedArray::zeros((3, 5, 5));
array[[1, 1, 1]] = 9.;
# }
```


+ An in-memory data structure
+ Elements of arbitrary type


# What is ndarray?

It is bounds checked like a regular Rust data structure:

```ignore
array[[3, 1, 1]] = 10.;
// PANIC! thread '<main>' panicked at 'ndarray: index [3, 1, 1] is out of bounds for array of shape [3, 5, 5]'
```

and it supports numerics:

```rust
# extern crate ndarray;
# use ndarray::OwnedArray;
# fn main() {
let x = OwnedArray::from_vec(vec![0., 1., 2., 1.]);
let x_hat = &x / x.scalar_sum();
println!("{:5.2}", x_hat);
// OUTPUT: [ 0.00,  0.25,  0.50,  0.25]
# }
```


# Array Types

+ A unique-owner array called `OwnedArray<A, D>`
  + Where `A` is the element type, and `D` is the dimensionality
+ Array views:
  + `ArrayView<A, D>`
  + `ArrayViewMut<A, D>`
+ Like `Vec<A>` and its view types `&[A]` and `&mut [A]`.
+ Views allow composable algorithms; access to arrays or parts of them
  is the same.
+ Views allow efficient and rustic expression of
  divide and conquer algorithms.

# Representation

<div style="text-align: center; margin-top: 10px">
<img src="rep-matrix.svg" height="600">
</div>

# Representation

<div style="text-align: center; margin-top: 10px">
<img src="rep-view.svg" height="600">
</div>

# Representation

<div style="text-align: center; margin-top: 10px">
<img src="rep-stride.svg" height="600">
</div>

# Representation

<div style="text-align: center; margin-top: 10px">
<img src="rep-ndarray.svg" height="600">
</div>

# Slicing

<div style="text-align: center; margin-top: 10px">
<img src="rep-slice.svg" height="600">
</div>


# Splitting

<div style="text-align: center; margin-top: 10px">
<img src="split_at.svg" height="600">
</div>

# Views and Iterators


```rust,ignore
.inner_iter()     // each iterator has a corresponding
.outer_iter()     // mutable version too.
.axis_iter(Axis)
.axis_chunks_iter(Axis, usize)
```
<div style="text-align: center; margin: 0px">
<img src="axis_iter.svg" height="350">
</div>

Iterators are a powerful way to access views of an array.

# An Array View of Anything

+ You don’t need an `OwnedArray`, you can create an array view of any data you
  can get a slice of.

```rust
# extern crate ndarray;
# use ndarray::ArrayViewMut;
# fn main() {
// Create a stack allocated Hilbert Matrix 
let mut data = [0.; 1024];
let mut view = ArrayViewMut::from(&mut data[..])
                            .into_shape((32, 32)).unwrap();
for ((i, j), elt) in view.indexed_iter_mut() {
    *elt = 1. / (1. + i as f32 + j as f32);
}
# }
```


# Higher Order Functions

```rust,ignore
// Closure types in pseudocode!
.map(&self, |&A| -> B) -> OwnedArray<B, D>
.mapv(&self, |A| -> B) -> OwnedArray<B, D>
.map_inplace(&mut self, |&mut A|)
.mapv_inplace(&mut self, |A| -> A)

.zip_mut_with(&mut self, rhs: &Array<B>, |&mut A, &B|)
```

Higher order functions are a powerful way to traverse / modify an array element
by element. They give ndarray flexibility to perform the operation
as efficiently as possible.


# Performance

These operations are efficient in ndarray. Our Rust code is autovectorized
by the compiler.

```rust
# extern crate ndarray;
# use ndarray::OwnedArray;
# fn main() {
# let mut array1 = OwnedArray::zeros((3, 5, 5));
# let array2 = OwnedArray::zeros((3, 5, 5));
/* Unary op */  array1 += 1.;
/* Unary op */  array1.mapv_inplace(f32::abs);
/* Binary op */ array1 += &array2;
/* Reduction */ array1.scalar_sum();
# }
```

(Overloading `+=` is a Rust 1.8 feature<br> — stable soon! 
)

Matrix multiplication and linear algebra is another story, uses integration
with BLAS.

# Performance

These operations are efficient in ndarray. Our Rust code is autovectorized
by the compiler.

```rust
# extern crate ndarray;
# use ndarray::OwnedArray;
# fn main() {
# let mut array1 = OwnedArray::zeros((3, 5, 5));
# let array2 = OwnedArray::zeros((3, 5, 5));
/* Unary op */  array1 += 1.;
/* Unary op */  array1.mapv_inplace(f32::abs);
/* Binary op */ array1 += &array2;
/* Reduction */ array1.scalar_sum();
# }
```

(Overloading `+=` is a Rust 1.8 feature<br> — stable in 
 <em id="stableclock"></em>)

Matrix multiplication and linear algebra is another story, uses integration
with BLAS.

# Performance Secret 1

# Performance Secret 1

+ `&[T]`. A contiguous slice of data.

# Performance Secret 1

+ `&[T]`. A contiguous slice of data.

Ndarray operations are efficient when they access the underlying data as a slice.

```rust
fn unary_operation(data: &mut [f32]) {
    for element in data {
        *element += 1.;
    }
}
```

# Performance Secret 2

Iterate two slices in lock step.

```rust
fn binary_operation(a: &mut [f32], b: &[f32]) {
    let len = std::cmp::min(a.len(), b.len());
    let a = &mut a[..len];
    let b = &b[..len];

    for i in 0..len {
        a[i] += b[i];
    }
}
```

# Performance Secret 3

Autovectorize a floating point sum.

*Ideally...*

```rust
fn sum(data: &[f32]) -> f32 {
    let mut sum = 0.;
    for &element in data {
        sum += element;
    }
    sum
}
```

# Performance Secret 3

Autovectorize a floating point sum.

```rust
fn sum(mut data: &[f32]) -> f32 {
    let (mut s0, mut s1, mut s2, mut s3,
         mut s4, mut s5, mut s6, mut s7) =
        (0., 0., 0., 0., 0., 0., 0., 0.);

    while data.len() >= 8 {
        s0 += data[0]; s1 += data[1];
        s2 += data[2]; s3 += data[3];
        s4 += data[4]; s5 += data[5];
        s6 += data[6]; s7 += data[7];
        data = &data[8..];
    }
    let mut sum = 0.;
    sum += s0 + s4; sum += s1 + s5;
    sum += s2 + s6; sum += s3 + s7;
    for i in 0..data.len() {
        sum += data[i];
    }
    sum
}
# assert_eq!(sum(&[1.; 33]), 33.);
```

# Performance Secrets

+ `&[T]`. A contiguous slice of data.

Ndarray operations are efficient when they access the underlying data as a slice
(they do when the memory layout allows).

+ If the array is contiguous, use it as a slice.
+ If the array has contiguous rows or columns, iterate over their slices.
+ Unary transformations don’t care which order elements are accessed.
+ Efficient binary operations require that the array layouts line
  up to some extent.
+ Regular loops optimize with autovectorization if we’re careful.

# Design Choices

+ A type parameter for dimensionality
  + Statically distinguishes a 1D view from a 2D view and so on
  + The low dimensional views are lightweight
  + and easy to see through for the optimizing compiler
  + ...but we have no operations that dynamically change the number of axes of an array.
+ A signed stride per axis
  + Can represent different memory layouts
  + ...but some operations must have multiple cases depending on layout


# End

+ ndarray is focused on the data structure.
+ We want to do the basics right and make it easy to extend.
+ For more information, see our [github][gh] or the [ndarray documentation][doc].



[gh]: https://github.com/bluss/rust-ndarray
[doc]: https://bluss.github.io/rust-ndarray/

<!-- vim: sts=4 sw=4 et
-->
