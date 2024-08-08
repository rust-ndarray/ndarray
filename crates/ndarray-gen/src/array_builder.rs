// Copyright 2024 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ndarray::Array;
use ndarray::Dimension;
use ndarray::IntoDimension;
use ndarray::Order;

use num_traits::Num;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ArrayBuilder<D: Dimension>
{
    dim: D,
    memory_order: Order,
    generator: ElementGenerator,
}

/// How to generate elements
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ElementGenerator
{
    Sequential,
    Zero,
}

impl<D: Dimension> Default for ArrayBuilder<D>
{
    fn default() -> Self
    {
        Self::new(D::zeros(D::NDIM.unwrap_or(1)))
    }
}

impl<D> ArrayBuilder<D>
where D: Dimension
{
    pub fn new(dim: impl IntoDimension<Dim = D>) -> Self
    {
        ArrayBuilder {
            dim: dim.into_dimension(),
            memory_order: Order::C,
            generator: ElementGenerator::Sequential,
        }
    }

    pub fn memory_order(mut self, order: Order) -> Self
    {
        self.memory_order = order;
        self
    }

    pub fn generator(mut self, generator: ElementGenerator) -> Self
    {
        self.generator = generator;
        self
    }

    pub fn build<T>(self) -> Array<T, D>
    where T: Num + Clone
    {
        let mut current = T::zero();
        let size = self.dim.size();
        let use_zeros = self.generator == ElementGenerator::Zero;
        Array::from_iter((0..size).map(|_| {
            let ret = current.clone();
            if !use_zeros {
                current = ret.clone() + T::one();
            }
            ret
        }))
        .into_shape_with_order((self.dim, self.memory_order))
        .unwrap()
    }
}

#[test]
fn test_order()
{
    let (m, n) = (12, 13);
    let c = ArrayBuilder::new((m, n))
        .memory_order(Order::C)
        .build::<i32>();
    let f = ArrayBuilder::new((m, n))
        .memory_order(Order::F)
        .build::<i32>();

    assert_eq!(c.shape(), &[m, n]);
    assert_eq!(f.shape(), &[m, n]);
    assert_eq!(c.strides(), &[n as isize, 1]);
    assert_eq!(f.strides(), &[1, m as isize]);
}
