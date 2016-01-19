use libnum;

use std::cmp;
use std::ops::{
    Add,
    Mul,
};

/// Compute the dot product.
///
/// `xs` and `ys` must be the same length
pub fn unrolled_dot<A>(xs: &[A], ys: &[A]) -> A
    where A: Clone + Add<Output=A> + Mul<Output=A> + libnum::Zero,
{
    debug_assert_eq!(xs.len(), ys.len());
    // eightfold unrolled so that floating point can be vectorized
    // (even with strict floating point accuracy semantics)
    let len = cmp::min(xs.len(), ys.len());
    let mut xs = &xs[..len];
    let mut ys = &ys[..len];
    let mut sum = A::zero();
    let (mut p0, mut p1, mut p2, mut p3,
         mut p4, mut p5, mut p6, mut p7) =
        (A::zero(), A::zero(), A::zero(), A::zero(),
         A::zero(), A::zero(), A::zero(), A::zero());
    while xs.len() >= 8 {
        p0 = p0 + xs[0].clone() * ys[0].clone();
        p1 = p1 + xs[1].clone() * ys[1].clone();
        p2 = p2 + xs[2].clone() * ys[2].clone();
        p3 = p3 + xs[3].clone() * ys[3].clone();
        p4 = p4 + xs[4].clone() * ys[4].clone();
        p5 = p5 + xs[5].clone() * ys[5].clone();
        p6 = p6 + xs[6].clone() * ys[6].clone();
        p7 = p7 + xs[7].clone() * ys[7].clone();

        xs = &xs[8..];
        ys = &ys[8..];
    }
    sum = sum.clone() + (p0 + p4);
    sum = sum.clone() + (p1 + p5);
    sum = sum.clone() + (p2 + p6);
    sum = sum.clone() + (p3 + p7);
    for i in 0..xs.len() {
        sum = sum.clone() + xs[i].clone() * ys[i].clone();
    }
    sum
}

/// Compute pairwise equality
///
/// `xs` and `ys` must be the same length
pub fn unrolled_eq<A>(xs: &[A], ys: &[A]) -> bool
    where A: PartialEq
{
    debug_assert_eq!(xs.len(), ys.len());
    // eightfold unrolled for performance (this is not done by llvm automatically)
    let len = cmp::min(xs.len(), ys.len());
    let mut xs = &xs[..len];
    let mut ys = &ys[..len];

    while xs.len() >= 8 {
        if (xs[0] != ys[0])
        | (xs[1] != ys[1])
        | (xs[2] != ys[2])
        | (xs[3] != ys[3])
        | (xs[4] != ys[4])
        | (xs[5] != ys[5])
        | (xs[6] != ys[6])
        | (xs[7] != ys[7]) { return false; }
        xs = &xs[8..];
        ys = &ys[8..];
    }

    for i in 0..xs.len() {
        if xs[i] != ys[i] {
            return false;
        }
    }

    return true;
}
