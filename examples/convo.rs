#![allow(unused)]
extern crate ndarray;

#[cfg(feature = "std")]
use num_traits::Float;

use ndarray::prelude::*;

const SOBEL_X: [[f32; 3]; 3] = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]];
const SOBEL_Y: [[f32; 3]; 3] = [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]];
const SHARPEN: [[f32; 3]; 3] = [[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]];

type Kernel3x3<A> = [[A; 3]; 3];

#[inline(never)]
#[cfg(feature = "std")]
fn conv_3x3<F>(a: &ArrayView2<'_, F>, out: &mut ArrayViewMut2<'_, F>, kernel: &Kernel3x3<F>)
where
    F: Float,
{
    let (n, m) = a.dim();
    let (np, mp) = out.dim();
    if n < 3 || m < 3 {
        return;
    }
    assert!(np >= n && mp >= m);
    // i, j offset by -1 so that we can use unsigned indices
    unsafe {
        for i in 0..n - 2 {
            for j in 0..m - 2 {
                let mut conv = F::zero();
                #[allow(clippy::needless_range_loop)]
                for k in 0..3 {
                    for l in 0..3 {
                        conv = conv + *a.uget((i + k, j + l)) * kernel[k][l];
                        //conv += a[[i + k, j + l]] * x_kernel[k][l];
                    }
                }
                *out.uget_mut((i + 1, j + 1)) = conv;
            }
        }
    }
}

#[cfg(feature = "std")]
fn main() {
    let n = 16;
    let mut a = Array::zeros((n, n));
    // make a circle
    let c = (8., 8.);
    for ((i, j), elt) in a.indexed_iter_mut() {
        {
            let s = ((i as f32) - c.0).powi(2) + (j as f32 - c.1).powi(2);
            if s.sqrt() > 3. && s.sqrt() < 6. {
                *elt = 1.;
            }
        }
    }
    println!("{:2}", a);
    let mut res = Array::zeros(a.dim());
    for _ in 0..1000 {
        conv_3x3(&a.view(), &mut res.view_mut(), &SOBEL_X);
    }
    println!("{:2}", res);
}
#[cfg(not(feature = "std"))]
fn main() {}
