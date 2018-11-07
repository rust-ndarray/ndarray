#![feature(test)]

extern crate test;
extern crate rawpointer;
use test::Bencher;
use test::black_box;
use rawpointer::PointerExt;

#[macro_use(s, azip)]
extern crate ndarray;
use ndarray::prelude::*;
use ndarray::{Zip, FoldWhile};

#[bench]
fn iter_sum_2d_regular(bench: &mut Bencher)
{
    let a = Array::<i32, _>::zeros((64, 64));
    bench.iter(|| {
        a.iter().fold(0, |acc, &x| acc + x)
    });
}

#[bench]
fn iter_sum_2d_cutout(bench: &mut Bencher)
{
    let a = Array::<i32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = av;
    bench.iter(|| {
        a.iter().fold(0, |acc, &x| acc + x)
    });
}

#[bench]
fn iter_all_2d_cutout(bench: &mut Bencher)
{
    let a = Array::<i32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = av;
    bench.iter(|| {
        a.iter().all(|&x| x >= 0)
    });
}

#[bench]
fn iter_sum_2d_transpose(bench: &mut Bencher)
{
    let a = Array::<i32, _>::zeros((66, 66));
    let a = a.t();
    bench.iter(|| {
        a.iter().fold(0, |acc, &x| acc + x)
    });
}

#[bench]
fn iter_filter_sum_2d_u32(bench: &mut Bencher)
{
    let a = Array::linspace(0., 1., 256).into_shape((16, 16)).unwrap();
    let b = a.mapv(|x| (x * 100.) as u32);
    bench.iter(|| {
        b.iter().filter(|&&x| x < 75).fold(0, |acc, &x| acc + x)
    });
}

#[bench]
fn iter_filter_sum_2d_f32(bench: &mut Bencher)
{
    let a = Array::linspace(0., 1., 256).into_shape((16, 16)).unwrap();
    let b = a * 100.;
    bench.iter(|| {
        b.iter().filter(|&&x| x < 75.).fold(0., |acc, &x| acc + x)
    });
}

#[bench]
fn iter_filter_sum_2d_stride_u32(bench: &mut Bencher)
{
    let a = Array::linspace(0., 1., 256).into_shape((16, 16)).unwrap();
    let b = a.mapv(|x| (x * 100.) as u32);
    let b = b.slice(s![.., ..;2]);
    bench.iter(|| {
        b.iter().filter(|&&x| x < 75).fold(0, |acc, &x| acc + x)
    });
}

#[bench]
fn iter_filter_sum_2d_stride_f32(bench: &mut Bencher)
{
    let a = Array::linspace(0., 1., 256).into_shape((16, 16)).unwrap();
    let b = a * 100.;
    let b = b.slice(s![.., ..;2]);
    bench.iter(|| {
        b.iter().filter(|&&x| x < 75.).fold(0., |acc, &x| acc + x)
    });
}

const ZIPSZ: usize = 10_000;

#[bench]
fn sum_3_std_zip1(bench: &mut Bencher)
{
    let a = vec![1; ZIPSZ];
    let b = vec![1; ZIPSZ];
    let c = vec![1; ZIPSZ];
    bench.iter(|| {
        a.iter().zip(b.iter().zip(&c)).fold(0, |acc, (&a, (&b, &c))| {
            acc + a + b + c
        })
    });
}

#[bench]
fn sum_3_std_zip2(bench: &mut Bencher)
{
    let a = vec![1; ZIPSZ];
    let b = vec![1; ZIPSZ];
    let c = vec![1; ZIPSZ];
    bench.iter(|| {
        a.iter().zip(b.iter()).zip(&c).fold(0, |acc, ((&a, &b), &c)| {
            acc + a + b + c
        })
    });
}

#[bench]
fn sum_3_std_zip3(bench: &mut Bencher)
{
    let a = vec![1; ZIPSZ];
    let b = vec![1; ZIPSZ];
    let c = vec![1; ZIPSZ];
    bench.iter(|| {
        let mut s = 0;
        for ((&a, &b), &c) in a.iter().zip(b.iter()).zip(&c) {
            s += a + b + c
        }
        s
    });
}

#[bench]
fn vector_sum_3_std_zip(bench: &mut Bencher)
{
    let a = vec![1.; ZIPSZ];
    let b = vec![1.; ZIPSZ];
    let mut c = vec![1.; ZIPSZ];
    bench.iter(|| {
        for ((&a, &b), c) in a.iter().zip(b.iter()).zip(&mut c) {
            *c += a + b;
        }
    });
}

#[bench]
fn sum_3_azip(bench: &mut Bencher)
{
    let a = vec![1; ZIPSZ];
    let b = vec![1; ZIPSZ];
    let c = vec![1; ZIPSZ];
    bench.iter(|| {
        let mut s = 0;
        azip!(a, b, c in {
            s += a + b + c;
        });
        s
    });
}

#[bench]
fn sum_3_azip_fold(bench: &mut Bencher)
{
    let a = vec![1; ZIPSZ];
    let b = vec![1; ZIPSZ];
    let c = vec![1; ZIPSZ];
    bench.iter(|| {
        Zip::from(&a).and(&b).and(&c).fold_while(0, |acc, &a, &b, &c| {
            FoldWhile::Continue(acc + a + b + c)
        }).into_inner()
    });
}

#[bench]
fn vector_sum_3_azip(bench: &mut Bencher)
{
    let a = vec![1.; ZIPSZ];
    let b = vec![1.; ZIPSZ];
    let mut c = vec![1.; ZIPSZ];
    bench.iter(|| {
        azip!(a, b, mut c in {
            *c += a + b;
        });
    });
}

fn vector_sum3_unchecked(a: &[f64], b: &[f64], c: &mut [f64]) {
    for i in 0..c.len() {
        unsafe {
            *c.get_unchecked_mut(i) += *a.get_unchecked(i) + *b.get_unchecked(i);
        }
    }
}

#[bench]
fn vector_sum_3_zip_unchecked(bench: &mut Bencher)
{
    let a = vec![1.; ZIPSZ];
    let b = vec![1.; ZIPSZ];
    let mut c = vec![1.; ZIPSZ];
    bench.iter(move || {
        vector_sum3_unchecked(&a, &b, &mut c);
    });
}

#[bench]
fn vector_sum_3_zip_unchecked_manual(bench: &mut Bencher)
{
    let a = vec![1.; ZIPSZ];
    let b = vec![1.; ZIPSZ];
    let mut c = vec![1.; ZIPSZ];
    bench.iter(move || {
        unsafe {
            let mut ap = a.as_ptr();
            let mut bp = b.as_ptr();
            let mut cp = c.as_mut_ptr();
            let cend = cp.offset(c.len() as isize);
            while cp != cend {
                *cp.post_inc() += *ap.post_inc() + *bp.post_inc();
            }
        }
    });
}

// index iterator size
const ISZ: usize = 16;
const I2DSZ: usize = 64;

#[bench]
fn indexed_iter_1d_ix1(bench: &mut Bencher) {
    let mut a = Array::<f64, _>::zeros(I2DSZ * I2DSZ);
    for (i, elt) in a.indexed_iter_mut() {
        *elt = i as _;
    }

    bench.iter(|| {
        for (i, &_elt) in a.indexed_iter() {
            //assert!(a[i] == elt);
            black_box(i);
        }
    })
}

#[bench]
fn indexed_zip_1d_ix1(bench: &mut Bencher) {
    let mut a = Array::<f64, _>::zeros(I2DSZ * I2DSZ);
    for (i, elt) in a.indexed_iter_mut() {
        *elt = i as _;
    }

    bench.iter(|| {
        Zip::indexed(&a)
            .apply(|i, &_elt| {
                black_box(i);
                //assert!(a[i] == elt);
            });
    })
}

#[bench]
fn indexed_iter_2d_ix2(bench: &mut Bencher) {
    let mut a = Array::<f64, _>::zeros((I2DSZ, I2DSZ));
    for ((i, j), elt) in a.indexed_iter_mut() {
        *elt = (i + 100 * j) as _;
    }

    bench.iter(|| {
        for (i, &_elt) in a.indexed_iter() {
            //assert!(a[i] == elt);
            black_box(i);
        }
    })
}
#[bench]
fn indexed_zip_2d_ix2(bench: &mut Bencher) {
    let mut a = Array::<f64, _>::zeros((I2DSZ, I2DSZ));
    for ((i, j), elt) in a.indexed_iter_mut() {
        *elt = (i + 100 * j) as _;
    }

    bench.iter(|| {
        Zip::indexed(&a)
            .apply(|i, &_elt| {
                black_box(i);
                //assert!(a[i] == elt);
            });
    })
}



#[bench]
fn indexed_iter_3d_ix3(bench: &mut Bencher) {
    let mut a = Array::<f64, _>::zeros((ISZ, ISZ, ISZ));
    for ((i, j, k), elt) in a.indexed_iter_mut() {
        *elt = (i + 100 * j + 10000 * k) as _;
    }

    bench.iter(|| {
        for (i, &_elt) in a.indexed_iter() {
            //assert!(a[i] == elt);
            black_box(i);
        }
    })
}

#[bench]
fn indexed_zip_3d_ix3(bench: &mut Bencher) {
    let mut a = Array::<f64, _>::zeros((ISZ, ISZ, ISZ));
    for ((i, j, k), elt) in a.indexed_iter_mut() {
        *elt = (i + 100 * j + 10000 * k) as _;
    }

    bench.iter(|| {
        Zip::indexed(&a)
            .apply(|i, &_elt| {
                black_box(i);
                //assert!(a[i] == elt);
            });
    })
}

#[bench]
fn indexed_iter_3d_dyn(bench: &mut Bencher) {
    let mut a = Array::<f64, _>::zeros((ISZ, ISZ, ISZ));
    for ((i, j, k), elt) in a.indexed_iter_mut() {
        *elt = (i + 100 * j + 10000 * k) as _;
    }
    let a = a.into_shape(&[ISZ; 3][..]).unwrap();

    bench.iter(|| {
        for (i, &_elt) in a.indexed_iter() {
            //assert!(a[i] == elt);
            black_box(i);
        }
    })
}
