#![cfg(feature="rayon")]

extern crate rayon;

extern crate ndarray;
extern crate itertools;
#[macro_use] extern crate lazy_static;

use ndarray::prelude::*;
use ndarray::parallel::prelude::*;
use std::env;
use std::str::FromStr;

lazy_static!{
    static ref Pool: rayon::ThreadPool = {
        let example_num_threads = env::var("NUM_THREADS")
            .expect("Please set env var NUM_THREADS for this example");
        let n_threads = usize::from_str(&example_num_threads)
            .expect("Failed to parse NUM_THREADS");
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build().unwrap();
        pool
    };
}

const FASTEXP: usize = 256 * 5;

#[inline]
fn fastexp(x: f64) -> f64 {
    let x = 1. + x/1024.;
    x.powi(1024)
}

fn rayon_fastexp_cut(repeats: usize)
{
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    let mut a = a.slice_mut(s![.., ..-1]);

    // prime rayon
    //a.view_mut().into_par_iter().for_each(|x| *x += 1.);

    // now go with real job
    let start = std::time::Instant::now();
        for _ in 0..repeats {
            Pool.install(|| {
                // inner operation is entirely serial
                a.mapv_inplace(|x| fastexp(x))
            });
        }
    let time = start.elapsed();
    println!("Elapsed: {}.{:06}", time.as_secs(), time.subsec_micros());
}


fn main() {
    rayon_fastexp_cut(500);
}

