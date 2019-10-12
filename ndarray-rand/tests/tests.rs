use ndarray::{Array, Array2, ArrayView1, Axis};
#[cfg(feature = "quickcheck")]
use ndarray_rand::rand::{distributions::Distribution, thread_rng};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::{RandomExt, SamplingStrategy};
use quickcheck::quickcheck;

#[test]
fn test_dim() {
    let (mm, nn) = (5, 5);
    for m in 0..mm {
        for n in 0..nn {
            let a = Array::random((m, n), Uniform::new(0., 2.));
            assert_eq!(a.shape(), &[m, n]);
            assert!(a.iter().all(|x| *x < 2.));
            assert!(a.iter().all(|x| *x >= 0.));
        }
    }
}

#[test]
fn test_standard_normal() {
    let shape = 2usize;
    let n = MultivariateStandardNormal::new(shape);
    let ref mut rng = rand::thread_rng();
    let s: ndarray::Array1<f64> = n.sample(rng);
    assert_eq!(s.shape(), &[2]);
}

#[cfg(features = "normaldist")]
#[test]
fn test_normal() {
    use ndarray::IntoDimension;
    use ndarray::{Array1, arr2};
    use ndarray_rand::normal::advanced::MultivariateNormal;
    let mean = Array1::from_vec([1., 0.]);
    let covar = arr2([
        [1., 0.8], [0.8, 1.]]);
    let ref mut rng = rand::thread_rng();
    let n = MultivariateNormal::new(mean, covar);
    if let Ok(n) = n {
        let x = n.sample(rng);
        assert_eq!(x.shape(), &[2, 2]);
    }
}
#[should_panic]
fn oversampling_without_replacement_should_panic() {
    let m = 5;
    let a = Array::random((m, 4), Uniform::new(0., 2.));
    let _samples = a.sample_axis(Axis(0), m + 1, SamplingStrategy::WithoutReplacement);
}

quickcheck! {
    fn oversampling_with_replacement_is_fine(m: usize, n: usize) -> bool {
        let a = Array::random((m, n), Uniform::new(0., 2.));
        // Higher than the length of both axes
        let n_samples = m + n + 1;

        // We don't want to deal with sampling from 0-length axes in this test
        if m != 0 {
            if !sampling_works(&a, SamplingStrategy::WithReplacement, Axis(0), n_samples) {
                return false;
            }
        }

        // We don't want to deal with sampling from 0-length axes in this test
        if n != 0 {
            if !sampling_works(&a, SamplingStrategy::WithReplacement, Axis(1), n_samples) {
                return false;
            }
        }

        true
    }
}

#[cfg(feature = "quickcheck")]
quickcheck! {
    fn sampling_behaves_as_expected(m: usize, n: usize, strategy: SamplingStrategy) -> bool {
        let a = Array::random((m, n), Uniform::new(0., 2.));
        let mut rng = &mut thread_rng();

        // We don't want to deal with sampling from 0-length axes in this test
        if m != 0 {
            let n_row_samples = Uniform::from(1..m+1).sample(&mut rng);
            if !sampling_works(&a, strategy.clone(), Axis(0), n_row_samples) {
                return false;
            }
        }

        // We don't want to deal with sampling from 0-length axes in this test
        if n != 0 {
            let n_col_samples = Uniform::from(1..n+1).sample(&mut rng);
            if !sampling_works(&a, strategy, Axis(1), n_col_samples) {
                return false;
            }
        }

        true
    }
}

fn sampling_works(
    a: &Array2<f64>,
    strategy: SamplingStrategy,
    axis: Axis,
    n_samples: usize,
) -> bool {
    let samples = a.sample_axis(axis, n_samples, strategy);
    samples
        .axis_iter(axis)
        .all(|lane| is_subset(&a, &lane, axis))
}

// Check if, when sliced along `axis`, there is at least one lane in `a` equal to `b`
fn is_subset(a: &Array2<f64>, b: &ArrayView1<f64>, axis: Axis) -> bool {
    a.axis_iter(axis).any(|lane| &lane == b)
}

#[test]
#[should_panic]
fn sampling_without_replacement_from_a_zero_length_axis_should_panic() {
    let n = 5;
    let a = Array::random((0, n), Uniform::new(0., 2.));
    let _samples = a.sample_axis(Axis(0), 1, SamplingStrategy::WithoutReplacement);
}

#[test]
#[should_panic]
fn sampling_with_replacement_from_a_zero_length_axis_should_panic() {
    let n = 5;
    let a = Array::random((0, n), Uniform::new(0., 2.));
    let _samples = a.sample_axis(Axis(0), 1, SamplingStrategy::WithReplacement);
}
