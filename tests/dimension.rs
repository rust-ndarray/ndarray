#![allow(clippy::float_cmp)]

use defmac::defmac;

use ndarray::{arr2, ArcArray, Array, Axis, Dim, Dimension, IxDyn, RemoveAxis};

use std::hash::{Hash, Hasher};

#[test]
fn insert_axis() {
    assert_eq!(Dim([]).insert_axis(Axis(0)), Dim([1]));

    assert_eq!(Dim([3]).insert_axis(Axis(0)), Dim([1, 3]));
    assert_eq!(Dim([3]).insert_axis(Axis(1)), Dim([3, 1]));

    assert_eq!(Dim([2, 3]).insert_axis(Axis(0)), Dim([1, 2, 3]));
    assert_eq!(Dim([2, 3]).insert_axis(Axis(1)), Dim([2, 1, 3]));
    assert_eq!(Dim([2, 3]).insert_axis(Axis(2)), Dim([2, 3, 1]));

    assert_eq!(Dim([2, 3, 4]).insert_axis(Axis(2)), Dim([2, 3, 1, 4]));

    assert_eq!(
        Dim([2, 3, 4, 5, 6, 7]).insert_axis(Axis(2)),
        Dim(vec![2, 3, 1, 4, 5, 6, 7])
    );

    assert_eq!(Dim(vec![]).insert_axis(Axis(0)), Dim(vec![1]));

    assert_eq!(Dim(vec![2, 3]).insert_axis(Axis(0)), Dim(vec![1, 2, 3]));
    assert_eq!(Dim(vec![2, 3]).insert_axis(Axis(1)), Dim(vec![2, 1, 3]));
    assert_eq!(Dim(vec![2, 3]).insert_axis(Axis(2)), Dim(vec![2, 3, 1]));

    assert_eq!(
        Dim(vec![2, 3, 4, 5, 6]).insert_axis(Axis(2)),
        Dim(vec![2, 3, 1, 4, 5, 6])
    );
    assert_eq!(
        Dim(vec![2, 3, 4, 5, 6, 7]).insert_axis(Axis(2)),
        Dim(vec![2, 3, 1, 4, 5, 6, 7])
    );
}

#[test]
fn remove_axis() {
    assert_eq!(Dim([3]).remove_axis(Axis(0)), Dim([]));
    assert_eq!(Dim([1, 2]).remove_axis(Axis(0)), Dim([2]));
    assert_eq!(Dim([4, 5, 6]).remove_axis(Axis(1)), Dim([4, 6]));

    assert_eq!(Dim(vec![1, 2]).remove_axis(Axis(0)), Dim(vec![2]));
    assert_eq!(Dim(vec![4, 5, 6]).remove_axis(Axis(1)), Dim(vec![4, 6]));

    let a = ArcArray::<f32, _>::zeros((4, 5));
    a.index_axis(Axis(1), 0);

    let a = ArcArray::<f32, _>::zeros(vec![4, 5, 6]);
    let _b = a
        .index_axis_move(Axis(1), 0)
        .reshape((4, 6))
        .reshape(vec![2, 3, 4]);
}

#[test]
#[allow(clippy::eq_op)]
fn dyn_dimension() {
    let a = arr2(&[[1., 2.], [3., 4.0]]).into_shape(vec![2, 2]).unwrap();
    assert_eq!(&a - &a, Array::zeros(vec![2, 2]));
    assert_eq!(a[&[0, 0][..]], 1.);
    assert_eq!(a[[0, 0]], 1.);

    let mut dim = vec![1; 1024];
    dim[16] = 4;
    dim[17] = 3;
    let z = Array::<f32, _>::zeros(dim.clone());
    assert_eq!(z.shape(), &dim[..]);
}

#[test]
fn dyn_insert() {
    let mut v = vec![2, 3, 4, 5];
    let mut dim = Dim(v.clone());
    defmac!(test_insert index => {
        dim = dim.insert_axis(Axis(index));
        v.insert(index, 1);
        assert_eq!(dim.slice(), &v[..]);
    });

    test_insert!(1);
    test_insert!(5);
    test_insert!(0);
    test_insert!(3);
    test_insert!(2);
    test_insert!(4);
    test_insert!(7);
}

#[test]
fn dyn_remove() {
    let mut v = vec![1, 2, 3, 4, 5, 6, 7];
    let mut dim = Dim(v.clone());
    defmac!(test_remove index => {
        dim = dim.remove_axis(Axis(index));
        v.remove(index);
        assert_eq!(dim.slice(), &v[..]);
    });

    test_remove!(1);
    test_remove!(2);
    test_remove!(3);
    test_remove!(0);
    test_remove!(2);
    test_remove!(0);
    test_remove!(0);
}

#[test]
fn fastest_varying_order() {
    let strides = Dim([2, 8, 4, 1]);
    let order = strides._fastest_varying_stride_order();
    assert_eq!(order.slice(), &[3, 0, 2, 1]);

    let strides = Dim([-2isize as usize, 8, -4isize as usize, -1isize as usize]);
    let order = strides._fastest_varying_stride_order();
    assert_eq!(order.slice(), &[3, 0, 2, 1]);

    assert_eq!(Dim([1, 3])._fastest_varying_stride_order(), Dim([0, 1]));
    assert_eq!(
        Dim([1, -3isize as usize])._fastest_varying_stride_order(),
        Dim([0, 1])
    );
    assert_eq!(Dim([7, 2])._fastest_varying_stride_order(), Dim([1, 0]));
    assert_eq!(
        Dim([-7isize as usize, 2])._fastest_varying_stride_order(),
        Dim([1, 0])
    );
    assert_eq!(
        Dim([6, 1, 3])._fastest_varying_stride_order(),
        Dim([1, 2, 0])
    );
    assert_eq!(
        Dim([-6isize as usize, 1, -3isize as usize])._fastest_varying_stride_order(),
        Dim([1, 2, 0])
    );

    // it's important that it produces distinct indices. Prefer the stable order
    // where 0 is before 1 when they are equal.
    assert_eq!(Dim([2, 2])._fastest_varying_stride_order(), [0, 1]);
    assert_eq!(Dim([2, 2, 1])._fastest_varying_stride_order(), [2, 0, 1]);
    assert_eq!(
        Dim([-2isize as usize, -2isize as usize, 3, 1, -2isize as usize])
            ._fastest_varying_stride_order(),
        [3, 0, 1, 4, 2]
    );
}

type ArrayF32<D> = Array<f32, D>;

/*
#[test]
fn min_stride_axis() {
    let a = ArrayF32::zeros(10);
    assert_eq!(a.min_stride_axis(), Axis(0));

    let a = ArrayF32::zeros((3, 3));
    assert_eq!(a.min_stride_axis(), Axis(1));
    assert_eq!(a.t().min_stride_axis(), Axis(0));

    let a = ArrayF32::zeros(vec![3, 3]);
    assert_eq!(a.min_stride_axis(), Axis(1));
    assert_eq!(a.t().min_stride_axis(), Axis(0));

    let min_axis = a.axes().min_by_key(|t| t.2.abs()).unwrap().axis();
    assert_eq!(min_axis, Axis(1));

    let mut b = ArrayF32::zeros(vec![2, 3, 4, 5]);
    assert_eq!(b.min_stride_axis(), Axis(3));
    for ax in 0..3 {
        b.swap_axes(3, ax);
        assert_eq!(b.min_stride_axis(), Axis(ax));
        b.swap_axes(3, ax);
    }

    let a = ArrayF32::zeros((3, 3));
    let v = a.broadcast((8, 3, 3)).unwrap();
    assert_eq!(v.min_stride_axis(), Axis(0));
}
*/

#[test]
fn max_stride_axis() {
    let a = ArrayF32::zeros(10);
    assert_eq!(a.max_stride_axis(), Axis(0));

    let a = ArrayF32::zeros((3, 3));
    assert_eq!(a.max_stride_axis(), Axis(0));
    assert_eq!(a.t().max_stride_axis(), Axis(1));

    let a = ArrayF32::zeros(vec![1, 3]);
    assert_eq!(a.max_stride_axis(), Axis(1));
    let a = ArrayF32::zeros((1, 3));
    assert_eq!(a.max_stride_axis(), Axis(1));

    let a = ArrayF32::zeros(vec![3, 3]);
    assert_eq!(a.max_stride_axis(), Axis(0));
    assert_eq!(a.t().max_stride_axis(), Axis(1));

    let mut b = ArrayF32::zeros(vec![2, 3, 4, 5]);
    assert_eq!(b.max_stride_axis(), Axis(0));
    for ax in 1..b.ndim() {
        b.swap_axes(0, ax);
        assert_eq!(b.max_stride_axis(), Axis(ax));
        b.swap_axes(0, ax);
    }
}

#[test]
fn test_indexing() {
    let mut x = Dim([1, 2]);

    assert_eq!(x[0], 1);
    assert_eq!(x[1], 2);

    x[0] = 7;
    assert_eq!(x, [7, 2]);
}

#[test]
fn test_operations() {
    let mut x = Dim([1, 2]);
    let mut y = Dim([1, 1]);

    assert_eq!(x + y, [2, 3]);

    x += y;
    assert_eq!(x, [2, 3]);
    x *= 2;
    assert_eq!(x, [4, 6]);

    y[0] -= 1;
    assert_eq!(y, [0, 1]);
}

#[test]
#[allow(clippy::cognitive_complexity)]
fn test_hash() {
    fn calc_hash<T: Hash>(value: &T) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }
    macro_rules! test_hash_eq {
        ($arr:expr) => {
            assert_eq!(calc_hash(&Dim($arr)), calc_hash(&Dim($arr)));
            assert_eq!(calc_hash(&Dim($arr)), calc_hash(&IxDyn(&$arr)));
        };
    }
    macro_rules! test_hash_ne {
        ($arr1:expr, $arr2:expr) => {
            assert_ne!(calc_hash(&Dim($arr1)), calc_hash(&Dim($arr2)));
            assert_ne!(calc_hash(&Dim($arr1)), calc_hash(&IxDyn(&$arr2)));
            assert_ne!(calc_hash(&IxDyn(&$arr1)), calc_hash(&Dim($arr2)));
        };
    }
    test_hash_eq!([]);
    test_hash_eq!([0]);
    test_hash_eq!([1]);
    test_hash_eq!([1, 2]);
    test_hash_eq!([3, 1, 2]);
    test_hash_eq!([3, 1, 4, 2]);
    test_hash_eq!([3, 1, 4, 2, 5]);
    test_hash_eq!([6, 3, 1, 4, 2, 5]);
    test_hash_ne!([0], [1]);
    test_hash_ne!([1, 2], [2, 1]);
    test_hash_ne!([3, 1, 2], [3, 1, 3]);
    test_hash_ne!([3, 1, 2, 4], [3, 1, 2, 3]);
    test_hash_ne!([3, 1, 2, 5, 4], [3, 1, 2, 4, 5]);
    test_hash_ne!([3, 1, 6, 2, 5, 4], [3, 1, 2, 4, 6, 5]);
}

#[test]
fn test_generic_operations() {
    fn test_dim<D: Dimension>(d: &D) {
        let mut x = d.clone();
        x[0] += 1;
        assert_eq!(x[0], 3);
        x += d;
        assert_eq!(x[0], 5);
    }

    test_dim(&Dim([2, 3, 4]));
    test_dim(&Dim(vec![2, 3, 4, 1]));
    test_dim(&Dim(2));
}

#[test]
fn test_array_view() {
    fn test_dim<D: Dimension>(d: &D) {
        assert_eq!(d.as_array_view().sum(), 7);
        assert_eq!(d.as_array_view().strides(), &[1]);
    }

    test_dim(&Dim([1, 2, 4]));
    test_dim(&Dim(vec![1, 1, 2, 3]));
    test_dim(&Dim(7));
}

#[test]
#[cfg(feature = "std")]
#[allow(clippy::cognitive_complexity)]
fn test_all_ndindex() {
    use ndarray::IntoDimension;
    macro_rules! ndindex {
    ($($i:expr),*) => {
        for &rev in &[false, true] {
            // rev is for C / F order
            let size = $($i *)* 1;
            let mut a = Array::linspace(0., (size - 1) as f64, size);
            if rev {
                a = a.reversed_axes();
            }
            for (i, &elt) in a.indexed_iter() {
                let dim = i.into_dimension();
                assert_eq!(elt, a[i]);
                assert_eq!(elt, a[dim]);
            }
            let dim = a.shape().to_vec();
            let b = a.broadcast(dim).unwrap();
            for (i, &elt) in b.indexed_iter() {
                let dim = i.into_dimension();
                assert_eq!(elt, b[dim.slice()]);
                assert_eq!(elt, b[&dim]);
                assert_eq!(elt, b[dim]);
            }
        }
    }
}
    ndindex!(10);
    ndindex!(10, 4);
    ndindex!(10, 4, 3);
    ndindex!(10, 4, 3, 2);
    ndindex!(10, 4, 3, 2, 2);
    ndindex!(10, 4, 3, 2, 2, 2);
}
