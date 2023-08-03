use ndarray::indices_of;
use ndarray::prelude::*;
use ndarray::Order;

#[test]
fn test_ixdyn_index_iterate() {
    for &order in &[Order::C, Order::F] {
        let mut a = Array::zeros((2, 3, 4).set_f(order.is_column_major()));
        let dim = a.shape().to_vec();
        for ((i, j, k), elt) in a.indexed_iter_mut() {
            *elt = i + 10 * j + 100 * k;
        }
        let a = a.into_shape_with_order((dim, order)).unwrap();
        println!("{:?}", a.dim());
        let mut c = 0;
        for i in indices_of(&a) {
            let ans = i[0] + 10 * i[1] + 100 * i[2];
            println!("{:?}", i);
            assert_eq!(a[i], ans);
            c += 1;
        }
        assert_eq!(c, a.len());
    }
}
