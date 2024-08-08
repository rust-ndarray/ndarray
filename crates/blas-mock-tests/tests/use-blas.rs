extern crate ndarray;

use ndarray::prelude::*;

use blas_mock_tests::CALL_COUNT;
use ndarray::linalg::general_mat_mul;
use ndarray::Order;
use ndarray_gen::array_builder::ArrayBuilder;

use itertools::iproduct;

#[test]
fn test_gen_mat_mul_uses_blas()
{
    let alpha = 1.0;
    let beta = 0.0;

    let sizes = vec![
        (8, 8, 8),
        (10, 10, 10),
        (8, 8, 1),
        (1, 10, 10),
        (10, 1, 10),
        (10, 10, 1),
        (1, 10, 1),
        (10, 1, 1),
        (1, 1, 10),
        (4, 17, 3),
        (17, 3, 22),
        (19, 18, 2),
        (16, 17, 15),
        (15, 16, 17),
        (67, 63, 62),
    ];
    let strides = &[1, 2, -1, -2];
    let cf_order = [Order::C, Order::F];

    // test different strides and memory orders
    for &(m, k, n) in &sizes {
        for (&s1, &s2) in iproduct!(strides, strides) {
            for (ord1, ord2, ord3) in iproduct!(cf_order, cf_order, cf_order) {
                println!("Case s1={}, s2={}, orders={:?}, {:?}, {:?}", s1, s2, ord1, ord2, ord3);

                let a = ArrayBuilder::new((m, k)).memory_order(ord1).build();
                let b = ArrayBuilder::new((k, n)).memory_order(ord2).build();
                let mut c = ArrayBuilder::new((m, n)).memory_order(ord3).build();

                {
                    let av;
                    let bv;
                    let mut cv;

                    if s1 != 1 || s2 != 1 {
                        av = a.slice(s![..;s1, ..;s2]);
                        bv = b.slice(s![..;s2, ..;s2]);
                        cv = c.slice_mut(s![..;s1, ..;s2]);
                    } else {
                        // different stride cases for slicing versus not sliced (for axes of
                        // len=1); so test not sliced here.
                        av = a.view();
                        bv = b.view();
                        cv = c.view_mut();
                    }

                    let pre_count = CALL_COUNT.with(|ctx| *ctx.borrow());
                    general_mat_mul(alpha, &av, &bv, beta, &mut cv);
                    let after_count = CALL_COUNT.with(|ctx| *ctx.borrow());
                    let ncalls = after_count - pre_count;
                    debug_assert!(ncalls <= 1);

                    let always_uses_blas = s1 == 1 && s2 == 1;

                    if always_uses_blas {
                        assert_eq!(ncalls, 1, "Contiguous arrays should use blas, orders={:?}", (ord1, ord2, ord3));
                    }

                    let should_use_blas = av.strides().iter().all(|&s| s > 0)
                        && bv.strides().iter().all(|&s| s > 0)
                        && cv.strides().iter().all(|&s| s > 0)
                        && av.strides().iter().any(|&s| s == 1)
                        && bv.strides().iter().any(|&s| s == 1)
                        && cv.strides().iter().any(|&s| s == 1);
                    assert_eq!(should_use_blas, ncalls > 0);
                }
            }
        }
    }
}
