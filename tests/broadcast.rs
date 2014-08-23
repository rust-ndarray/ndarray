
extern crate ndarray;

use ndarray::{Array, Dimension};
use ndarray::{d1, d2, d3, d4};

#[test]
fn broadcast_1()
{
    let a_dim = d4(2, 4, 2, 2);
    let b_dim = d4(2, 1, 2, 1);
    let a = Array::from_iter(range(0.0, a_dim.size() as f32)).reshape(a_dim);
    let b = Array::from_iter(range(0.0, b_dim.size() as f32)).reshape(b_dim);
    assert!(b.broadcast_iter(a.dim()).is_some());

    let c_dim = d2(2, 1);
    let c = Array::from_iter(range(0.0, c_dim.size() as f32)).reshape(c_dim);
    assert!(c.broadcast_iter(d1(1)).is_none());
    assert!(c.broadcast_iter(()).is_none());
    assert!(c.broadcast_iter(d2(2, 1)).is_some());
    assert!(c.broadcast_iter(d2(2, 2)).is_some());
    assert!(c.broadcast_iter(d3(32, 2, 1)).is_some());
    assert!(c.broadcast_iter(d3(32, 1, 2)).is_none());

    /* () can be broadcast to anything */
    let z = Array::<f32,_>::zeros(());
    assert!(z.broadcast_iter(()).is_some());
    assert!(z.broadcast_iter(d1(1)).is_some());
    assert!(z.broadcast_iter(d1(3)).is_some());
    assert!(z.broadcast_iter(d3(7,2,9)).is_some());
}

#[test]
fn test_add()
{
    let a_dim = d4(2, 4, 2, 2);
    let b_dim = d4(2, 1, 2, 1);
    let mut a = Array::from_iter(range(0.0, a_dim.size() as f32)).reshape(a_dim);
    let b = Array::from_iter(range(0.0, b_dim.size() as f32)).reshape(b_dim);
    a.iadd(&b);
    let t = Array::from_elem((), 1.0f32);
    a.iadd(&t);
}

#[test] #[should_fail]
fn test_add_incompat()
{
    let a_dim = d4(2, 4, 2, 2);
    let mut a = Array::from_iter(range(0.0, a_dim.size() as f32)).reshape(a_dim);
    let incompat = Array::from_elem(d1(3), 1.0f32);
    a.iadd(&incompat);
}
