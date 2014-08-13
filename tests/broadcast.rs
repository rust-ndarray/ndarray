
extern crate ndarray;

use ndarray::{Array, Dimension};

#[test]
fn broadcast_1()
{
    let a_dim = (2u, 4u, 2u, 2u);
    let b_dim = (2u, 1u, 2u, 1u);
    let a = Array::from_iter(range(0.0, a_dim.size() as f32)).reshape(a_dim);
    let b = Array::from_iter(range(0.0, b_dim.size() as f32)).reshape(b_dim);
    assert!(b.broadcast_iter(a.dim()).is_some());

    let c_dim = (2u, 1u);
    let c = Array::from_iter(range(0.0, c_dim.size() as f32)).reshape(c_dim);
    assert!(c.broadcast_iter(1u).is_none());
    assert!(c.broadcast_iter(()).is_none());
    assert!(c.broadcast_iter((2u, 1u)).is_some());
    assert!(c.broadcast_iter((2u, 2u)).is_some());
    assert!(c.broadcast_iter((32u, 2u, 1u)).is_some());
    assert!(c.broadcast_iter((32u, 1u, 2u)).is_none());

    /* () can be broadcast to anything */
    let z = Array::<f32,_>::zeros(());
    assert!(z.broadcast_iter(()).is_some());
    assert!(z.broadcast_iter(1u).is_some());
    assert!(z.broadcast_iter(3u).is_some());
    assert!(z.broadcast_iter((7u,2u,9u)).is_some());
}

#[test]
fn test_add()
{
    let a_dim = (2u, 4u, 2u, 2u);
    let b_dim = (2u, 1u, 2u, 1u);
    let mut a = Array::from_iter(range(0.0, a_dim.size() as f32)).reshape(a_dim);
    let b = Array::from_iter(range(0.0, b_dim.size() as f32)).reshape(b_dim);
    a.iadd(&b);
    let t = Array::from_elem((), 1.0f32);
    a.iadd(&t);
}

#[test] #[should_fail]
fn test_add_incompat()
{
    let a_dim = (2u, 4u, 2u, 2u);
    let mut a = Array::from_iter(range(0.0, a_dim.size() as f32)).reshape(a_dim);
    let incompat = Array::from_elem(3u, 1.0f32);
    a.iadd(&incompat);
}
