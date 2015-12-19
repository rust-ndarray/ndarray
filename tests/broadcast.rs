
extern crate ndarray;

use ndarray::{Array, Dimension};

#[test]
fn broadcast_1()
{
    let a_dim = (2, 4, 2, 2);
    let b_dim = (2, 1, 2, 1);
    let a = Array::linspace(0., 1., a_dim.size()).reshape(a_dim);
    let b = Array::linspace(0., 1., b_dim.size()).reshape(b_dim);
    assert!(b.broadcast_iter(a.dim()).is_some());

    let c_dim = (2, 1);
    let c = Array::linspace(0., 1., c_dim.size()).reshape(c_dim);
    assert!(c.broadcast_iter(1).is_none());
    assert!(c.broadcast_iter(()).is_none());
    assert!(c.broadcast_iter((2, 1)).is_some());
    assert!(c.broadcast_iter((2, 2)).is_some());
    assert!(c.broadcast_iter((32, 2, 1)).is_some());
    assert!(c.broadcast_iter((32, 1, 2)).is_none());

    /* () can be broadcast to anything */
    let z = Array::<f32,_>::zeros(());
    assert!(z.broadcast_iter(()).is_some());
    assert!(z.broadcast_iter(1).is_some());
    assert!(z.broadcast_iter(3).is_some());
    assert!(z.broadcast_iter((7,2,9)).is_some());
}

#[test]
fn test_add()
{
    let a_dim = (2, 4, 2, 2);
    let b_dim = (2, 1, 2, 1);
    let mut a = Array::linspace(0.0, 1., a_dim.size()).reshape(a_dim);
    let b = Array::linspace(0.0, 1., b_dim.size()).reshape(b_dim);
    a.iadd(&b);
    let t = Array::from_elem((), 1.0f32);
    a.iadd(&t);
}

#[test] #[should_panic]
fn test_add_incompat()
{
    let a_dim = (2, 4, 2, 2);
    let mut a = Array::linspace(0.0, 1., a_dim.size()).reshape(a_dim);
    let incompat = Array::from_elem(3, 1.0f32);
    a.iadd(&incompat);
}
