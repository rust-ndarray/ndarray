
extern crate ndarray;

use ndarray::{RcArray, Dimension};

#[test]
fn broadcast_1()
{
    let a_dim = (2, 4, 2, 2);
    let b_dim = (2, 1, 2, 1);
    let a = RcArray::linspace(0., 1., a_dim.size()).reshape(a_dim);
    let b = RcArray::linspace(0., 1., b_dim.size()).reshape(b_dim);
    assert!(b.broadcast(a.dim()).is_some());

    let c_dim = (2, 1);
    let c = RcArray::linspace(0., 1., c_dim.size()).reshape(c_dim);
    assert!(c.broadcast(1).is_none());
    assert!(c.broadcast(()).is_none());
    assert!(c.broadcast((2, 1)).is_some());
    assert!(c.broadcast((2, 2)).is_some());
    assert!(c.broadcast((32, 2, 1)).is_some());
    assert!(c.broadcast((32, 1, 2)).is_none());

    /* () can be broadcast to anything */
    let z = RcArray::<f32,_>::zeros(());
    assert!(z.broadcast(()).is_some());
    assert!(z.broadcast(1).is_some());
    assert!(z.broadcast(3).is_some());
    assert!(z.broadcast((7,2,9)).is_some());
}

#[test]
fn test_add()
{
    let a_dim = (2, 4, 2, 2);
    let b_dim = (2, 1, 2, 1);
    let mut a = RcArray::linspace(0.0, 1., a_dim.size()).reshape(a_dim);
    let b = RcArray::linspace(0.0, 1., b_dim.size()).reshape(b_dim);
    a += &b;
    let t = RcArray::from_elem((), 1.0f32);
    a += &t;
}

#[test] #[should_panic]
fn test_add_incompat()
{
    let a_dim = (2, 4, 2, 2);
    let mut a = RcArray::linspace(0.0, 1., a_dim.size()).reshape(a_dim);
    let incompat = RcArray::from_elem(3, 1.0f32);
    a += &incompat;
}
