use ndarray::prelude::*;

#[test]
fn broadcast_1()
{
    let a_dim = Dim([2, 4, 2, 2]);
    let b_dim = Dim([2, 1, 2, 1]);
    let a = Array::from_iter(0..a_dim.size())
        .into_shape_with_order(a_dim)
        .unwrap();
    let b = Array::from_iter(0..b_dim.size())
        .into_shape_with_order(b_dim)
        .unwrap();
    assert!(b.broadcast(a.dim()).is_some());

    let c_dim = Dim([2, 1]);
    let c = Array::from_iter(0..c_dim.size())
        .into_shape_with_order(c_dim)
        .unwrap();
    assert!(c.broadcast(1).is_none());
    assert!(c.broadcast(()).is_none());
    assert!(c.broadcast((2, 1)).is_some());
    assert!(c.broadcast((2, 2)).is_some());
    assert!(c.broadcast((32, 2, 1)).is_some());
    assert!(c.broadcast((32, 1, 2)).is_none());

    /* () can be broadcast to anything */
    let z = Array::<f32, _>::zeros(());
    assert!(z.broadcast(()).is_some());
    assert!(z.broadcast(1).is_some());
    assert!(z.broadcast(3).is_some());
    assert!(z.broadcast((7, 2, 9)).is_some());
}

#[test]
fn test_add()
{
    let a_dim = Dim([2, 4, 2, 2]);
    let b_dim = Dim([2, 1, 2, 1]);
    let mut a = Array::from_iter(0..a_dim.size())
        .into_shape_with_order(a_dim)
        .unwrap();
    let b = Array::from_iter(0..b_dim.size())
        .into_shape_with_order(b_dim)
        .unwrap();
    a += &b;
    let t = Array::from_elem((), 1);
    a += &t;
}

#[test]
#[should_panic]
fn test_add_incompat()
{
    let a_dim = Dim([2, 4, 2, 2]);
    let mut a = Array::from_iter(0..a_dim.size())
        .into_shape_with_order(a_dim)
        .unwrap();
    let incompat = Array::from_elem(3, 1);
    a += &incompat;
}

#[test]
fn test_broadcast()
{
    let (_, n, k) = (16, 16, 16);
    let x1 = 1.;
    // b0 broadcast 1 -> n, k
    let x = Array::from(vec![x1]);
    let b0 = x.broadcast((n, k)).unwrap();
    // b1 broadcast n -> n, k
    let b1 = Array::from_elem(n, x1);
    let b1 = b1.broadcast((n, k)).unwrap();
    // b2 is n, k
    let b2 = Array::from_elem((n, k), x1);

    println!("b0=\n{:?}", b0);
    println!("b1=\n{:?}", b1);
    println!("b2=\n{:?}", b2);
    assert_eq!(b0, b1);
    assert_eq!(b0, b2);
}

#[test]
fn test_broadcast_1d()
{
    let n = 16;
    let x1 = 1.;
    // b0 broadcast 1 -> n
    let x = Array::from(vec![x1]);
    let b0 = x.broadcast(n).unwrap();
    let b2 = Array::from_elem(n, x1);

    println!("b0=\n{:?}", b0);
    println!("b2=\n{:?}", b2);
    assert_eq!(b0, b2);
}
