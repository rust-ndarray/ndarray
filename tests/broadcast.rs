use ndarray::prelude::*;

#[test]
#[cfg(feature = "std")]
fn broadcast_1() {
    let a_dim = Dim([2, 4, 2, 2]);
    let b_dim = Dim([2, 1, 2, 1]);
    let a = ArcArray::linspace(0., 1., a_dim.size()).reshape(a_dim);
    let b = ArcArray::linspace(0., 1., b_dim.size()).reshape(b_dim);
    assert!(b.broadcast(a.dim()).is_some());

    let c_dim = Dim([2, 1]);
    let c = ArcArray::linspace(0., 1., c_dim.size()).reshape(c_dim);
    assert!(c.broadcast(1).is_none());
    assert!(c.broadcast(()).is_none());
    assert!(c.broadcast((2, 1)).is_some());
    assert!(c.broadcast((2, 2)).is_some());
    assert!(c.broadcast((32, 2, 1)).is_some());
    assert!(c.broadcast((32, 1, 2)).is_none());

    /* () can be broadcast to anything */
    let z = ArcArray::<f32, _>::zeros(());
    assert!(z.broadcast(()).is_some());
    assert!(z.broadcast(1).is_some());
    assert!(z.broadcast(3).is_some());
    assert!(z.broadcast((7, 2, 9)).is_some());
}

#[test]
#[cfg(feature = "std")]
fn test_add() {
    let a_dim = Dim([2, 4, 2, 2]);
    let b_dim = Dim([2, 1, 2, 1]);
    let mut a = ArcArray::linspace(0.0, 1., a_dim.size()).reshape(a_dim);
    let b = ArcArray::linspace(0.0, 1., b_dim.size()).reshape(b_dim);
    a += &b;
    let t = ArcArray::from_elem((), 1.0f32);
    a += &t;
}

#[test]
#[should_panic]
#[cfg(feature = "std")]
fn test_add_incompat() {
    let a_dim = Dim([2, 4, 2, 2]);
    let mut a = ArcArray::linspace(0.0, 1., a_dim.size()).reshape(a_dim);
    let incompat = ArcArray::from_elem(3, 1.0f32);
    a += &incompat;
}

#[test]
fn test_broadcast() {
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
fn test_broadcast_1d() {
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
