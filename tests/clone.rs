use ndarray::arr2;

#[test]
fn test_clone_from() {
    let a = arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    let b = arr2(&[[7, 7, 7]]);
    let mut c = b.clone();
    c.clone_from(&a);
    assert_eq!(a, c);

    let mut bv = b.view();
    bv.clone_from(&a.view());
    assert_eq!(&a, &bv);
}
