use ndarray::arr2;
use ndarray::ArcArray;

#[test]
fn test_swap() {
    let mut a = arr2(&[[(); 3]; 3]);

    let b = a.clone();

    for i in 0..a.nrows() {
        for j in i + 1..a.ncols() {
            a.swap((i, j), (j, i));
        }
    }
    assert_eq!(a, b.t());
}

#[test]
#[allow(clippy::all)] //not sure what lint keeps fixing this, never shows up with clippy , but changes with --fix
fn test() {
    let c = ArcArray::<(), _>::default((3, 4));
    let mut d = c.clone();
    for _ in d.iter_mut() {}
}
