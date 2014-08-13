extern crate test;
extern crate ndarray;

use ndarray::Array;

use std::fmt;

fn test_oper(op: &str, a: &[f32], b: &[f32], c: &[f32])
{
    let aa = Array::from_slice(a);
    let bb = Array::from_slice(b);
    let cc = Array::from_slice(c);
    test_oper_arr(op, aa.clone(), bb.clone(), cc.clone());
    let dim = (2u, 2u);
    let aa = aa.reshape(dim);
    let bb = bb.reshape(dim);
    let cc = cc.reshape(dim);
    test_oper_arr(op, aa.clone(), bb.clone(), cc.clone());
    let dim = (1u, 2u, 1u, 2u);
    let aa = aa.reshape(dim);
    let bb = bb.reshape(dim);
    let cc = cc.reshape(dim);
    test_oper_arr(op, aa.clone(), bb.clone(), cc.clone());
}

fn test_oper_arr<A: Primitive + fmt::Show, D: ndarray::Dimension>
    (op: &str, mut aa: Array<A,D>, bb: Array<A, D>, cc: Array<A, D>)
{
    match op {
        "+" => {
            assert_eq!(aa + bb, cc);
            aa.iadd(&bb);
            assert_eq!(aa, cc);
        },
        "-" => {
            assert_eq!(aa - bb, cc);
            aa.isub(&bb);
            assert_eq!(aa, cc);
        },
        "*" => {
            assert_eq!(aa * bb, cc);
            aa.imul(&bb);
            assert_eq!(aa, cc);
        },
        "/" => {
            assert_eq!(aa / bb, cc);
            aa.idiv(&bb);
            assert_eq!(aa, cc);
        },
        "%" => {
            assert_eq!(aa % bb, cc);
            aa.irem(&bb);
            assert_eq!(aa, cc);
        },
        "neg" => {
            assert_eq!(-aa, cc);
            aa.ineg();
            assert_eq!(aa, cc);
        },
        _ => fail!()
    }
}

#[test]
fn operations()
{
    test_oper("+", [1.0,2.0,3.0,4.0], [0.0, 1.0, 2.0, 3.0], [1.0,3.0,5.0,7.0]);
    test_oper("-", [1.0,2.0,3.0,4.0], [0.0, 1.0, 2.0, 3.0], [1.0,1.0,1.0,1.0]);
    test_oper("*", [1.0,2.0,3.0,4.0], [0.0, 1.0, 2.0, 3.0], [0.0,2.0,6.0,12.0]);
    test_oper("/", [1.0,2.0,3.0,4.0], [1.0, 1.0, 2.0, 3.0], [1.0,2.0,3.0/2.0,4.0/3.0]);
    test_oper("%", [1.0,2.0,3.0,4.0], [1.0, 1.0, 2.0, 3.0], [0.0,0.0,1.0,1.0]);
    test_oper("neg", [1.0,2.0,3.0,4.0], [1.0, 1.0, 2.0, 3.0], [-1.0,-2.0,-3.0,-4.0]);
}
