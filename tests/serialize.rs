#![feature(macro_rules)]

extern crate "rustc-serialize" as serialize;
extern crate ndarray;

use serialize::json;
use ndarray::{arr0, arr1, arr2, Array, Ix, S, Si};

#[test]
fn serial_many_dim()
{
    {
        let a = arr0::<f32>(2.72);
        println!("{:?}", a);
        let serial = json::encode(&a);
        println!("{:?}", serial);
        let res = json::decode::<Array<f32, _>>(&serial[]);
        println!("{:?}", res);
        assert_eq!(a, res.unwrap());
    }

    {
        let a = arr1::<f32>(&[2.72, 1., 2.]);
        println!("{:?}", a);
        let serial = json::encode(&a);
        println!("{:?}", serial);
        let res = json::decode::<Array<f32, _>>(&serial[]);
        println!("{:?}", res);
        assert_eq!(a, res.unwrap());
    }

    {
        let a = arr2::<f32>(&[&[3., 1., 2.2], &[3.1, 4., 7.]]);
        println!("{:?}", a);
        let serial = json::encode(&a);
        println!("{:?}", serial);
        let res = json::decode::<Array<f32, _>>(&serial[]);
        println!("{:?}", res);
        assert_eq!(a, res.unwrap());
        let text = r##"{"v":1,"dim":[2,3],"data":[3,1,2.2,3.1,4,7]}"##;
        let b = json::decode::<Array<f32, (Ix, Ix)>>(text);
        assert_eq!(a, b.unwrap());
    }


    {
        // Test a sliced array.
        let mut a = Array::range(0., 32.0_f32).reshape((2, 2, 2, 4));
        a.islice(&[Si(0, None, -1), S, S, Si(0, Some(2), 1)]);
        println!("{:?}", a);
        let serial = json::encode(&a);
        println!("{:?}", serial);
        let res = json::decode::<Array<f32, _>>(&serial[]);
        println!("{:?}", res);
        assert_eq!(a, res.unwrap());
    }
}

#[test]
fn serial_wrong_count()
{
    // one element too few
    let text = r##"{"v":1,"dim":[2,3],"data":[3,1,2.2,3.1,4]}"##;
    let arr = json::decode::<Array<f32, (Ix, Ix)>>(text);
    println!("{:?}", arr);
    assert!(arr.is_err());

    // future version
    let text = r##"{"v":200,"dim":[2,3],"data":[3,1,2.2,3.1,4,7]}"##;
    let arr = json::decode::<Array<f32, (Ix, Ix)>>(text);
    println!("{:?}", arr);
    assert!(arr.is_err());
}
