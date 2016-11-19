#![cfg(any(feature = "rustc-serialize", feature = "serde"))]
#[cfg(feature = "rustc-serialize")]
extern crate rustc_serialize as serialize;

#[macro_use] extern crate ndarray;

#[cfg(feature = "serde")]
extern crate serde;

#[cfg(feature = "serde")]
extern crate serde_json;

#[cfg(feature = "serde")]
extern crate rmp_serde;

#[cfg(feature = "rustc-serialize")]
use serialize::json;

use ndarray::{arr0, arr1, arr2, RcArray, RcArray1, RcArray2};

#[cfg(feature = "rustc-serialize")]
#[test]
fn serial_many_dim()
{
    {
        let a = arr0::<f32>(2.72);
        let serial = json::encode(&a).unwrap();
        println!("Encode {:?} => {:?}", a, serial);
        let res = json::decode::<RcArray<f32, _>>(&serial);
        println!("{:?}", res);
        assert_eq!(a, res.unwrap());
    }

    {
        let a = arr1::<f32>(&[2.72, 1., 2.]);
        let serial = json::encode(&a).unwrap();
        println!("Encode {:?} => {:?}", a, serial);
        let res = json::decode::<RcArray1<f32>>(&serial);
        println!("{:?}", res);
        assert_eq!(a, res.unwrap());
    }

    {
        let a = arr2(&[[3., 1., 2.2], [3.1, 4., 7.]]);
        let serial = json::encode(&a).unwrap();
        println!("Encode {:?} => {:?}", a, serial);
        let res = json::decode::<RcArray2<f32>>(&serial);
        println!("{:?}", res);
        assert_eq!(a, res.unwrap());
        let text = r##"{"v":1,"dim":[2,3],"data":[3,1,2.2,3.1,4,7]}"##;
        let b = json::decode::<RcArray2<f32>>(text);
        assert_eq!(a, b.unwrap());
    }


    {
        // Test a sliced array.
        let mut a = RcArray::linspace(0., 31., 32).reshape((2, 2, 2, 4));
        a.islice(s![..;-1, .., .., ..2]);
        let serial = json::encode(&a).unwrap();
        println!("Encode {:?} => {:?}", a, serial);
        let res = json::decode::<RcArray<f32, _>>(&serial);
        println!("{:?}", res);
        assert_eq!(a, res.unwrap());
    }
}

#[cfg(feature = "rustc-serialize")]
#[test]
fn serial_wrong_count()
{
    // one element too few
    let text = r##"{"v":1,"dim":[2,3],"data":[3,1,2.2,3.1,4]}"##;
    let arr = json::decode::<RcArray2<f32>>(text);
    println!("{:?}", arr);
    assert!(arr.is_err());

    // future version
    let text = r##"{"v":200,"dim":[2,3],"data":[3,1,2.2,3.1,4,7]}"##;
    let arr = json::decode::<RcArray2<f32>>(text);
    println!("{:?}", arr);
    assert!(arr.is_err());
}

#[cfg(feature = "serde")]
#[test]
fn serial_many_dim_serde()
{
    {
        let a = arr0::<f32>(2.72);
        let serial = serde_json::to_string(&a).unwrap();
        println!("Serde encode {:?} => {:?}", a, serial);
        let res = serde_json::from_str::<RcArray<f32, _>>(&serial);
        println!("{:?}", res);
        assert_eq!(a, res.unwrap());
    }

    {
        let a = arr1::<f32>(&[2.72, 1., 2.]);
        let serial = serde_json::to_string(&a).unwrap();
        println!("Serde encode {:?} => {:?}", a, serial);
        let res = serde_json::from_str::<RcArray1<f32>>(&serial);
        println!("{:?}", res);
        assert_eq!(a, res.unwrap());
    }

    {
        let a = arr2(&[[3., 1., 2.2], [3.1, 4., 7.]]);
        let serial = serde_json::to_string(&a).unwrap();
        println!("Serde encode {:?} => {:?}", a, serial);
        let res = serde_json::from_str::<RcArray2<f32>>(&serial);
        println!("{:?}", res);
        assert_eq!(a, res.unwrap());
        let text = r##"{"v":1,"dim":[2,3],"data":[3,1,2.2,3.1,4,7]}"##;
        let b = serde_json::from_str::<RcArray2<f32>>(text);
        assert_eq!(a, b.unwrap());
    }

    {
        // Test a sliced array.
        let mut a = RcArray::linspace(0., 31., 32).reshape((2, 2, 2, 4));
        a.islice(s![..;-1, .., .., ..2]);
        let serial = serde_json::to_string(&a).unwrap();
        println!("Encode {:?} => {:?}", a, serial);
        let res = serde_json::from_str::<RcArray<f32, _>>(&serial);
        println!("{:?}", res);
        assert_eq!(a, res.unwrap());
    }
}

#[cfg(feature = "serde")]
#[test]
fn serial_wrong_count_serde()
{
    // one element too few
    let text = r##"{"v":1,"dim":[2,3],"data":[3,1,2.2,3.1,4]}"##;
    let arr = serde_json::from_str::<RcArray2<f32>>(text);
    println!("{:?}", arr);
    assert!(arr.is_err());

    // future version
    let text = r##"{"v":200,"dim":[2,3],"data":[3,1,2.2,3.1,4,7]}"##;
    let arr = serde_json::from_str::<RcArray2<f32>>(text);
    println!("{:?}", arr);
    assert!(arr.is_err());
}

#[cfg(feature = "serde")]
#[test]
fn serial_many_dim_serde_msgpack()
{
    {
        let a = arr0::<f32>(2.72);

        let mut buf = Vec::new();
        serde::Serialize::serialize(&a, &mut rmp_serde::Serializer::new(&mut buf)).ok().unwrap();

        let mut deserializer = rmp_serde::Deserializer::new(&buf[..]);
        let a_de: RcArray<f32, _> = serde::Deserialize::deserialize(&mut deserializer).unwrap();

        assert_eq!(a, a_de);
    }

    {
        let a = arr1::<f32>(&[2.72, 1., 2.]);

        let mut buf = Vec::new();
        serde::Serialize::serialize(&a, &mut rmp_serde::Serializer::new(&mut buf)).ok().unwrap();

        let mut deserializer = rmp_serde::Deserializer::new(&buf[..]);
        let a_de: RcArray<f32, _> = serde::Deserialize::deserialize(&mut deserializer).unwrap();

        assert_eq!(a, a_de);
    }

    {
        let a = arr2(&[[3., 1., 2.2], [3.1, 4., 7.]]);

        let mut buf = Vec::new();
        serde::Serialize::serialize(&a, &mut rmp_serde::Serializer::new(&mut buf)).ok().unwrap();

        let mut deserializer = rmp_serde::Deserializer::new(&buf[..]);
        let a_de: RcArray<f32, _> = serde::Deserialize::deserialize(&mut deserializer).unwrap();

        assert_eq!(a, a_de);
    }

    {
        // Test a sliced array.
        let mut a = RcArray::linspace(0., 31., 32).reshape((2, 2, 2, 4));
        a.islice(s![..;-1, .., .., ..2]);

        let mut buf = Vec::new();
        serde::Serialize::serialize(&a, &mut rmp_serde::Serializer::new(&mut buf)).ok().unwrap();

        let mut deserializer = rmp_serde::Deserializer::new(&buf[..]);
        let a_de: RcArray<f32, _> = serde::Deserialize::deserialize(&mut deserializer).unwrap();

        assert_eq!(a, a_de);
    }
}
