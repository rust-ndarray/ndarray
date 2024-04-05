extern crate ndarray;

extern crate serde;

extern crate serde_json;

extern crate rmp_serde;

#[cfg(feature = "ron")]
extern crate ron;

extern crate borsh;

use ndarray::{arr0, arr1, arr2, s, ArcArray, ArcArray2, ArrayD, IxDyn};

#[test]
fn serial_many_dim_serde()
{
    {
        let a = arr0::<f32>(2.72);
        let serial = serde_json::to_string(&a).unwrap();
        println!("Serde encode {:?} => {:?}", a, serial);
        let res = serde_json::from_str::<ArcArray<f32, _>>(&serial);
        println!("{:?}", res);
        assert_eq!(a, res.unwrap());
    }

    {
        let a = arr1::<f32>(&[2.72, 1., 2.]);
        let serial = serde_json::to_string(&a).unwrap();
        println!("Serde encode {:?} => {:?}", a, serial);
        let res = serde_json::from_str::<ArcArray<f32, _>>(&serial);
        println!("{:?}", res);
        assert_eq!(a, res.unwrap());
    }

    {
        let a = arr2(&[[3., 1., 2.2], [3.1, 4., 7.]]);
        let serial = serde_json::to_string(&a).unwrap();
        println!("Serde encode {:?} => {:?}", a, serial);
        let res = serde_json::from_str::<ArcArray<f32, _>>(&serial);
        println!("{:?}", res);
        assert_eq!(a, res.unwrap());
        let text = r##"{"v":1,"dim":[2,3],"data":[3,1,2.2,3.1,4,7]}"##;
        let b = serde_json::from_str::<ArcArray<f32, _>>(text);
        assert_eq!(a, b.unwrap());
    }

    {
        // Test a sliced array.
        let mut a = ArcArray::linspace(0., 31., 32)
            .into_shape_with_order((2, 2, 2, 4))
            .unwrap();
        a.slice_collapse(s![..;-1, .., .., ..2]);
        let serial = serde_json::to_string(&a).unwrap();
        println!("Encode {:?} => {:?}", a, serial);
        let res = serde_json::from_str::<ArcArray<f32, _>>(&serial);
        println!("{:?}", res);
        assert_eq!(a, res.unwrap());
    }
}

#[test]
fn serial_ixdyn_serde()
{
    {
        let a = arr0::<f32>(2.72).into_dyn();
        let serial = serde_json::to_string(&a).unwrap();
        println!("Serde encode {:?} => {:?}", a, serial);
        let res = serde_json::from_str::<ArcArray<f32, _>>(&serial);
        println!("{:?}", res);
        assert_eq!(a, res.unwrap());
    }

    {
        let a = arr1::<f32>(&[2.72, 1., 2.]).into_dyn();
        let serial = serde_json::to_string(&a).unwrap();
        println!("Serde encode {:?} => {:?}", a, serial);
        let res = serde_json::from_str::<ArrayD<f32>>(&serial);
        println!("{:?}", res);
        assert_eq!(a, res.unwrap());
    }

    {
        let a = arr2(&[[3., 1., 2.2], [3.1, 4., 7.]])
            .into_shape_with_order(IxDyn(&[3, 1, 1, 1, 2, 1]))
            .unwrap();
        let serial = serde_json::to_string(&a).unwrap();
        println!("Serde encode {:?} => {:?}", a, serial);
        let res = serde_json::from_str::<ArrayD<f32>>(&serial);
        println!("{:?}", res);
        assert_eq!(a, res.unwrap());
    }

    {
        let a = arr2(&[[3., 1., 2.2], [3.1, 4., 7.]]).into_dyn();
        let text = r##"{"v":1,"dim":[2,3],"data":[3,1,2.2,3.1,4,7]}"##;
        let b = serde_json::from_str::<ArrayD<f32>>(text);
        assert_eq!(a, b.unwrap());
    }
}

#[test]
fn serial_wrong_count_serde()
{
    // one element too few
    let text = r##"{"v":1,"dim":[2,3],"data":[3,1,2.2,3.1,4]}"##;
    let arr = serde_json::from_str::<ArcArray2<f32>>(text);
    println!("{:?}", arr);
    assert!(arr.is_err());

    // future version
    let text = r##"{"v":200,"dim":[2,3],"data":[3,1,2.2,3.1,4,7]}"##;
    let arr = serde_json::from_str::<ArcArray2<f32>>(text);
    println!("{:?}", arr);
    assert!(arr.is_err());
}

#[test]
fn serial_many_dim_serde_msgpack()
{
    {
        let a = arr0::<f32>(2.72);

        let mut buf = Vec::new();
        serde::Serialize::serialize(&a, &mut rmp_serde::Serializer::new(&mut buf))
            .ok()
            .unwrap();

        let mut deserializer = rmp_serde::Deserializer::new(&buf[..]);
        let a_de: ArcArray<f32, _> = serde::Deserialize::deserialize(&mut deserializer).unwrap();

        assert_eq!(a, a_de);
    }

    {
        let a = arr1::<f32>(&[2.72, 1., 2.]);

        let mut buf = Vec::new();
        serde::Serialize::serialize(&a, &mut rmp_serde::Serializer::new(&mut buf))
            .ok()
            .unwrap();

        let mut deserializer = rmp_serde::Deserializer::new(&buf[..]);
        let a_de: ArcArray<f32, _> = serde::Deserialize::deserialize(&mut deserializer).unwrap();

        assert_eq!(a, a_de);
    }

    {
        let a = arr2(&[[3., 1., 2.2], [3.1, 4., 7.]]);

        let mut buf = Vec::new();
        serde::Serialize::serialize(&a, &mut rmp_serde::Serializer::new(&mut buf))
            .ok()
            .unwrap();

        let mut deserializer = rmp_serde::Deserializer::new(&buf[..]);
        let a_de: ArcArray<f32, _> = serde::Deserialize::deserialize(&mut deserializer).unwrap();

        assert_eq!(a, a_de);
    }

    {
        // Test a sliced array.
        let mut a = ArcArray::linspace(0., 31., 32)
            .into_shape_with_order((2, 2, 2, 4))
            .unwrap();
        a.slice_collapse(s![..;-1, .., .., ..2]);

        let mut buf = Vec::new();
        serde::Serialize::serialize(&a, &mut rmp_serde::Serializer::new(&mut buf))
            .ok()
            .unwrap();

        let mut deserializer = rmp_serde::Deserializer::new(&buf[..]);
        let a_de: ArcArray<f32, _> = serde::Deserialize::deserialize(&mut deserializer).unwrap();

        assert_eq!(a, a_de);
    }
}

#[test]
#[cfg(feature = "ron")]
fn serial_many_dim_ron()
{
    use ron::de::from_str as ron_deserialize;
    use ron::ser::to_string as ron_serialize;

    {
        let a = arr0::<f32>(2.72);

        let a_s = ron_serialize(&a).unwrap();

        let a_de: ArcArray<f32, _> = ron_deserialize(&a_s).unwrap();

        assert_eq!(a, a_de);
    }

    {
        let a = arr1::<f32>(&[2.72, 1., 2.]);

        let a_s = ron_serialize(&a).unwrap();

        let a_de: ArcArray<f32, _> = ron_deserialize(&a_s).unwrap();

        assert_eq!(a, a_de);
    }

    {
        let a = arr2(&[[3., 1., 2.2], [3.1, 4., 7.]]);

        let a_s = ron_serialize(&a).unwrap();

        let a_de: ArcArray<f32, _> = ron_deserialize(&a_s).unwrap();

        assert_eq!(a, a_de);
    }

    {
        // Test a sliced array.
        let mut a = ArcArray::linspace(0., 31., 32)
            .into_shape_with_order((2, 2, 2, 4))
            .unwrap();
        a.slice_collapse(s![..;-1, .., .., ..2]);

        let a_s = ron_serialize(&a).unwrap();

        let a_de: ArcArray<f32, _> = ron_deserialize(&a_s).unwrap();

        assert_eq!(a, a_de);
    }
}

#[test]
fn serial_ixdyn_borsh() {
    {
        let a = arr0::<f32>(2.72).into_dyn();
        let serial = borsh::to_vec(&a).unwrap();
        println!("Borsh encode {:?} => {:?}", a, serial);
        let res = borsh::from_slice::<ArcArray<f32, _>>(&serial);
        println!("{:?}", res);
        assert_eq!(a, res.unwrap());
    }

    {
        let a = arr1::<f32>(&[2.72, 1., 2.]).into_dyn();
        let serial = borsh::to_vec(&a).unwrap();
        println!("Borsh encode {:?} => {:?}", a, serial);
        let res = borsh::from_slice::<ArrayD<f32>>(&serial);
        println!("{:?}", res);
        assert_eq!(a, res.unwrap());
    }

    {
        let a = arr2(&[[3., 1., 2.2], [3.1, 4., 7.]])
            .into_shape(IxDyn(&[3, 1, 1, 1, 2, 1]))
            .unwrap();
        let serial = borsh::to_vec(&a).unwrap();
        println!("Borsh encode {:?} => {:?}", a, serial);
        let res = borsh::from_slice::<ArrayD<f32>>(&serial);
        println!("{:?}", res);
        assert_eq!(a, res.unwrap());
    }
}

#[test]
fn serial_many_dim_borsh() {
    use borsh::from_slice as borsh_deserialize;
    use borsh::to_vec as borsh_serialize;

    {
        let a = arr0::<f32>(2.72);
        let a_s = borsh_serialize(&a).unwrap();
        let a_de: ArcArray<f32, _> = borsh_deserialize(&a_s).unwrap();
        assert_eq!(a, a_de);
    }

    {
        let a = arr1::<f32>(&[2.72, 1., 2.]);
        let a_s = borsh_serialize(&a).unwrap();
        let a_de: ArcArray<f32, _> = borsh_deserialize(&a_s).unwrap();
        assert_eq!(a, a_de);
    }

    {
        let a = arr2(&[[3., 1., 2.2], [3.1, 4., 7.]]);
        let a_s = borsh_serialize(&a).unwrap();
        let a_de: ArcArray<f32, _> = borsh_deserialize(&a_s).unwrap();
        assert_eq!(a, a_de);
    }

    {
        // Test a sliced array.
        let mut a = ArcArray::linspace(0., 31., 32).reshape((2, 2, 2, 4));
        a.slice_collapse(s![..;-1, .., .., ..2]);
        let a_s = borsh_serialize(&a).unwrap();
        let a_de: ArcArray<f32, _> = borsh_deserialize(&a_s).unwrap();
        assert_eq!(a, a_de);
    }
}
