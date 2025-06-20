use ndarray::{arr0, arr1, arr2, s, ArcArray, ArrayBase, Dim, IxDyn, IxDynImpl, OwnedArcRepr, OwnedRepr};
// No test: ArcArray2, ArrayD

#[test]
fn store_many_dim_excrate() {
    {
        let a = arr0::<f32>(2.72);
        let store_bytes = bincode::encode_to_vec(&a, bincode::config::standard()).unwrap();
        println!("Bincode encode {:?} => {:?}", &a, store_bytes);
        let res = bincode::decode_from_slice::<ArrayBase<OwnedRepr<f32>, Dim<[usize; 0]>>, _>(
            &store_bytes,
            bincode::config::standard(),
        );
        println!("{:?}", res);
        assert_eq!(a, res.unwrap().0);
    }

    {
        let a = arr1::<f32>(&[2.72, 1., 2.]);
        let store_bytes = bincode::encode_to_vec(&a, bincode::config::standard()).unwrap();
        println!("Bincode encode {:?} => {:?}", &a, store_bytes);
        let res = bincode::decode_from_slice::<ndarray::ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>, _>(
            &store_bytes,
            bincode::config::standard(),
        );
        println!("{:?}", res);
        assert_eq!(a, res.unwrap().0);
    }

    {
        let a = arr2(&[[3., 1., 2.2], [3.1, 4., 7.]]);
        let store_bytes = bincode::encode_to_vec(&a, bincode::config::standard()).unwrap();
        println!("Bincode encode {:?} => {:?}", &a, store_bytes);
        let res = bincode::decode_from_slice::<ndarray::ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>, _>(
            &store_bytes,
            bincode::config::standard(),
        );
        println!("{:?}", res);
        assert_eq!(a, res.unwrap().0);
    }

    {
        // Test a sliced array.
        let mut a = ArcArray::from_iter(0..32)
            .into_shape_with_order((2, 2, 2, 4))
            .unwrap();
        a.slice_collapse(s![..;-1, .., .., ..2]);
        let store_bytes = bincode::encode_to_vec(&a, bincode::config::standard()).unwrap();
        println!("Bincode encode {:?} => {:?}", &a, store_bytes);
        let res = bincode::decode_from_slice::<ndarray::ArrayBase<OwnedArcRepr<i32>, Dim<[usize; 4]>>, _>(
            &store_bytes,
            bincode::config::standard(),
        );
        println!("{:?}", res);
        assert_eq!(a, res.unwrap().0);
    }
}

#[test]
fn serial_ixdyn_serde() {
    {
        let a = arr0::<f32>(2.72).into_dyn();
        let store_bytes = bincode::encode_to_vec(&a, bincode::config::standard()).unwrap();
        println!("Bincode encode {:?} => {:?}", &a, store_bytes);
        let res = bincode::decode_from_slice::<ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>, _>(
            &store_bytes,
            bincode::config::standard(),
        );
        println!("{:?}", res);
        assert_eq!(a, res.unwrap().0);
    }

    {
        let a = arr1::<f32>(&[2.72, 1., 2.]).into_dyn();
        let store_bytes = bincode::encode_to_vec(&a, bincode::config::standard()).unwrap();
        println!("Bincode encode {:?} => {:?}", &a, store_bytes);
        let res = bincode::decode_from_slice::<ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>, _>(
            &store_bytes,
            bincode::config::standard(),
        );
        println!("{:?}", res);
        assert_eq!(a, res.unwrap().0);
    }

    {
        let a = arr2(&[[3., 1., 2.2], [3.1, 4., 7.]])
            .into_shape_with_order(IxDyn(&[3, 1, 1, 1, 2, 1]))
            .unwrap();
        let store_bytes = bincode::encode_to_vec(&a, bincode::config::standard()).unwrap();
        println!("Bincode encode {:?} => {:?}", &a, store_bytes);
        let res = bincode::decode_from_slice::<ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>, _>(
            &store_bytes,
            bincode::config::standard(),
        );
        println!("{:?}", res);
        assert_eq!(a, res.unwrap().0);
    }
}
