use core::ptr::NonNull;
use std::marker::PhantomData;

use dlpark::prelude::*;

use crate::{ArrayBase, Dimension, IntoDimension, IxDyn, ManagedArray, RawData};

impl<A, S, D> ToTensor for ArrayBase<S, D>
where
    A: InferDtype,
    S: RawData<Elem = A>,
    D: Dimension,
{
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.as_ptr() as *mut std::ffi::c_void
    }

    fn byte_offset(&self) -> u64 {
        0
    }

    fn device(&self) -> Device {
        Device::CPU
    }

    fn dtype(&self) -> DataType {
        A::infer_dtype()
    }

    fn shape(&self) -> CowIntArray {
        dlpark::prelude::CowIntArray::from_owned(
            self.shape().into_iter().map(|&x| x as i64).collect(),
        )
    }

    fn strides(&self) -> Option<CowIntArray> {
        Some(dlpark::prelude::CowIntArray::from_owned(
            self.strides().into_iter().map(|&x| x as i64).collect(),
        ))
    }
}

pub struct ManagedRepr<A> {
    managed_tensor: ManagedTensor,
    _ty: PhantomData<A>,
}

impl<A> ManagedRepr<A> {
    pub fn new(managed_tensor: ManagedTensor) -> Self {
        Self {
            managed_tensor,
            _ty: PhantomData,
        }
    }

    pub fn as_slice(&self) -> &[A] {
        self.managed_tensor.as_slice()
    }

    pub fn as_ptr(&self) -> *const A {
        self.managed_tensor.data_ptr() as *const A
    }
}

unsafe impl<A> Sync for ManagedRepr<A> where A: Sync {}
unsafe impl<A> Send for ManagedRepr<A> where A: Send {}

impl<A> FromDLPack for ManagedArray<A, IxDyn> {
    fn from_dlpack(dlpack: NonNull<dlpark::ffi::DLManagedTensor>) -> Self {
        let managed_tensor = ManagedTensor::new(dlpack);
        let shape: Vec<usize> = managed_tensor
            .shape()
            .into_iter()
            .map(|x| *x as _)
            .collect();

        let strides: Vec<usize> = match (managed_tensor.strides(), managed_tensor.is_contiguous()) {
            (Some(s), _) => s.into_iter().map(|&x| x as _).collect(),
            (None, true) => managed_tensor
                .calculate_contiguous_strides()
                .into_iter()
                .map(|x| x as _)
                .collect(),
            (None, false) => panic!("dlpack: invalid strides"),
        };
        let ptr = managed_tensor.data_ptr() as *mut A;

        let managed_repr = ManagedRepr::<A>::new(managed_tensor);
        unsafe {
            ArrayBase::from_data_ptr(managed_repr, NonNull::new_unchecked(ptr))
                .with_strides_dim(strides.into_dimension(), shape.into_dimension())
        }
    }
}
