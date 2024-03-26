#![warn(clippy::pedantic, clippy::nursery)]

use crate::extension::nonnull;
#[cfg(not(feature = "std"))]
use alloc::borrow::ToOwned;
use alloc::slice;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use std::mem;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;

use rawpointer::PointerExt;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Device {
    Host,
    #[cfg(feature = "opencl")]
    OpenCL,
    #[cfg(feature = "cuda")]
    CUDA,
}

/// Array's representation.
///
/// *Don’t use this type directly—use the type alias
/// [`Array`](crate::Array) for the array type!*
// Like a Vec, but with non-unique ownership semantics
//
// repr(C) to make it transmutable OwnedRepr<A> -> OwnedRepr<B> if
// transmutable A -> B.
#[derive(Debug)]
#[repr(C)]
pub struct OwnedRepr<A> {
    ptr: NonNull<A>,
    len: usize,
    capacity: usize,
    device: Device,
}

impl<A> OwnedRepr<A> {
    pub(crate) fn from(v: Vec<A>) -> Self {
        let mut v = ManuallyDrop::new(v);
        let len = v.len();
        let capacity = v.capacity();
        let ptr = nonnull::nonnull_from_vec_data(&mut v);
        let device = Device::Host;
        Self {
            ptr,
            len,
            capacity,
            device,
        }
    }

    /// Move this storage object to a specified device.
    #[allow(clippy::unnecessary_wraps)]
    pub(crate) fn copy_to_device(self, device: Device) -> Option<Self> {
        // println!("Copying to {device:?}");
        // let mut self_ = ManuallyDrop::new(self);
        // self_.device = device;

        let len = self.len;
        let capacity = self.capacity;

        match (self.device, device) {
            (Device::Host, Device::Host) => {
                // println!("Copying to Host");
                Some(self)
            }

            #[cfg(feature = "opencl")]
            (Device::Host, Device::OpenCL) => {
                let bytes = std::mem::size_of::<A>() * self.capacity;

                unsafe {
                    if let Ok(buffer) =
                        hasty_::opencl::opencl_allocate(bytes, hasty_::opencl::OpenCLMemoryType::ReadWrite)
                    {
                        // println!("Allocated OpenCL Buffer");
                        if let Ok(_) =
                            hasty_::opencl::opencl_write(buffer, self.ptr.as_ptr() as *const std::ffi::c_void, bytes)
                        {
                            // println!("Wrote to OpenCL Buffer");

                            Some(Self {
                                ptr: NonNull::new(buffer as *mut A)?,
                                len,
                                capacity,
                                device,
                            })
                        } else {
                            // println!("Failed to write to OpenCL Buffer");
                            None
                        }
                    } else {
                        // println!("Failed to allocate OpenCL Buffer");
                        None
                    }
                }
            }

            #[cfg(feature = "opencl")]
            (Device::OpenCL, Device::Host) => {
                let bytes = std::mem::size_of::<A>() * capacity;

                unsafe {
                    let mut data = ManuallyDrop::new(Vec::<A>::with_capacity(self.capacity));
                    data.set_len(self.len);
                    if let Ok(_) = hasty_::opencl::opencl_read(
                        data.as_mut_ptr() as *mut std::ffi::c_void,
                        self.ptr.as_ptr() as *mut std::ffi::c_void,
                        bytes,
                    ) {
                        Some(Self {
                            ptr: nonnull::nonnull_from_vec_data(&mut data),
                            len,
                            capacity,
                            device,
                        })
                    } else {
                        None
                    }
                }
            }

            #[cfg(feature = "opencl")]
            (Device::OpenCL, Device::OpenCL) => {
                todo!();
            }

            #[cfg(feature = "cuda")]
            (Device::Host, Device::CUDA) => {
                todo!();
            }

            #[cfg(feature = "cuda")]
            (Device::CUDA, Device::Host) => {
                todo!();
            }

            #[cfg(feature = "cuda")]
            (Device::CUDA, Device::CUDA) => {
                todo!();
            }

            #[cfg(all(feature = "opencl", feature = "cuda"))]
            (Device::OpenCL, Device::CUDA) => {
                todo!();
            }

            #[cfg(all(feature = "opencl", feature = "cuda"))]
            (Device::CUDA, Device::OpenCL) => {
                todo!();
            }
        }
    }

    /// Drop the object and free the memory
    pub(crate) unsafe fn drop_impl(&mut self) -> Vec<A> {
        let capacity = self.capacity;
        let len = self.len;
        self.len = 0;
        self.capacity = 0;
        let ptr = self.ptr.as_ptr();

        match self.device {
            Device::Host => unsafe {
                // println!("Dropping Host pointer");
                Vec::from_raw_parts(ptr, len, capacity)
            },

            #[cfg(feature = "opencl")]
            Device::OpenCL => {
                // Free `ptr`
                // println!("Freeing OpenCL pointer");

                hasty_::opencl::opencl_free(ptr as *mut std::ffi::c_void);

                // Should be optimised out, since nothing is allocated
                Vec::new()
            }

            #[cfg(feature = "cuda")]
            Device::CUDA => {
                // Free `ptr`
                println!("Freeing CUDA pointer");
                Vec::new()
            }
        }
    }

    /// Convert `self` into a [Vec].
    ///
    /// # Panics
    /// Will panic if the underlying memory is not allocated on
    /// the host device.
    pub(crate) fn into_vec(self) -> Vec<A> {
        // Creating a Vec requires the data to be on the host device
        assert_eq!(self.device, Device::Host);
        ManuallyDrop::new(self).take_as_vec()
    }

    /// Get a slice representation of `self`.
    ///
    /// # Panics
    /// Will panic if the underlying memory is not allocated
    /// on the host device.
    pub(crate) fn as_slice(&self) -> &[A] {
        // Cannot create a slice of a device pointer
        assert_eq!(self.device, Device::Host);
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub(crate) const fn len(&self) -> usize {
        self.len
    }

    /// Extract the raw underlying pointer from this object.
    ///
    /// ## Safety
    /// The pointer **may not necessarily point to the host**.
    /// Using a non-host pointer on the host will almost certainly
    /// cause a segmentation-fault.
    pub(crate) const fn as_ptr(&self) -> *const A {
        self.ptr.as_ptr()
    }

    /// Extract the raw underlying pointer from this object as mut
    ///
    /// ## Safety
    /// The pointer **may not necessarily point to the host**.
    /// Using a non-host pointer on the host will almost certainly
    /// cause a segmentation-fault.
    pub(crate) const fn as_ptr_mut(&self) -> *mut A {
        self.ptr.as_ptr()
    }

    /// Return underlying [`NonNull`] ptr.
    ///
    /// ## Safety
    /// The pointer **may not necessarily point to the host**.
    /// Using a non-host pointer on the host will almost certainly
    /// cause a segmentation-fault.
    pub(crate) fn as_nonnull_mut(&mut self) -> NonNull<A> {
        self.ptr
    }

    /// Return end pointer
    ///
    /// ## Safety
    /// The pointer **may not necessarily point to the host**.
    /// Using a non-host pointer on the host will almost certainly
    /// cause a segmentation-fault.
    pub(crate) fn as_end_nonnull(&self) -> NonNull<A> {
        unsafe { self.ptr.add(self.len) }
    }

    /// Reserve `additional` elements; return the new pointer
    ///
    /// ## Safety
    /// Note that existing pointers into the data are invalidated
    #[must_use = "must use new pointer to update existing pointers"]
    pub(crate) fn reserve(&mut self, additional: usize) -> NonNull<A> {
        self.modify_as_vec(|mut v| {
            v.reserve(additional);
            v
        });
        self.as_nonnull_mut()
    }

    /// Set the valid length of the data
    ///
    /// ## Safety
    /// The first `new_len` elements of the data should be valid.
    pub(crate) unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.capacity);
        self.len = new_len;
    }

    /// Return the length (number of elements in total) and set
    /// the internal length to zero.
    ///
    /// todo: Is this valid/safe? Mark as unsafe?
    pub(crate) fn release_all_elements(&mut self) -> usize {
        let ret = self.len;
        self.len = 0;
        ret
    }

    /// Cast self into equivalent repr of other element type
    ///
    /// ## Safety
    /// Caller must ensure the two types have the same representation.
    /// **Panics** if sizes don't match (which is not a sufficient check).
    pub(crate) unsafe fn data_subst<B>(self) -> OwnedRepr<B> {
        // necessary but not sufficient check
        assert_eq!(mem::size_of::<A>(), mem::size_of::<B>());
        let self_ = ManuallyDrop::new(self);
        OwnedRepr {
            ptr: self_.ptr.cast::<B>(),
            len: self_.len,
            capacity: self_.capacity,
            device: self_.device,
        }
    }

    /// Apply a `f(Vec<A>) -> Vec<A>` to this storage object and update `self`.
    fn modify_as_vec(&mut self, f: impl FnOnce(Vec<A>) -> Vec<A>) {
        let v = self.take_as_vec();
        *self = Self::from(f(v));
    }

    /// Take `self` as a `Vec` object. This invalidates `self`.
    ///
    /// # Panics
    /// Will panic if the underlying memory is not allocated
    /// on the host device.
    fn take_as_vec(&mut self) -> Vec<A> {
        assert_eq!(self.device, Device::Host);
        let capacity = self.capacity;
        let len = self.len;
        self.len = 0;
        self.capacity = 0;
        unsafe { Vec::from_raw_parts(self.ptr.as_ptr(), len, capacity) }
    }
}

impl<A> Clone for OwnedRepr<A>
where
    A: Clone,
{
    fn clone(&self) -> Self {
        match self.device {
            Device::Host => Self::from(self.as_slice().to_owned()),

            #[cfg(feature = "opencl")]
            Device::OpenCL => {
                println!("Performing OpenCL Clone");
                // todo: OpenCL clone
                Self::from(self.as_slice().to_owned())
            }

            #[cfg(feature = "cuda")]
            Device::CUDA => {
                println!("Performing CUDA Clone");
                // todo: CUDA clone
                Self::from(self.as_slice().to_owned())
            }
        }
    }

    fn clone_from(&mut self, other: &Self) {
        match self.device {
            Device::Host => {
                let mut v = self.take_as_vec();
                let other = other.as_slice();

                if v.len() > other.len() {
                    v.truncate(other.len());
                }
                let (front, back) = other.split_at(v.len());
                v.clone_from_slice(front);
                v.extend_from_slice(back);
                *self = Self::from(v);
            }

            #[cfg(feature = "opencl")]
            Device::OpenCL => {
                println!("Performing OpenCL Clone From");
                // todo: OpenCL clone from
                let mut v = self.take_as_vec();
                let other = other.as_slice();

                if v.len() > other.len() {
                    v.truncate(other.len());
                }
                let (front, back) = other.split_at(v.len());
                v.clone_from_slice(front);
                v.extend_from_slice(back);
                *self = Self::from(v);
            }

            #[cfg(feature = "cuda")]
            Device::CUDA => {
                println!("Performing CUDA Clone From");
                // todo: CUDA clone from
                let mut v = self.take_as_vec();
                let other = other.as_slice();

                if v.len() > other.len() {
                    v.truncate(other.len());
                }
                let (front, back) = other.split_at(v.len());
                v.clone_from_slice(front);
                v.extend_from_slice(back);
                *self = Self::from(v);
            }
        }
    }
}

impl<A> Drop for OwnedRepr<A> {
    fn drop(&mut self) {
        if self.capacity > 0 {
            // correct because: If the elements don't need dropping, an
            // empty Vec is ok. Only the Vec's allocation needs dropping.
            //
            // implemented because: in some places in ndarray
            // where A: Copy (hence does not need drop) we use uninitialized elements in
            // vectors. Setting the length to 0 avoids that the vector tries to
            // drop, slice or otherwise produce values of these elements.
            // (The details of the validity letting this happen with nonzero len, are
            // under discussion as of this writing.)
            if !mem::needs_drop::<A>() {
                self.len = 0;
            }
            // drop as a Vec.
            unsafe { self.drop_impl() };
        }
    }
}

unsafe impl<A> Sync for OwnedRepr<A> where A: Sync {}
unsafe impl<A> Send for OwnedRepr<A> where A: Send {}
