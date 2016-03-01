
//! The data (inner representation) traits for ndarray

use std::mem;
use std::rc::Rc;

use {
    ArrayBase,
    Dimension,
    ViewRepr,
};

/// Array’s inner representation.
///
/// ***Note:*** `Data` is not an extension interface at this point.
/// Traits in Rust can serve many different roles. This trait is public because
/// it is used as a bound on public methods.
pub unsafe trait Data {
    type Elem;
    fn slice(&self) -> &[Self::Elem];
}

/// Array’s writable inner representation.
pub unsafe trait DataMut : Data {
    fn slice_mut(&mut self) -> &mut [Self::Elem];
    #[inline]
    fn ensure_unique<D>(&mut ArrayBase<Self, D>)
        where Self: Sized,
              D: Dimension
    { }

    #[inline]
    fn is_unique(&mut self) -> bool {
        true
    }
}

/// Clone an Array’s storage.
pub unsafe trait DataClone : Data {
    /// Unsafe because, `ptr` must point inside the current storage.
    unsafe fn clone_with_ptr(&self, ptr: *mut Self::Elem) -> (Self, *mut Self::Elem);
}

unsafe impl<A> Data for Rc<Vec<A>> {
    type Elem = A;
    fn slice(&self) -> &[A] {
        self
    }
}

// NOTE: Copy on write
unsafe impl<A> DataMut for Rc<Vec<A>>
    where A: Clone
{
    fn slice_mut(&mut self) -> &mut [A] {
        &mut Rc::make_mut(self)[..]
    }

    fn ensure_unique<D>(self_: &mut ArrayBase<Self, D>)
        where Self: Sized,
              D: Dimension
    {
        if Rc::get_mut(&mut self_.data).is_some() {
            return;
        }
        if self_.dim.size() <= self_.data.len() / 2 {
            // Create a new vec if the current view is less than half of
            // backing data.
            unsafe {
                *self_ = ArrayBase::from_vec_dim_unchecked(self_.dim.clone(),
                                                           self_.iter()
                                                            .cloned()
                                                            .collect());
            }
            return;
        }
        let our_off = (self_.ptr as isize - self_.data.as_ptr() as isize) /
                      mem::size_of::<A>() as isize;
        let rvec = Rc::make_mut(&mut self_.data);
        unsafe {
            self_.ptr = rvec.as_mut_ptr().offset(our_off);
        }
    }

    fn is_unique(&mut self) -> bool {
        Rc::get_mut(self).is_some()
    }
}

unsafe impl<A> DataClone for Rc<Vec<A>> {
    unsafe fn clone_with_ptr(&self, ptr: *mut Self::Elem) -> (Self, *mut Self::Elem) {
        // pointer is preserved
        (self.clone(), ptr)
    }
}

unsafe impl<A> Data for Vec<A> {
    type Elem = A;
    fn slice(&self) -> &[A] {
        self
    }
}

unsafe impl<A> DataMut for Vec<A> {
    fn slice_mut(&mut self) -> &mut [A] {
        self
    }
}

unsafe impl<A> DataClone for Vec<A>
    where A: Clone
{
    unsafe fn clone_with_ptr(&self, ptr: *mut Self::Elem) -> (Self, *mut Self::Elem) {
        let mut u = self.clone();
        let our_off = (self.as_ptr() as isize - ptr as isize) /
                      mem::size_of::<A>() as isize;
        let new_ptr = u.as_mut_ptr().offset(our_off);
        (u, new_ptr)
    }
}

unsafe impl<'a, A> Data for ViewRepr<&'a A> {
    type Elem = A;
    fn slice(&self) -> &[A] {
        &[]
    }
}

unsafe impl<'a, A> DataClone for ViewRepr<&'a A> {
    unsafe fn clone_with_ptr(&self, ptr: *mut Self::Elem) -> (Self, *mut Self::Elem) {
        (*self, ptr)
    }
}

unsafe impl<'a, A> Data for ViewRepr<&'a mut A> {
    type Elem = A;
    fn slice(&self) -> &[A] {
        &[]
    }
}

unsafe impl<'a, A> DataMut for ViewRepr<&'a mut A> {
    fn slice_mut(&mut self) -> &mut [A] {
        &mut []
    }
}

/// Array representation that is a unique or shared owner of its data.
pub unsafe trait DataOwned : Data {
    fn new(elements: Vec<Self::Elem>) -> Self;
    fn into_shared(self) -> Rc<Vec<Self::Elem>>;
}

/// Array representation that is a lightweight view.
pub unsafe trait DataShared : Clone + DataClone { }

unsafe impl<A> DataShared for Rc<Vec<A>> {}
unsafe impl<'a, A> DataShared for ViewRepr<&'a A> {}

unsafe impl<A> DataOwned for Vec<A> {
    fn new(elements: Vec<A>) -> Self {
        elements
    }
    fn into_shared(self) -> Rc<Vec<A>> {
        Rc::new(self)
    }
}

unsafe impl<A> DataOwned for Rc<Vec<A>> {
    fn new(elements: Vec<A>) -> Self {
        Rc::new(elements)
    }
    fn into_shared(self) -> Rc<Vec<A>> {
        self
    }
}

