use crate::{Array, ArrayBase, DataMut, Dimension, IntoNdProducer, NdProducer, Zip};
use crate::AssignElem;

use crate::parallel::prelude::*;
use crate::parallel::par::ParallelSplits;
use super::send_producer::SendProducer;

/// # Parallel methods
///
/// These methods require crate feature `rayon`.
impl<A, S, D> ArrayBase<S, D>
where
    S: DataMut<Elem = A>,
    D: Dimension,
    A: Send + Sync,
{
    /// Parallel version of `map_inplace`.
    ///
    /// Modify the array in place by calling `f` by mutable reference on each element.
    ///
    /// Elements are visited in arbitrary order.
    pub fn par_map_inplace<F>(&mut self, f: F)
    where
        F: Fn(&mut A) + Sync + Send,
    {
        self.view_mut().into_par_iter().for_each(f)
    }

    /// Parallel version of `mapv_inplace`.
    ///
    /// Modify the array in place by calling `f` by **v**alue on each element.
    /// The array is updated with the new values.
    ///
    /// Elements are visited in arbitrary order.
    pub fn par_mapv_inplace<F>(&mut self, f: F)
    where
        F: Fn(A) -> A + Sync + Send,
        A: Clone,
    {
        self.view_mut()
            .into_par_iter()
            .for_each(move |x| *x = f(x.clone()))
    }
}

// Zip

const COLLECT_MAX_SPLITS: usize = 10;

macro_rules! zip_impl {
    ($([$notlast:ident $($p:ident)*],)+) => {
        $(
        #[allow(non_snake_case)]
        impl<D, $($p),*> Zip<($($p,)*), D>
            where $($p::Item : Send , )*
                  $($p : Send , )*
                  D: Dimension,
                  $($p: NdProducer<Dim=D> ,)*
        {
            /// The `par_apply` method for `Zip`.
            ///
            /// This is a shorthand for using `.into_par_iter().for_each()` on
            /// `Zip`.
            ///
            /// Requires crate feature `rayon`.
            pub fn par_apply<F>(self, function: F)
                where F: Fn($($p::Item),*) + Sync + Send
            {
                self.into_par_iter().for_each(move |($($p,)*)| function($($p),*))
            }

            expand_if!(@bool [$notlast]

            /// Apply and collect the results into a new array, which has the same size as the
            /// inputs.
            ///
            /// If all inputs are c- or f-order respectively, that is preserved in the output.
            pub fn par_apply_collect<R>(self, f: impl Fn($($p::Item,)* ) -> R + Sync + Send)
                -> Array<R, D>
                where R: Send
            {
                let mut output = self.uninitalized_for_current_layout::<R>();
                let total_len = output.len();

                // Create a parallel iterator that produces chunks of the zip with the output
                // array.  It's crucial that both parts split in the same way, and in a way
                // so that the chunks of the output are still contig.
                //
                // Use a raw view so that we can alias the output data here and in the partial
                // result.
                let splits = unsafe {
                    ParallelSplits {
                        iter: self.and(SendProducer::new(output.raw_view_mut().cast::<R>())),
                        // Keep it from splitting the Zip down too small
                        max_splits: COLLECT_MAX_SPLITS,
                    }
                };

                let collect_result = splits.map(move |zip| {
                    // Create a partial result for the contiguous slice of data being written to
                    let output = zip.last_producer();
                    debug_assert!(output.is_contiguous());
                    let mut partial;
                    unsafe {
                        partial = Partial::new(output.as_ptr());
                    }

                    // Apply the mapping function on this chunk of the zip
                    let partial_len = &mut partial.len;
                    let f = &f;
                    zip.apply(move |$($p,)* output_elem: *mut R| unsafe {
                        output_elem.write(f($($p),*));
                        if std::mem::needs_drop::<R>() {
                            *partial_len += 1;
                        }
                    });

                    partial
                })
                .reduce(Partial::stub, Partial::try_merge);

                if std::mem::needs_drop::<R>() {
                    debug_assert_eq!(total_len, collect_result.len,
                        "collect len is not correct, expected {}", total_len);
                    assert!(collect_result.len == total_len,
                        "Collect: Expected number of writes not completed");
                }

                // Here the collect result is complete, and we release its ownership and transfer
                // it to the output array.
                collect_result.release_ownership();
                unsafe {
                    output.assume_init()
                }
            }

            /// Apply and assign the results into the producer `into`, which should have the same
            /// size as the other inputs.
            ///
            /// The producer should have assignable items as dictated by the `AssignElem` trait,
            /// for example `&mut R`.
            pub fn par_apply_assign_into<R, Q>(self, into: Q, f: impl Fn($($p::Item,)* ) -> R + Sync + Send)
                where Q: IntoNdProducer<Dim=D>,
                      Q::Item: AssignElem<R> + Send,
                      Q::Output: Send,
            {
                self.and(into)
                    .par_apply(move |$($p, )* output_| {
                        output_.assign_elem(f($($p ),*));
                    });
            }
            );
        }
        )+
    }
}

zip_impl! {
    [true P1],
    [true P1 P2],
    [true P1 P2 P3],
    [true P1 P2 P3 P4],
    [true P1 P2 P3 P4 P5],
    [false P1 P2 P3 P4 P5 P6],
}

/// Partial is a partially written contiguous slice of data;
/// it is the owner of the elements, but not the allocation,
/// and will drop the elements on drop.
#[must_use]
pub(crate) struct Partial<T> {
    /// Data pointer
    ptr: *mut T,
    /// Current length
    len: usize,
}

impl<T> Partial<T> {
    /// Create an empty partial for this data pointer
    ///
    /// Safety: Unless ownership is released, the 
    /// Partial acts as an owner of the slice of data (not the allocation);
    /// and will free the elements on drop; the pointer must be dereferenceable
    /// and the `len` elements following it valid.
    pub(crate) unsafe fn new(ptr: *mut T) -> Self {
        Self {
            ptr,
            len: 0,
        }
    }

    pub(crate) fn stub() -> Self {
        Self { len: 0, ptr: 0 as *mut _ }
    }

    pub(crate) fn is_stub(&self) -> bool {
        self.ptr.is_null()
    }

    /// Release Partial's ownership of the written elements, and return the current length
    pub(crate) fn release_ownership(mut self) -> usize {
        let ret = self.len;
        self.len = 0;
        ret
    }

    /// Merge if they are in order (left to right) and contiguous.
    /// Skips merge if T does not need drop.
    pub(crate) fn try_merge(mut left: Self, right: Self) -> Self {
        if !std::mem::needs_drop::<T>() {
            return left;
        }
        // Merge the partial collect results; the final result will be a slice that
        // covers the whole output.
        if left.is_stub() {
            right
        } else if left.ptr.wrapping_add(left.len) == right.ptr {
            left.len += right.release_ownership();
            left
        } else {
            // failure to merge; this is a bug in collect, so we will never reach this
            debug_assert!(false, "Partial: failure to merge left and right parts");
            left
        }
    }
}

unsafe impl<T> Send for Partial<T> where T: Send { }

impl<T> Drop for Partial<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                std::ptr::drop_in_place(std::slice::from_raw_parts_mut(self.ptr, self.len));
            }
        }
    }
}
