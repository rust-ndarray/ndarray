use crate::{Array, ArrayBase, DataMut, Dimension, IntoNdProducer, NdProducer, Zip};
use crate::AssignElem;

use crate::parallel::prelude::*;

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
            ///
            /// Restricted to functions that produce copyable results for technical reasons; other
            /// cases are not yet implemented.
            pub fn par_apply_collect<R>(self, f: impl Fn($($p::Item,)* ) -> R + Sync + Send) -> Array<R, D>
                where R: Copy + Send
            {
                let mut output = self.uninitalized_for_current_layout::<R>();
                self.par_apply_assign_into(&mut output, f);
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
