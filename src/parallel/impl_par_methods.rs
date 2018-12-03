
use {
    Dimension,
    NdProducer,
    Zip,
    ArrayBase,
    DataMut,
};

use parallel::prelude::*;


/// # Parallel methods
///
/// These methods require crate feature `rayon`.
impl<A, S, D> ArrayBase<S, D>
    where S: DataMut<Elem=A>,
          D: Dimension,
          A: Send + Sync,
{
    /// Parallel version of `map_inplace`.
    ///
    /// Modify the array in place by calling `f` by mutable reference on each element.
    ///
    /// Elements are visited in arbitrary order.
    pub fn par_map_inplace<F>(&mut self, f: F)
        where F: Fn(&mut A) + Sync + Send
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
        where F: Fn(A) -> A + Sync + Send,
              A: Clone,
    {
        self.view_mut().into_par_iter()
            .for_each(move |x| *x = f(x.clone()))
    }
}




// Zip

macro_rules! zip_impl {
    ($([$($p:ident)*],)+) => {
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
        }
        )+
    }
}

zip_impl!{
    [P1],
    [P1 P2],
    [P1 P2 P3],
    [P1 P2 P3 P4],
    [P1 P2 P3 P4 P5],
    [P1 P2 P3 P4 P5 P6],
}
