
use ndarray::{
    Dimension,
    NdProducer,
    Zip,
    ArrayBase,
    DataMut,
};

use prelude::*;

// Arrays

/// Parallel versions of `map_inplace` and `mapv_inplace`.
pub trait ParMap {
    type Item;
    fn par_map_inplace<F>(&mut self, f: F)
        where F: Fn(&mut Self::Item) + Sync;
    fn par_mapv_inplace<F>(&mut self, f: F)
        where F: Fn(Self::Item) -> Self::Item + Sync,
              Self::Item: Clone;
}

impl<A, S, D> ParMap for ArrayBase<S, D>
    where S: DataMut<Elem=A>,
          D: Dimension,
          A: Send + Sync,
{
    type Item = A;
    fn par_map_inplace<F>(&mut self, f: F)
        where F: Fn(&mut Self::Item) + Sync
    {
        self.view_mut().into_par_iter().for_each(f)
    }
    fn par_mapv_inplace<F>(&mut self, f: F)
        where F: Fn(Self::Item) -> Self::Item + Sync,
              Self::Item: Clone
    {
        self.view_mut().into_par_iter()
            .for_each(move |x| *x = f(x.clone()))
    }
}




// Zip

macro_rules! zip_impl {
    ($([$name:ident $($p:ident)*],)+) => {
        $(
        /// The `par_apply` method for `Zip`.
        ///
        /// This is a shorthand for using `.into_par_iter().for_each()` on
        /// `Zip`.
        pub trait $name<$($p),*> {
            fn par_apply<F>(self, function: F)
                where F: Fn($($p),*) + Sync;
        }

        #[allow(non_snake_case)]
        impl<Dim: Dimension, $($p: NdProducer<Dim=Dim>),*> $name<$($p::Item),*> for Zip<($($p,)*), Dim>
            where $($p::Item : Send , )*
                  $($p : Send , )*
        {
            fn par_apply<F>(self, function: F)
                where F: Fn($($p::Item),*) + Sync
            {
                self.into_par_iter().for_each(move |($($p,)*)| function($($p),*))
            }
        }
        )+
    }
}

zip_impl!{
    [ParApply1 P1],
    [ParApply2 P1 P2],
    [ParApply3 P1 P2 P3],
    [ParApply4 P1 P2 P3 P4],
    [ParApply5 P1 P2 P3 P4 P5],
    [ParApply6 P1 P2 P3 P4 P5 P6],
}
