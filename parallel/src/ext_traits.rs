
use ndarray::{
    Dimension,
    NdProducer,
    Zip,
};

use prelude::*;

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
