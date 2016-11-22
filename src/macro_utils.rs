
/// Derive Copy and Clone using the parameters (and bounds) as specified in []
macro_rules! copy_and_clone {
    ([$($parm:tt)*] $type_:ty) => {
        impl<$($parm)*> Copy for $type_ { }
        impl<$($parm)*> Clone for $type_ {
            #[inline(always)]
            fn clone(&self) -> Self { *self }
        }
    };
    ($type_:ty) => {
        copy_and_clone!{ [] $type_ }
    }
}
