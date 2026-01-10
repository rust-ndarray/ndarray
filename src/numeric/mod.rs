mod impl_numeric;

mod impl_float_maths;

#[cfg(feature = "std")]
pub use self::impl_float_maths::HasAngle;
