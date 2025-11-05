mod impl_numeric;

mod impl_float_maths;

#[cfg(feature = "std")]
pub use self::impl_float_maths::{angle, angle_preserve, angle_scalar, HasAngle, HasAngle64};
