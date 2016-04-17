
use Dimension;
use {Shape, StrideShape};

/// A trait for `Shape` and `D where D: Dimension` that allows
/// customizing the memory layout (strides) of an array shape.
///
/// This trait is used together with array constructor methods like
/// `OwnedArray::from_shape_vec`.
pub trait ShapeBuilder {
    type Dim: Dimension;

    fn f(self) -> Shape<Self::Dim>;
    fn set_f(self, is_f: bool) -> Shape<Self::Dim>;
    fn strides(self, strides: Self::Dim) -> StrideShape<Self::Dim>;
}

impl<D> From<D> for Shape<D>
    where D: Dimension
{
    fn from(d: D) -> Self {
        Shape {
            dim: d,
            is_c: true,
        }
    }
}

impl<D> From<D> for StrideShape<D>
    where D: Dimension
{
    fn from(d: D) -> Self {
        StrideShape {
            strides: d.default_strides(),
            dim: d,
            custom: false,
        }
    }
}

impl<D> From<Shape<D>> for StrideShape<D>
    where D: Dimension
{
    fn from(shape: Shape<D>) -> Self {
        let d = shape.dim;
        let st = if shape.is_c { d.default_strides() } else { d.fortran_strides() };
        StrideShape {
            strides: st,
            dim: d,
            custom: false,
        }
    }
}

impl<D> ShapeBuilder for D
    where D: Dimension
{
    type Dim = D;
    fn f(self) -> Shape<D> { self.set_f(true) }
    fn set_f(self, is_f: bool) -> Shape<D> {
        Shape::from(self).set_f(is_f)
    }
    fn strides(self, st: D) -> StrideShape<D> {
        Shape::from(self).strides(st)
    }
}

impl<D> ShapeBuilder for Shape<D>
    where D: Dimension
{
    type Dim = D;
    fn f(self) -> Self { self.set_f(true) }
    fn set_f(mut self, is_f: bool) -> Self {
        self.is_c = !is_f;
        self
    }
    fn strides(self, st: D) -> StrideShape<D> {
        StrideShape {
            dim: self.dim,
            strides: st,
            custom: true,
        }
    }
}


