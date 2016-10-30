
use Dimension;
use {Shape, StrideShape};
use Ix;
use dimension::IntoDimension;

/// A trait for `Shape` and `D where D: Dimension` that allows
/// customizing the memory layout (strides) of an array shape.
///
/// This trait is used together with array constructor methods like
/// `Array::from_shape_vec`.
pub trait ShapeBuilder {
    type Dim: Dimension;
    type Strides;

    fn f(self) -> Shape<Self::Dim>;
    fn set_f(self, is_f: bool) -> Shape<Self::Dim>;
    fn strides(self, strides: Self::Strides) -> StrideShape<Self::Dim>;
}

pub trait IntoShape {
    type Dim: Dimension;
    fn into_shape(self) -> Shape<Self::Dim>;
}

impl<D> IntoShape for D
    where D: IntoDimension,
{
    type Dim = D::Dim;
    fn into_shape(self) -> Shape<Self::Dim> {
        Shape {
            dim: self.into_dimension(),
            is_c: true,
        }
    }
}
impl<D> IntoShape for Shape<D>
    where D: Dimension,
{
    type Dim = D;
    fn into_shape(self) -> Shape<Self::Dim> {
        self
    }
}
/*

impl IntoShape for () {
    type Dim = [Ix; 0];
    fn into_shape(self) -> Shape<Self::Dim> {
        Shape {
            dim: [],
            is_c: true,
        }
    }
}

impl IntoShape for Ix {
    type Dim = [Ix; 1];
    fn into_shape(self) -> Shape<Self::Dim> {
        Shape {
            dim: [self],
            is_c: true,
        }
    }
}

impl IntoShape for (Ix, Ix) {
    type Dim = [Ix; 2];
    fn into_shape(self) -> Shape<Self::Dim> {
        Shape {
            dim: [self.0, self.1],
            is_c: true,
        }
    }
}
*/

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

impl From<Ix> for Shape<[Ix; 1]>
{
    fn from(ix: Ix) -> Self {
        Shape {
            dim: [ix],
            is_c: true,
        }
    }
}

impl From<(Ix, Ix)> for Shape<[Ix; 2]>
{
    fn from(ix: (Ix, Ix)) -> Self {
        Shape {
            dim: [ix.0, ix.1],
            is_c: true,
        }
    }
}

impl<T, D> From<T> for StrideShape<D>
    where D: Dimension,
          T: IntoShape<Dim=D>,
{
    fn from(value: T) -> Self {
        let shape = value.into_shape();
        let d = shape.dim;
        let st = if shape.is_c { d.default_strides() } else { d.fortran_strides() };
        StrideShape {
            strides: st,
            dim: d,
            custom: false,
        }
    }
}

/*
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
*/

impl<T> ShapeBuilder for T
    where T: IntoDimension
{
    type Dim = T::Dim;
    type Strides = T;
    fn f(self) -> Shape<Self::Dim> { self.set_f(true) }
    fn set_f(self, is_f: bool) -> Shape<Self::Dim> {
        self.into_shape().set_f(is_f)
    }
    fn strides(self, st: T) -> StrideShape<Self::Dim> {
        self.into_shape().strides(st.into_dimension())
    }
}

impl<D> ShapeBuilder for Shape<D>
    where D: Dimension
{
    type Dim = D;
    type Strides = D;
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


