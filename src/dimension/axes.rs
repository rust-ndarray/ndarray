
use {Dimension, Axis, Ix, Ixs};

/// Create a new Axes iterator
pub fn axes_of<'a, D>(d: &'a D, strides: &'a D) -> Axes<'a, D>
    where D: Dimension,
{
    Axes {
        dim: d,
        strides: strides,
        start: 0,
        end: d.ndim(),
    }
}

/// An iterator over the length and stride of each axis of an array.
///
/// See [`.axes()`](struct.ArrayBase.html#method.axes) for more information.
#[derive(Debug)]
pub struct Axes<'a, D: 'a> {
    dim: &'a D,
    strides: &'a D,
    start: usize,
    end: usize,
}

/// Description of the axis, its length and its stride.
#[derive(Copy, Clone, Debug)]
pub struct AxisDescription(pub Axis, pub Ix, pub Ixs);

impl AxisDescription {
    /// Return axis
    #[inline(always)]
    pub fn axis(self) -> Axis { self.0 }
    /// Return length
    #[inline(always)]
    pub fn len(self) -> Ix { self.1 }
    /// Return stride
    #[inline(always)]
    pub fn stride(self) -> Ixs { self.2 }
}

impl<'a, D> Copy for Axes<'a, D> { }
impl<'a, D> Clone for Axes<'a, D> {
    fn clone(&self) -> Self { *self }
}

impl<'a, D> Iterator for Axes<'a, D>
    where D: Dimension,
{
    /// Description of the axis, its length and its stride.
    type Item = AxisDescription;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            let i = self.start.post_inc();
            Some(AxisDescription(Axis(i), self.dim[i], self.strides[i] as Ixs))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.end - self.start;
        (len, Some(len))
    }
}

impl<'a, D> DoubleEndedIterator for Axes<'a, D>
    where D: Dimension,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            let i = self.end.pre_dec();
            Some(AxisDescription(Axis(i), self.dim[i], self.strides[i] as Ixs))
        } else {
            None
        }
    }
}

trait IncOps : Copy {
    fn post_inc(&mut self) -> Self;
    fn post_dec(&mut self) -> Self;
    fn pre_dec(&mut self) -> Self;
}

impl IncOps for usize {
    #[inline(always)]
    fn post_inc(&mut self) -> Self {
        let x = *self;
        *self += 1;
        x
    }
    #[inline(always)]
    fn post_dec(&mut self) -> Self {
        let x = *self;
        *self -= 1;
        x
    }
    #[inline(always)]
    fn pre_dec(&mut self) -> Self {
        *self -= 1;
        *self
    }
}

