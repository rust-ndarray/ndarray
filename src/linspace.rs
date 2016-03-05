use libnum::Float;

/// An iterator of a sequence of evenly spaced floats.
///
/// Iterator element type is `F`.
pub struct Linspace<F> {
    start: F,
    step: F,
    index: usize,
    len: usize,
}

impl<F> Iterator for Linspace<F>
    where F: Float,
{
    type Item = F;

    #[inline]
    fn next(&mut self) -> Option<F> {
        if self.index >= self.len {
            None
        } else {
            // Calculate the value just like numpy.linspace does
            let i = self.index;
            self.index += 1;
            Some(self.start + self.step * F::from(i).unwrap())
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.len - self.index;
        (n, Some(n))
    }
}

impl<F> DoubleEndedIterator for Linspace<F>
    where F: Float,
{
    #[inline]
    fn next_back(&mut self) -> Option<F> {
        if self.index >= self.len {
            None
        } else {
            // Calculate the value just like numpy.linspace does
            self.len -= 1;
            let i = self.len;
            Some(self.start + self.step * F::from(i).unwrap())
        }
    }
}

impl<F> ExactSizeIterator for Linspace<F>
    where Linspace<F>: Iterator
{}

/// Return an iterator of evenly spaced floats.
///
/// The `Linspace` has `n` elements, where the first
/// element is `a` and the last element is `b`.
///
/// Iterator element type is `F`, where `F` must be
/// either `f32` or `f64`.
#[inline]
pub fn linspace<F>(a: F, b: F, n: usize) -> Linspace<F>
    where F: Float
{
    let step = if n > 1 {
        let nf: F = F::from(n).unwrap();
        (b - a) / (nf - F::one())
    } else {
        F::zero()
    };
    Linspace {
        start: a,
        step: step,
        index: 0,
        len: n,
    }
}

/// Return an iterator of floats spaced by `step`.
///
/// The `Linspace` has `n` elements, where the first
/// element is `a` and `b` is not included.
/// Numerical reasons can result in `b` being included
/// in the result.
///
/// Iterator element type is `F`, where `F` must be
/// either `f32` or `f64`.
#[inline]
pub fn arange<F>(a: F, b: F, step: F) -> Linspace<F>
    where F: Float
{
    let len = b - a;
    let steps = F::ceil(len / step);
    Linspace {
        start: a,
        step: step,
        len: steps.to_usize().unwrap(),
        index: 0,
    }
}
