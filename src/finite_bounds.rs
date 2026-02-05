use num_traits::Float;

pub enum Bound<F>
{
    Included(F),
    Excluded(F),
}

/// A version of std::ops::RangeBounds that only implements a..b and a..=b ranges.
pub trait FiniteBounds<F>
{
    fn start_bound(&self) -> F;
    fn end_bound(&self) -> Bound<F>;
}

impl<F> FiniteBounds<F> for std::ops::Range<F>
where F: Float
{
    fn start_bound(&self) -> F
    {
        self.start
    }

    fn end_bound(&self) -> Bound<F>
    {
        Bound::Excluded(self.end)
    }
}

impl<F> FiniteBounds<F> for std::ops::RangeInclusive<F>
where F: Float
{
    fn start_bound(&self) -> F
    {
        *self.start()
    }

    fn end_bound(&self) -> Bound<F>
    {
        Bound::Included(*self.end())
    }
}
