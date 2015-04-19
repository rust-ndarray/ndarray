use super::Dimension;

/// An iterator of the indexes of an array shape.
///
/// Iterator element type is `D`.
#[derive(Clone)]
pub struct Indexes<D> {
    dim: D,
    index: Option<D>,
}

impl<D: Dimension> Indexes<D>
{
    /// Create an iterator over the array shape `dim`.
    pub fn new(dim: D) -> Indexes<D>
    {
        Indexes {
            index: dim.first_index(),
            dim: dim,
        }
    }
}

impl<D: Dimension> Iterator for Indexes<D>
{
    type Item = D;
    #[inline]
    fn next(&mut self) -> Option<D>
    {
        let index = match self.index {
            None => return None,
            Some(ref ix) => ix.clone(),
        };
        self.index = self.dim.next_for(index.clone());
        Some(index)
    }

    fn size_hint(&self) -> (usize, Option<usize>)
    {
        let l = match self.index {
            None => 0,
            Some(ref ix) => {
                let gone = self.dim.default_strides().slice().iter()
                            .zip(ix.slice().iter())
                                 .fold(0, |s, (&a, &b)| s + a as usize * b as usize);
                self.dim.size() - gone
            }
        };
        (l, Some(l))
    }
}

