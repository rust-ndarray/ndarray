
use imp_prelude::*;
use IntoDimension;
use {NdProducer, Layout};
use ::ElementsBase;
use ::ElementsBaseMut;

impl_ndproducer! {
    ['a, A, D: Dimension]
    [Clone => 'a, A, D: Clone ]
    WholeChunks {
        base,
        chunk,
        inner_strides,
    }
    WholeChunks<'a, A, D> {
        type Dim = D;
        type Item = ArrayView<'a, A, D>;

        unsafe fn item(&self, ptr) {
            ArrayView::new_(ptr, self.chunk.clone(),
                            self.inner_strides.clone())
        }
    }
}

type BaseProducerRef<'a, A, D> = ArrayView<'a, A, D>;
type BaseProducerMut<'a, A, D> = ArrayViewMut<'a, A, D>;

/// Whole chunks producer and iterable.
///
/// See [`.whole_chunks()`](struct.ArrayBase.html#method.whole_chunks) for more
/// information.
//#[derive(Debug)]
pub struct WholeChunks<'a, A: 'a, D> {
    base: BaseProducerRef<'a, A, D>,
    chunk: D,
    inner_strides: D,
}

/// **Panics** if any chunk dimension is zero<br>
pub fn whole_chunks_of<A, D, E>(mut a: ArrayView<A, D>, chunk: E) -> WholeChunks<A, D>
    where D: Dimension,
          E: IntoDimension<Dim=D>,
{
    let chunk = chunk.into_dimension();
    ndassert!(a.ndim() == chunk.ndim(),
              concat!("Chunk dimension {} does not match array dimension {} ",
                      "(with array of shape {:?})"),
             chunk.ndim(), a.ndim(), a.shape());
    for i in 0..a.ndim() {
        a.dim[i] /= chunk[i];
    }
    let inner_strides = a.raw_strides();
    a.strides *= &chunk;

    WholeChunks {
        base: a,
        chunk: chunk,
        inner_strides: inner_strides,
    }
}

impl<'a, A, D> IntoIterator for WholeChunks<'a, A, D>
    where D: Dimension,
          A: 'a,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = WholeChunksIter<'a, A, D>;
    fn into_iter(self) -> Self::IntoIter {
        WholeChunksIter {
            iter: self.base.into_elements_base(),
            chunk: self.chunk,
            inner_strides: self.inner_strides,
        }
    }
}

/// Whole chunks iterator.
///
/// See [`.whole_chunks()`](struct.ArrayBase.html#method.whole_chunks) for more
/// information.
pub struct WholeChunksIter<'a, A: 'a, D> {
    iter: ElementsBase<'a, A, D>,
    chunk: D,
    inner_strides: D,
}

impl_ndproducer! {
    ['a, A, D: Dimension]
    [Clone => ]
    WholeChunksMut {
        base,
        chunk,
        inner_strides,
    }
    WholeChunksMut<'a, A, D> {
        type Dim = D;
        type Item = ArrayViewMut<'a, A, D>;

        unsafe fn item(&self, ptr) {
            ArrayViewMut::new_(ptr,
                               self.chunk.clone(),
                               self.inner_strides.clone())
        }
    }
}

/// Whole chunks producer and iterable.
///
/// See [`.whole_chunks_mut()`](struct.ArrayBase.html#method.whole_chunks_mut)
/// for more information.
//#[derive(Debug)]
pub struct WholeChunksMut<'a, A: 'a, D> {
    base: BaseProducerMut<'a, A, D>,
    chunk: D,
    inner_strides: D,
}

/// **Panics** if any chunk dimension is zero<br>
pub fn whole_chunks_mut_of<A, D, E>(mut a: ArrayViewMut<A, D>, chunk: E)
    -> WholeChunksMut<A, D>
    where D: Dimension,
          E: IntoDimension<Dim=D>,
{
    let chunk = chunk.into_dimension();
    ndassert!(a.ndim() == chunk.ndim(),
              concat!("Chunk dimension {} does not match array dimension {} ",
                      "(with array of shape {:?})"),
             chunk.ndim(), a.ndim(), a.shape());
    for i in 0..a.ndim() {
        a.dim[i] /= chunk[i];
    }
    let inner_strides = a.raw_strides();
    a.strides *= &chunk;

    WholeChunksMut {
        base: a,
        chunk: chunk,
        inner_strides: inner_strides,
    }
}

impl<'a, A, D> IntoIterator for WholeChunksMut<'a, A, D>
    where D: Dimension,
          A: 'a,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = WholeChunksIterMut<'a, A, D>;
    fn into_iter(self) -> Self::IntoIter {
        WholeChunksIterMut {
            iter: self.base.into_elements_base(),
            chunk: self.chunk,
            inner_strides: self.inner_strides,
        }
    }
}

macro_rules! impl_iterator {
    (
    [$($typarm:tt)*]
    [Clone => $($cloneparm:tt)*]
     $typename:ident {
         $base:ident,
         $(
             $fieldname:ident,
         )*
     }
     $fulltype:ty {
        type Item = $ity:ty;

        fn item(&mut $self_:ident, $elt:pat) {
            $refexpr:expr
        }
    }) => { 
         expand_if!(@nonempty [$($cloneparm)*] 

            impl<$($cloneparm)*> Clone for $fulltype {
                fn clone(&self) -> Self {
                    $typename {
                        $base: self.$base.clone(),
                        $(
                            $fieldname: self.$fieldname.clone(),
                        )*
                    }
                }
            }

         );
        impl<$($typarm)*> Iterator for $fulltype {
            type Item = $ity;

            fn next(&mut $self_) -> Option<Self::Item> {
                $self_.$base.next().map(|$elt| {
                    $refexpr
                })
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                self.$base.size_hint()
            }
        }
    }
}

impl_iterator!{
    ['a, A, D: Dimension]
    [Clone => 'a, A, D: Clone]
    WholeChunksIter {
        iter,
        chunk,
        inner_strides,
    }
    WholeChunksIter<'a, A, D> {
        type Item = ArrayView<'a, A, D>;

        fn item(&mut self, elt) {
            unsafe {
                ArrayView::new_(
                    elt,
                    self.chunk.clone(),
                    self.inner_strides.clone())
            }
        }
    }
}

impl_iterator!{
    ['a, A, D: Dimension]
    [Clone => ]
    WholeChunksIterMut {
        iter,
        chunk,
        inner_strides,
    }
    WholeChunksIterMut<'a, A, D> {
        type Item = ArrayViewMut<'a, A, D>;

        fn item(&mut self, elt) {
            unsafe {
                ArrayViewMut::new_(
                    elt,
                    self.chunk.clone(),
                    self.inner_strides.clone())
            }
        }
    }
}

/// Whole chunks iterator.
///
/// See [`.whole_chunks_mut()`](struct.ArrayBase.html#method.whole_chunks_mut)
/// for more information.
pub struct WholeChunksIterMut<'a, A: 'a, D> {
    iter: ElementsBaseMut<'a, A, D>,
    chunk: D,
    inner_strides: D,
}
