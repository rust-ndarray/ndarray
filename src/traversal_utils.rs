// Copyright 2020 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::{
    IntoNdProducer,
    AssignElem,
    Zip,
};

/// Assign values from producer P1 to producer P2
/// P1 and P2 must be of the same shape and dimension
pub(crate) fn assign_to<'a, P1, P2, A>(from: P1, to: P2)
    where P1: IntoNdProducer<Item = &'a A>,
          P2: IntoNdProducer<Dim = P1::Dim>,
          P2::Item: AssignElem<A>,
          A: Clone + 'a
{
    Zip::from(from)
        .apply_assign_into(to, A::clone);
}

