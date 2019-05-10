// Copyright 2017 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::Layout;
use super::LayoutPriv;
use itertools::Itertools;

const LAYOUT_NAMES: &'static [&'static str] = &["C", "F"];

use std::fmt;

impl fmt::Debug for Layout {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == 0 {
            write!(f, "Custom")
        } else {
            write!(
                f,
                "{}",
                (0..32)
                    .filter(|&i| self.is(1 << i))
                    .format_with(" | ", |i, f| if let Some(name) = LAYOUT_NAMES.get(i) {
                        f(name)
                    } else {
                        f(&format_args!("0x{:x}", i))
                    })
            )
        }?;
        write!(f, " ({:#x})", self.0)
    }
}
