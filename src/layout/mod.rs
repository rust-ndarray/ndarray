mod layoutfmt;

// Layout it a bitset used for internal layout description of
// arrays, producers and sets of producers.
// The type is public but users don't interact with it.
#[doc(hidden)]
/// Memory layout description
#[derive(Copy, Clone)]
pub struct Layout(u32);

impl Layout {
    #[inline(always)]
    pub(crate) fn is(self, flag: u32) -> bool {
        self.0 & flag != 0
    }

    /// Return layout common to both inputs
    #[inline(always)]
    pub(crate) fn intersect(self, other: Layout) -> Layout {
        Layout(self.0 & other.0)
    }

    /// Return a layout that simultaneously "is" what both of the inputs are
    #[inline(always)]
    pub(crate) fn also(self, other: Layout) -> Layout {
        Layout(self.0 | other.0)
    }

    #[inline(always)]
    pub(crate) fn one_dimensional() -> Layout {
        Layout::c().also(Layout::f())
    }

    #[inline(always)]
    pub(crate) fn c() -> Layout {
        Layout(CORDER | CPREFER)
    }

    #[inline(always)]
    pub(crate) fn f() -> Layout {
        Layout(FORDER | FPREFER)
    }

    #[inline(always)]
    pub(crate) fn cpref() -> Layout {
        Layout(CPREFER)
    }

    #[inline(always)]
    pub(crate) fn fpref() -> Layout {
        Layout(FPREFER)
    }

    #[inline(always)]
    pub(crate) fn none() -> Layout {
        Layout(0)
    }

    /// A simple "score" method which scores positive for preferring C-order, negative for F-order
    /// Subject to change when we can describe other layouts
    pub(crate) fn tendency(self) -> i32 {
        (self.is(CORDER) as i32 - self.is(FORDER) as i32) +
        (self.is(CPREFER) as i32 - self.is(FPREFER) as i32)

    }
}

pub const CORDER: u32 = 0b01;
pub const FORDER: u32 = 0b10;
pub const CPREFER: u32 = 0b0100;
pub const FPREFER: u32 = 0b1000;
