mod layoutfmt;

// public struct but users don't interact with it
#[doc(hidden)]
/// Memory layout description
#[derive(Copy, Clone)]
pub struct Layout(u32);

impl Layout {
    #[inline(always)]
    pub(crate) fn new(x: u32) -> Self {
        Layout(x)
    }

    #[inline(always)]
    pub(crate) fn is(self, flag: u32) -> bool {
        self.0 & flag != 0
    }
    #[inline(always)]
    pub(crate) fn and(self, flag: Layout) -> Layout {
        Layout(self.0 & flag.0)
    }

    #[inline(always)]
    pub(crate) fn flag(self) -> u32 {
        self.0
    }
}

impl Layout {
    #[doc(hidden)]
    #[inline(always)]
    pub fn one_dimensional() -> Layout {
        Layout(CORDER | FORDER)
    }
    #[doc(hidden)]
    #[inline(always)]
    pub fn c() -> Layout {
        Layout(CORDER)
    }
    #[doc(hidden)]
    #[inline(always)]
    pub fn f() -> Layout {
        Layout(FORDER)
    }
    #[inline(always)]
    #[doc(hidden)]
    pub fn none() -> Layout {
        Layout(0)
    }
}

pub const CORDER: u32 = 0b01;
pub const FORDER: u32 = 0b10;
