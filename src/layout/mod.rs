
mod layoutfmt;

// public but users don't interact with it
#[doc(hidden)]
/// Memory layout description
#[derive(Copy, Clone)]
pub struct Layout(u32);

pub trait LayoutPriv : Sized {
    fn new(x: u32) -> Self;
    fn and(self, flag: Self) -> Self;
    fn is(self, flag: u32) -> bool;
    fn flag(self) -> u32;
}

impl LayoutPriv for Layout {
    #[inline(always)]
    fn new(x: u32) -> Self { Layout(x) }

    #[inline(always)]
    fn is(self, flag: u32) -> bool {
        self.0 & flag != 0
    }
    #[inline(always)]
    fn and(self, flag: Layout) -> Layout {
        Layout(self.0 & flag.0)
    }

    #[inline(always)]
    fn flag(self) -> u32 {
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

pub const CORDER: u32 = 1 << 0;
pub const FORDER: u32 = 1 << 1;
