use std::cell::Cell;
use std::mem::MaybeUninit;

use crate::math_cell::MathCell;

/// A producer element that can be assigned to once
pub trait AssignElem<T>
{
    /// Assign the value `input` to the element that self represents.
    fn assign_elem(self, input: T);
}

/// Assignable element, simply `*self = input`.
impl<T> AssignElem<T> for &mut T
{
    fn assign_elem(self, input: T)
    {
        *self = input;
    }
}

/// Assignable element, simply `self.set(input)`.
impl<T> AssignElem<T> for &Cell<T>
{
    fn assign_elem(self, input: T)
    {
        self.set(input);
    }
}

/// Assignable element, simply `self.set(input)`.
impl<T> AssignElem<T> for &MathCell<T>
{
    fn assign_elem(self, input: T)
    {
        self.set(input);
    }
}

/// Assignable element, the item in the MaybeUninit is overwritten (prior value, if any, is not
/// read or dropped).
impl<T> AssignElem<T> for &mut MaybeUninit<T>
{
    fn assign_elem(self, input: T)
    {
        *self = MaybeUninit::new(input);
    }
}
