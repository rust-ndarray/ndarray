
// Send and Sync
// All the iterators are thread safe the same way the slice's iterator are

// read-only iterators use Sync => Send rules, same as `std::slice::Iter`.
macro_rules! send_sync_read_only {
    ($name:ident) => {
        unsafe impl<'a, A, D> Send for $name<'a, A, D> where A: Sync, D: Send { }
        unsafe impl<'a, A, D> Sync for $name<'a, A, D> where A: Sync, D: Sync { }
    }
}

// read-write iterators use Send => Send rules, same as `std::slice::IterMut`.
macro_rules! send_sync_read_write {
    ($name:ident) => {
        unsafe impl<'a, A, D> Send for $name<'a, A, D> where A: Send, D: Send { }
        unsafe impl<'a, A, D> Sync for $name<'a, A, D> where A: Sync, D: Sync { }
    }
}

