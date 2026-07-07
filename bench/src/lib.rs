pub mod ffi;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Impl {
    Cpp,
    Rs,
}
