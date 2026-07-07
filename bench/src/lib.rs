pub mod ffi;
pub mod workloads;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Impl {
    Cpp,
    Rs,
}
