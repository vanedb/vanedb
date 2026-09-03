pub mod abtest;
pub mod config;
pub mod coverage;
pub mod ffi;
pub mod ground_truth;
pub mod workloads;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Impl {
    Cpp,
    Rs,
}
