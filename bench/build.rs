fn main() {
    let dst = cmake::Config::new("vendor/vanedb-cpp")
        .define("VANEDB_BUILD_CAPI", "ON")
        .define("VANEDB_BUILD_TESTS", "OFF")
        .define("VANEDB_BUILD_BENCHMARKS", "OFF")
        .define("VANEDB_BUILD_PYTHON", "OFF")
        .define("VANEDB_BUILD_EXAMPLES", "OFF")
        .build_target("vanedb_cpp_capi")
        .build();

    // CMake (non-install) places the archive under <dst>/build.
    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-lib=static=vanedb_cpp_capi");

    // C++ standard library.
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
    // Watch everything the static lib is built from — a stale lib here would
    // silently benchmark old C++ code (headers included, this is header-only).
    println!("cargo:rerun-if-changed=vendor/vanedb-cpp/capi");
    println!("cargo:rerun-if-changed=vendor/vanedb-cpp/src/core");
    println!("cargo:rerun-if-changed=vendor/vanedb-cpp/CMakeLists.txt");
}
