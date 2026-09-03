#!/usr/bin/env bash
# Execute the installed wheel, not a rebuild, under each emulated CPU profile.
set -euo pipefail

wheel_python="${1:?Pass the absolute path to the Python containing the wheel}"
test_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

# NumPy's own binary CPU baseline can exceed SSE2. Check our module's import
# without NumPy on qemu64, then exercise the complete stack on no-AVX Nehalem.
qemu-x86_64 -cpu qemu64 "$wheel_python" -I -c \
  'import vanedb_cpp as v; assert v.simd_backend() == "scalar"; print("baseline module import: scalar")'

for cpu_profile in 'Nehalem' 'SandyBridge' 'Haswell,-fma' 'Haswell'; do
  expected_backend=scalar
  if [[ "$cpu_profile" == Haswell ]]; then
    expected_backend=avx2_fma
  fi
  export VANEDB_EXPECT_BACKEND="$expected_backend"
  echo "CPU=$cpu_profile expected=$expected_backend"
  qemu-x86_64 -cpu "$cpu_profile" "$wheel_python" -I -c \
    'import os, vanedb_cpp as v; print(v.__file__, v.simd_backend()); assert v.simd_backend() == os.environ["VANEDB_EXPECT_BACKEND"]'
  qemu-x86_64 -cpu "$cpu_profile" "$wheel_python" -I -m pytest \
    -q -p no:cacheprovider "$test_directory/test_python_bindings.py"
done
