#!/usr/bin/env bash

set -euo pipefail

# Build a pinned OpenMP runtime for Clang CI and macOS wheels. This avoids
# system package downloads and preserves the macOS 12 deployment target.
runtime_root=$1
runtime_version=20.1.8
runtime_args=()
if [[ $(uname -s) == Darwin ]]; then
    runtime_args+=(-DCMAKE_OSX_DEPLOYMENT_TARGET=12.0)
fi
mkdir -p "$runtime_root/source/openmp" "$runtime_root/source/cmake"
for component in openmp cmake; do
    archive="$runtime_root/$component.tar.xz"
    curl --fail --location --retry 3 \
        "https://github.com/llvm/llvm-project/releases/download/llvmorg-$runtime_version/$component-$runtime_version.src.tar.xz" \
        --output "$archive"
done
printf '%s  %s\n' \
    b21c04ee9cbe56e200c5d83823765a443ee6389bbc3f64154c96e94016e6cee9 "$runtime_root/openmp.tar.xz" \
    3319203cfd1172bbac50f06fa68e318af84dcb5d65353310c0586354069d6634 "$runtime_root/cmake.tar.xz" \
    | shasum -a 256 --check -
for component in openmp cmake; do
    tar -xf "$runtime_root/$component.tar.xz" --strip-components=1 -C "$runtime_root/source/$component"
done
cmake -S "$runtime_root/source/openmp" -B "$runtime_root/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$runtime_root/install" \
    -DLLVM_COMMON_CMAKE_UTILS="$runtime_root/source/cmake" \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DLIBOMP_INSTALL_ALIASES=OFF \
    -DLIBOMP_OMPT_SUPPORT=OFF \
    "${runtime_args[@]}"
cmake --build "$runtime_root/build" --parallel 3
cmake --install "$runtime_root/build"
