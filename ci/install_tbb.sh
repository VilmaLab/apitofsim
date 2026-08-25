#!/usr/bin/env bash

set -euo pipefail

tbb_version=2023.1.0
tbb_sha256=349d0e8b08cae4a5ab2668d54ff4e90b0fa012a332de6fb156961ddc119cd617
tbb_root=/opt/oneapi-tbb-${tbb_version}

if [[ ! -f "${tbb_root}/lib/pkgconfig/tbb.pc" ]]; then
  tbb_tmp=$(mktemp -d)
  trap 'rm -rf "${tbb_tmp}"' EXIT
  tbb_archive="${tbb_tmp}/oneapi-tbb-${tbb_version}-lin.tgz"
  curl -L --fail --silent --show-error \
    "https://github.com/uxlfoundation/oneTBB/releases/download/v${tbb_version}/oneapi-tbb-${tbb_version}-lin.tgz" \
    -o "${tbb_archive}"
  echo "${tbb_sha256}  ${tbb_archive}" | sha256sum --check -
  tar -xzf "${tbb_archive}" -C /opt
fi

install -d /usr/local/include /usr/local/lib/pkgconfig
ln -sfn "${tbb_root}/include/oneapi" /usr/local/include/oneapi
ln -sfn "${tbb_root}/include/tbb" /usr/local/include/tbb
for tbb_library in "${tbb_root}"/lib/intel64/gcc4.8/libtbb.so*; do
  ln -sfn "${tbb_library}" "/usr/local/lib/$(basename "${tbb_library}")"
done
sed "s|^prefix=.*|prefix=${tbb_root}|" \
  "${tbb_root}/lib/pkgconfig/tbb.pc" > /usr/local/lib/pkgconfig/tbb.pc
ldconfig
