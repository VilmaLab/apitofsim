dnf -y install clang clang-libs curl
bash ci/install_tbb.sh
PIPX_DEFAULT_PYTHON=/usr/bin/python3 pipx install -f patchelf==0.19.1.0rc1
