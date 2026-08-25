dnf -y install clang clang-libs curl
bash ci/install_tbb.sh
pipx install -f patchelf==0.19.1.0rc1
