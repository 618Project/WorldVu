#! /bin/bash

sudo apt-get --assume-yes install software-properties-common
sudo add-apt-repository ppa:george-edison55/cmake-3.x
sudo apt-get --assume-yes update
sudo apt-get --assume-yes install cmake && sudo apt-get --assume-yes upgrade cmake
sudo apt-get --assume-yes install git
sudo apt --assume-yes install subversion
sudo apt-get --assume-yes update
sudo apt-get --assume-yes install python python-dev
sudo apt-get --assume-yes install libgflags2v5 libgflags-dev
sudo apt-get --assume-yes install libgoogle-glog-dev
sudo apt-get --assume-yes install libatlas-base-dev
sudo apt-get --assume-yes install libeigen3-dev
cd ~
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
git checkout 85c6b5c
mkdir ceres-bin
cd ceres-bin
cmake ..
make -j3
sudo make install
sudo ln -s /usr/include/eigen3/Eigen /usr/local/include/Eigen
sudo apt-get --assume-yes install python-numpy
cd ~
git clone https://github.com/Itseez/opencv.git
cd opencv
git checkout f109c01
cmake -DWITH_IPP=OFF
make
sudo make install
sudo apt-get --assume-yes install python-pip
sudo pip install --upgrade pip
sudo pip install Gooey
sudo apt-get --assume-yes install python-wxgtk3.0
echo "deb http://archive.ubuntu.com/ubuntu wily main universe" | sudo tee /etc/apt/sources.list.d/wily-copies.list
sudo apt update
sudo apt --assume-yes install python-wxgtk2.8
sudo rm /etc/apt/sources.list.d/wily-copies.list
sudo apt update
sudo apt-get --assume-yes install python-pil
sudo apt-get --assume-yes install libtinfo-dev libjpeg-dev
cd ~
svn co https://llvm.org/svn/llvm-project/llvm/branches/release_37 llvm3.7
svn co https://llvm.org/svn/llvm-project/cfe/branches/release_37 llvm3.7/tools/clang
cd llvm3.7
mkdir build
cd build
cmake -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_TARGETS_TO_BUILD="X86;ARM;NVPTX;AArch64;Mips;PowerPC" -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=Release ..
make -j8
export LLVM_CONFIG=$HOME/llvm3.7/build/bin/llvm-config
export CLANG=$HOME/llvm3.7/build/bin/clang
cd ~
git clone https://github.com/halide/Halide.git
cd Halide
git checkout 970f749
mkdir cmake_build
cd cmake_build
cmake -DLLVM_DIR=$HOME/llvm3.7/build/share/llvm/cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_VERSION=37 -DWARNINGS_AS_ERRORS=OFF ..
sudo apt-get --assume-yes install libboost-dev
cd ~
git clone https://github.com/google/double-conversion.git
cd double-conversion
cmake -DBUILD_SHARED_LIBS=ON .
make
sudo make install
cd ~
git clone https://github.com/facebook/folly.git
cd folly/folly/test
rm -rf gtest
wget https://github.com/google/googletest/archive/release-1.8.0.tar.gz
tar zxf release-1.8.0.tar.gz
rm -f release-1.8.0.tar.gz
mv googletest-release-1.8.0 gtest
sudo apt-get install libiberty-dev

cd ~/folly/folly
sudo apt install autoconf
sudo apt-get --assume-yes install libtool
sudo apt-get --assume-yes install libssl-dev
autoreconf -ivf
sudo apt-get install --assume-yes libboost-all-dev
sudo apt-get install --assume-yes libevent-dev
./configure
make
make check
sudo make install

