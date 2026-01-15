#!/bin/bash
# Linux 构建脚本

set -e

BUILD_TYPE=${1:-Release}

echo "========================================"
echo "Building TemplateMatcher DLL"
echo "Build Type: $BUILD_TYPE"
echo "========================================"
echo

# 创建构建目录
mkdir -p build
cd build

# 配置CMake
echo "Configuring CMake..."
cmake .. -DCMAKE_BUILD_TYPE=$BUILD_TYPE

# 编译
echo
echo "Building..."
make -j$(nproc)

echo
echo "========================================"
echo "Build completed successfully!"
echo "========================================"
echo
echo "Output files are in: build/lib"
echo

cd ..
