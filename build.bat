@echo off
REM Windows 构建脚本

setlocal enabledelayedexpansion

REM 设置默认配置
set BUILD_TYPE=Release
if not "%1"=="" set BUILD_TYPE=%1

echo ========================================
echo Building TemplateMatcher DLL
echo Build Type: %BUILD_TYPE%
echo ========================================
echo.

REM 创建构建目录
if not exist build mkdir build
cd build

REM 配置CMake
echo Configuring CMake...
if defined OpenCV_DIR (
    echo Using OpenCV_DIR: %OpenCV_DIR%
    cmake .. -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DOpenCV_DIR=%OpenCV_DIR%
) else (
    cmake .. -DCMAKE_BUILD_TYPE=%BUILD_TYPE%
)
if errorlevel 1 (
    echo CMake configuration failed!
    cd ..
    pause
    exit /b 1
)

REM 编译
echo.
echo Building...
cmake --build . --config %BUILD_TYPE%
if errorlevel 1 (
    echo Build failed!
    cd ..
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo Output files are in: build\bin (Release) or build\bin\Debug (Debug)
echo.

cd ..
pause
