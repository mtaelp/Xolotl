@echo off
rem build-ds41-windows.bat - compile the DeepSeek-V4.1 engine (ds41_generate.exe) on Windows.
rem Needs: Visual Studio 2022 (or 2019) C++ build tools, the CUDA 13.x toolkit, cmake + ninja on PATH, and
rem third_party\llama.cpp at the pinned commit with tools\xolotl\ggml\xolotl-ggml.patch applied.
rem   build-ds41-windows.bat              (every NVIDIA card in this PC, from nvidia-smi: RTX 50 = 120a, RTX 40 = 89,
rem                                        RTX 30 = 86; none found = 120a)
rem   set CUDA_ARCH=120a;86 ^& build-ds41-windows.bat     (choose; --gpu2 needs the second card's arch in the list too)
setlocal EnableDelayedExpansion
cd /d "%~dp0"
rem RTX 50 needs the arch-specific 120a: ggml's MXFP4 MMQ uses block-scaled MMA, which plain sm_120 does not have
if "%CUDA_ARCH%"=="" (
    for /f "tokens=1,2 delims=." %%a in ('nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader 2^>nul') do (
        set "_cc=%%a%%b"
        if "!_cc:~0,2!"=="12" set "_cc=!_cc!a"
        set "_dup="
        for %%x in (!CUDA_ARCH:;= !) do if "%%x"=="!_cc!" set "_dup=1"
        if not defined _dup (if "!CUDA_ARCH!"=="" (set "CUDA_ARCH=!_cc!") else (set "CUDA_ARCH=!CUDA_ARCH!;!_cc!"))
    )
)
if "%CUDA_ARCH%"=="" set CUDA_ARCH=120a
echo CUDA architectures: %CUDA_ARCH%
if "%CUDA_PATH_V13%"=="" (
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" set "CUDA_PATH_V13=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
)
if "%VCVARS%"=="" (
    for %%V in ("C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools" "C:\Program Files\Microsoft Visual Studio\2022\Community" "C:\Program Files\Microsoft Visual Studio\2022\Professional" "C:\Program Files\Microsoft Visual Studio\2022\Enterprise") do (
        if exist "%%~V\VC\Auxiliary\Build\vcvars64.bat" if not defined VCVARS set "VCVARS=%%~V\VC\Auxiliary\Build\vcvars64.bat"
    )
)
if "%VCVARS%"=="" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%" >nul || (echo cannot run "%VCVARS%" & exit /b 1)
set "PATH=%CUDA_PATH_V13%\bin;%PATH%"
rem an existing build for other architectures is reconfigured (its CUDA objects rebuild once)
set "_have="
if exist build-ds41\CMakeCache.txt (
    for /f "tokens=2 delims==" %%a in ('findstr /b /c:"CMAKE_CUDA_ARCHITECTURES:" build-ds41\CMakeCache.txt') do set "_have=%%a"
    if not "!_have!"=="%CUDA_ARCH%" (
        echo build-ds41 was configured for !_have! - reconfiguring for %CUDA_ARCH%
        cmake build-ds41 "-DCMAKE_CUDA_ARCHITECTURES=%CUDA_ARCH%" || exit /b 1
    )
)
if not exist build-ds41\CMakeCache.txt (
    cmake -S . -B build-ds41 -G Ninja -DCMAKE_BUILD_TYPE=Release ^
        -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_BUILD_TESTS=OFF ^
        "-DCMAKE_CUDA_ARCHITECTURES=%CUDA_ARCH%" "-DCMAKE_CUDA_COMPILER=%CUDA_PATH_V13%\bin\nvcc.exe" ^
        "-DCUDAToolkit_ROOT=%CUDA_PATH_V13%" "-DSTRATA_GGML_DIR=%~dp0third_party\llama.cpp" ^
        -DSTRATA_DSV41_FQ_OP=ON -DSTRATA_DSV41_HC_MIX=ON || exit /b 1
)
cmake --build build-ds41 --target ds41_generate %* || exit /b 1
echo built: %~dp0build-ds41\ds41_generate.exe
