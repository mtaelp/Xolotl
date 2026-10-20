@echo off
rem START-DS41.bat - serve DeepSeek-V4.1-Flash (GSQ) on Windows: an OpenAI / Anthropic API at http://127.0.0.1:8150/v1
rem   START-DS41.bat                         (the config: tools\v41\serve\strata-ds41-windows.json - edit the model paths there)
rem   START-DS41.bat --config my.json --port 8151
rem The first run compiles the engine (build-ds41-windows.bat) and makes a Python environment in .venv.
rem The config's "--gpu2 1" puts a second VRAM expert cache on card 1 (the RTX 3090; see docs\WINDOWS_DS41.md "Dual GPU").
rem Without that card the engine says so in the log and runs on card 0 alone; remove the two entries to skip the check.
rem The config's "STRATA_BLOB_MIRROR" reads half of each SSD expert from a copy of the coldest experts on a second SSD
rem (C:), made once by MAKE-DS41-MIRROR.bat (docs\WINDOWS_DS41.md "Second SSD"). Without that file the engine logs
rem "blob mirror: NOT used" and reads everything from the model's drive.
setlocal
cd /d "%~dp0"
if not exist build-ds41\ds41_generate.exe (
    echo The DeepSeek-V4.1 engine is not built yet - compiling it ^(this takes a while, once^) ...
    call build-ds41-windows.bat || exit /b 1
)
if not exist .venv\Scripts\python.exe (
    echo Making the Python environment ^(.venv^) ...
    py -3 -m venv .venv 2>nul || python -m venv .venv || (echo Python 3.10+ is needed: https://www.python.org/downloads/ & exit /b 1)
    .venv\Scripts\python.exe -m pip install -q -r requirements.txt || exit /b 1
)
if not exist logs mkdir logs
set PYTHONUTF8=1
set "ARGS=%*"
if "%ARGS%"=="" set "ARGS=--config tools\v41\serve\strata-ds41-windows.json --port 8150"
.venv\Scripts\python.exe tools\v41\serve\serve_ds41.py --engine strata %ARGS%
