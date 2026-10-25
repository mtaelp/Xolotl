@echo off
rem MAKE-DS41-MIRROR.bat - copy the coldest DeepSeek-V4.1 experts to a SECOND SSD, once, so the engine reads each
rem SSD expert half from each drive (the blob mirror; docs\WINDOWS_DS41.md "Second SSD").
rem   MAKE-DS41-MIRROR.bat                                   (C:\ai\ds41-mirror\V41-GSQ-1017.mirror, 85 GiB)
rem   MAKE-DS41-MIRROR.bat D:\ds41\V41-GSQ-1017.mirror 48    (another place / size in GiB)
rem   MAKE-DS41-MIRROR.bat C:\ai\ds41-mirror\ISTA-IQ2_XXS.mirror 48 tools\v41\serve\strata-ds41-ista-windows.json
rem                                                          (another model: its config, third)
rem The model and --rank come from the config (default tools\v41\serve\strata-ds41-windows.json).  The engine refuses before writing anything
rem unless at least max(100 GiB, 10% of the drive) stays free afterwards (MIRROR_KEEP_GIB=N to change the 100).  It
rem then compares every mirrored expert with the model file.  Turn it on in the config's "env":
rem   "STRATA_BLOB_MIRROR": "<the file>", "STRATA_BLOB_MIRROR_SHARE": "0.5"
rem The mirror belongs to that model file: rebuild it after the model or the rank file changes (the engine checks the
rem size and 8 experts at start and runs without the mirror if they do not match).
setlocal
cd /d "%~dp0"
set "OUT=%~1"
if "%OUT%"=="" set "OUT=C:\ai\ds41-mirror\V41-GSQ-1017.mirror"
set "GIB=%~2"
if "%GIB%"=="" set "GIB=85"
set "CFG=%~3"
if "%CFG%"=="" set "CFG=tools\v41\serve\strata-ds41-windows.json"
if "%MIRROR_KEEP_GIB%"=="" set "MIRROR_KEEP_GIB=100"
if not exist build-ds41\ds41_generate.exe (echo build the engine first: build-ds41-windows.bat & exit /b 1)
for /f "usebackq delims=" %%a in (`powershell -NoProfile -Command "$c = Get-Content '%CFG%' -Raw | ConvertFrom-Json; $a = @($c.args); $m = $a[[array]::IndexOf($a, '-m') + 1]; $r = $a[[array]::IndexOf($a, '--rank') + 1]; Write-Output ($m + '|' + $r)"`) do set "MR=%%a"
for /f "tokens=1,2 delims=|" %%a in ("%MR%") do (set "MODEL=%%a" & set "RANK=%%b")
if "%MODEL%"=="" (echo cannot read -m / --rank from %CFG% & exit /b 1)
for %%d in ("%OUT%") do if not exist "%%~dpd" mkdir "%%~dpd" || exit /b 1
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;%PATH%"
echo model %MODEL%
echo rank  %RANK%
echo mirror %OUT% (%GIB% GiB, keeping at least %MIRROR_KEEP_GIB% GiB free)
build-ds41\ds41_generate.exe -m "%MODEL%" --rank "%RANK%" --build-mirror "%OUT%" --mirror-gib %GIB% --mirror-keep-gib %MIRROR_KEEP_GIB% || exit /b 1
echo checking every mirrored expert against the model file ...
build-ds41\ds41_generate.exe -m "%MODEL%" --check-mirror "%OUT%" || (echo the mirror does NOT match the model file & exit /b 1)
echo done.  Add to %CFG% "env":
echo   "STRATA_BLOB_MIRROR": "%OUT:\=/%",
echo   "STRATA_BLOB_MIRROR_SHARE": "0.5"
