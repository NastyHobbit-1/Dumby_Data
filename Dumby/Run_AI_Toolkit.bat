@echo off
setlocal
set SCRIPT_DIR=%~dp0
set PY=python
where %PY% >nul 2>&1
if errorlevel 1 (
  set PY=py
)

REM Ensure output root exists
if not exist "%SCRIPT_DIR%Boxxed_Data" mkdir "%SCRIPT_DIR%Boxxed_Data"

"%PY%" "%SCRIPT_DIR%Bulked_AI\main.py" --interactive %*
set EXITCODE=%ERRORLEVEL%
echo.
echo Press any key to close this window...
pause >nul
endlocal & exit /b %EXITCODE%

