@echo off
cd /d "%~dp0"
set PY=python
where python >nul 2>nul || set PY=py
%PY% --version || (echo Python not found. Install it from https://www.python.org/downloads/ and tick "Add to PATH". & pause & exit /b 1)
echo.
echo [1/4] Installing required packages...
%PY% -m pip install -q -r requirements.txt
echo.
echo [2/4] Downloading datasets...
%PY% 01_download_data.py || (echo Some downloads failed. Run this file again to retry. & pause & exit /b 1)
echo.
echo [3/4] Building inventory and checking duplicates...
%PY% 02_inventory.py
echo.
echo [4/4] Building the modelling table...
%PY% 03_harmonize.py
echo.
echo Done. Results are in the "outputs" folder.
pause
