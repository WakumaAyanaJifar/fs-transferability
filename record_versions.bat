@echo off
cd /d "%~dp0"
set PY=python
where python >nul 2>nul || set PY=py
> outputs\package_versions.txt 2>&1 (
  %PY% --version
  %PY% -c "import sys; print(sys.executable)"
  %PY% -m pip freeze
)
echo Package versions saved to outputs\package_versions.txt
pause
