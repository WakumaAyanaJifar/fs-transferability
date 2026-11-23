@echo off
cd /d "%~dp0"
set PY=python
where python >nul 2>nul || set PY=py
echo Installing/updating packages (scikit-learn, xgboost, shap, matplotlib)...
%PY% -m pip install -q -r requirements.txt
if not exist data\processed\fs_model_table.csv %PY% 03_harmonize.py
echo.
echo Running models (about 5-15 minutes)...
%PY% 04_model.py > outputs\model_log.txt 2>&1
type outputs\model_log.txt
echo.
echo Done. Results are in outputs\model.
pause
