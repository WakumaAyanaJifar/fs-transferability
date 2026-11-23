@echo off
cd /d "%~dp0"
set PY=python
where python >nul 2>nul || set PY=py
if not exist data\processed\fs_model_table.csv (echo Run run_all.bat first. & pause & exit /b 1)
echo Running additional analyses R1-R13 (about 30-60 minutes)...
%PY% 05_revision.py > outputs\revision_log.txt 2>&1
type outputs\revision_log.txt
echo.
echo Building figures and LaTeX tables...
cd figures_tables
%PY% make_figures.py
%PY% figures_pro.py
%PY% figures_design.py
%PY% figures_revision.py
%PY% make_tables.py
cd ..
echo Done. Figures in outputs\figures, tables in outputs\tables.
pause
