@echo off
REM Add Graphviz to PATH for this session
set PATH=%PATH%;C:\Program Files\Graphviz\bin

REM Activate conda environment (if needed)
REM call conda activate your_env_name

REM Run visualization script
cd /d "%~dp0"
python visualize_model.py

pause
