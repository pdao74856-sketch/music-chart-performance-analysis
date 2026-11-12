@echo off
setlocal
cd /d "%~dp0"
python -m streamlit run app/main.py --global.developmentMode false --server.address 127.0.0.1 --server.port 8502
if errorlevel 1 pause
