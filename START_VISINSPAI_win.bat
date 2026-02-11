@echo off
title Pornire Aplicatie VisInspAI
echo ========================================
echo    PORNIRE APLICATIE VISINSPAI         
echo ========================================

:: folder cu fisier
cd /d "%~dp0"

:: rulam streamlit
python -m streamlit run src/app/app.py

pause