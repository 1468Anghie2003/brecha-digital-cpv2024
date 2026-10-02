@echo off
chcp 65001 >nul
title Brecha digital CPV 2024 - Aplicacion web
cd /d "%~dp0"

echo ===============================================================
echo   BRECHA DIGITAL Y EQUIPAMIENTO DEL HOGAR - CPV 2024
echo   Universidad Mayor de San Andres - Carrera de Informatica
echo ===============================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python no esta instalado o no esta en el PATH.
    echo          Descarguelo de python.org y marque la casilla
    echo          "Add Python to PATH" durante la instalacion.
    echo.
    pause
    exit /b 1
)

echo  Verificando librerias...
python -c "import streamlit, plotly, joblib, sklearn" >nul 2>&1
if errorlevel 1 (
    echo  Instalando librerias necesarias. Esto puede tardar varios
    echo  minutos la primera vez.
    echo.
    python -m pip install --quiet --upgrade pip
    python -m pip install -r requirements.txt
    echo.
)

if not exist "mlops\registro.json" (
    echo  [ERROR] No se encuentra la carpeta mlops.
    echo          Debe estar en la misma carpeta que app_mlops.py
    echo.
    pause
    exit /b 1
)

echo  Iniciando la aplicacion...
echo  Se abrira en el navegador: http://localhost:8501
echo  Para cerrarla, pulse Ctrl + C en esta ventana.
echo.
python -m streamlit run app_mlops.py
pause
