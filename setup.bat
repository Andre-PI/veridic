@echo off
REM setup.bat — Veridic: setup automático (Windows)

echo.
echo   Veridic — setup
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REM Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado. Instale em https://python.org
    pause & exit /b 1
)
echo [OK] Python encontrado

REM venv
if not exist venv (
    echo Criando ambiente virtual...
    python -m venv venv
)
call venv\Scripts\activate.bat
echo [OK] Ambiente virtual ativo

REM dependencias
echo Instalando dependencias...
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo [OK] Dependencias instaladas

REM pesos
if not exist weights\csrnet_sha.pth (
    echo Baixando pesos do CSRNet...
    python download_weights.py
) else (
    echo [OK] Pesos do modelo encontrados
)

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo [OK] Setup concluido!
echo.
echo Para iniciar o Veridic:
echo   venv\Scripts\activate
echo   streamlit run app.py
echo.

set /p resp="Iniciar agora? (s/n): "
if /i "%resp%"=="s" venv\Scripts\streamlit run app.py
