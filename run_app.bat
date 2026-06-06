@echo off
echo ============================================
echo  Vidhya Mitra - Starting Application
echo ============================================
cd /d D:\Offline-tutor-application

echo [1] Checking Python...
venv\Scripts\python.exe --version
if errorlevel 1 (
    echo ERROR: Python venv not found!
    pause
    exit /b 1
)

echo [2] Testing critical imports...
venv\Scripts\python.exe -c "import aiosqlite; print('aiosqlite OK')"
if errorlevel 1 (
    echo ERROR: aiosqlite broken! Reinstalling...
    venv\Scripts\python.exe -m pip install --force-reinstall --no-cache-dir aiosqlite==0.20.0
)

venv\Scripts\python.exe -c "import sqlalchemy; print('sqlalchemy OK')"
if errorlevel 1 (
    echo ERROR: sqlalchemy broken! Reinstalling...
    venv\Scripts\python.exe -m pip install --force-reinstall --no-cache-dir SQLAlchemy==2.0.30
)

venv\Scripts\python.exe -c "import fastapi; print('fastapi OK')"
if errorlevel 1 (
    echo ERROR: fastapi broken! Reinstalling...
    venv\Scripts\python.exe -m pip install --force-reinstall --no-cache-dir fastapi==0.111.0 uvicorn[standard]==0.29.0
)

echo [3] Starting server on http://localhost:8000 ...
echo ============================================
venv\Scripts\python.exe main.py
pause
