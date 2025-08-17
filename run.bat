@echo off
setlocal enabledelayedexpansion

REM === Cấu hình cơ bản ===
REM Tên thư mục venv (đặt cùng cấp file bat). Đổi nếu bạn dùng tên khác.
set VENV_DIR=venv

REM Tên file app Streamlit và module API
set STREAMLIT_APP=app.py
set API_APP=api:app

REM Port tuỳ chỉnh (đổi nếu bị trùng)
set STREAMLIT_PORT=8501
set API_PORT=8000

REM Lấy thư mục hiện tại (nơi chứa file bat)
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM === Kiểm tra venv tồn tại ===
if not exist "%VENV_DIR%\Scripts\activate" (
    echo [!] Khong tim thay venv tai "%VENV_DIR%\Scripts\activate".
    echo     Vui long tao venv:  python -m venv %VENV_DIR%
    echo     Sau do cai thu vien: pip install -r requirements.txt
    pause
    exit /b 1
)

REM === Kiểm tra file apps ===
if not exist "%STREAMLIT_APP%" (
    echo [!] Khong tim thay %STREAMLIT_APP% trong %SCRIPT_DIR%
    pause
    exit /b 1
)

if not exist "api.py" (
    echo [!] Khong tim thay api.py trong %SCRIPT_DIR%
    pause
    exit /b 1
)

REM === Mở cửa sổ 1: Streamlit ===
start "Streamlit UI" cmd /k ^
 "call %VENV_DIR%\Scripts\activate && ^
  python -m streamlit run %STREAMLIT_APP% --server.port=%STREAMLIT_PORT%"

REM === Mở cửa sổ 2: FastAPI ===
start "FastAPI Server" cmd /k ^
 "call %VENV_DIR%\Scripts\activate && ^
  python -m uvicorn %API_APP% --reload --port %API_PORT%"

echo ------------------------------------------------------------
echo Streamlit dang chay o: http://127.0.0.1:%STREAMLIT_PORT%
echo FastAPI  dang chay o:  http://127.0.0.1:%API_PORT%  (docs: /docs)
echo Hai cua so cmd da duoc mo rieng. Dong file nay an toan.
echo ------------------------------------------------------------
pause
endlocal
