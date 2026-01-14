@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: Fischer Data Processing App - Installation Script
:: ============================================================================
:: Purpose: Automated installation for Windows Server
:: Target: C:\Program Files\FischerDataApp\
:: Requirements: Python 3.11+, Git, Administrator rights
:: ============================================================================

title Fischer Data App - Installation Wizard
color 0A
echo.
echo ========================================
echo  Fischer Data Processing App
echo  Installation Wizard
echo ========================================
echo.

:: ============================================================================
:: PHASE 1: Prerequisites Check
:: ============================================================================

echo [1/8] Checking prerequisites...

:: Check admin rights
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Administrator rights required!
    echo Please right-click install.bat and select "Run as administrator"
    pause
    exit /b 1
)

:: Check Python 3.11+
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo Please install Python 3.11 or later from python.org
    pause
    exit /b 1
)

:: Get Python version and validate
for /f "tokens=2" %%a in ('python --version 2^>^&1') do set PYTHON_VERSION=%%a
echo Found Python %PYTHON_VERSION%

:: Check Git
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Git not found!
    echo Please install Git from git-scm.com
    pause
    exit /b 1
)

echo ✓ Prerequisites OK
echo.

:: ============================================================================
:: PHASE 2: Define Paths
:: ============================================================================

set "INSTALL_DIR=C:\Program Files\FischerDataApp"
set "GITHUB_URL=https://github.com/mmcgeh6/fischer-data-app-production.git"

echo [2/8] Installation directory: %INSTALL_DIR%
echo.

:: ============================================================================
:: PHASE 3: Clone Repository
:: ============================================================================

echo [3/8] Downloading application from GitHub...

if exist "%INSTALL_DIR%" (
    echo WARNING: Installation directory already exists!
    choice /M "Remove existing installation and reinstall"
    if !errorlevel! equ 2 (
        echo Installation cancelled.
        pause
        exit /b 0
    )
    rmdir /s /q "%INSTALL_DIR%"
)

git clone "%GITHUB_URL%" "%INSTALL_DIR%"
if %errorlevel% neq 0 (
    echo ERROR: Failed to clone repository!
    pause
    exit /b 1
)

echo ✓ Download complete
echo.

cd /d "%INSTALL_DIR%"

:: ============================================================================
:: PHASE 4: Create Virtual Environment
:: ============================================================================

echo [4/8] Creating Python virtual environment...

python -m venv .venv311
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment!
    pause
    exit /b 1
)

echo ✓ Virtual environment created
echo.

:: ============================================================================
:: PHASE 5: Install Dependencies
:: ============================================================================

echo [5/8] Installing Python packages (this may take 2-3 minutes)...

call .venv311\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)

echo ✓ Dependencies installed
echo.

:: ============================================================================
:: PHASE 6: Create .env File
:: ============================================================================

echo [6/8] Setting up API key...
echo.
echo You need a Claude AI API key for automatic column detection.
echo Get your key from: https://console.anthropic.com/
echo.

set /p API_KEY="Enter your Claude API key (sk-ant-...): "

if "!API_KEY!"=="" (
    echo WARNING: No API key provided. You can add it later to .env file.
    echo CLAUDE_API_KEY= > .env
) else (
    echo CLAUDE_API_KEY=!API_KEY! > .env
    echo ✓ API key saved
)
echo.

:: ============================================================================
:: PHASE 7: Grant Folder Permissions
:: ============================================================================

echo [7/8] Configuring folder permissions...

mkdir logs 2>nul
mkdir archive 2>nul
mkdir CSVdata 2>nul

icacls "%INSTALL_DIR%\logs" /grant Users:(OI)(CI)M /T /Q
icacls "%INSTALL_DIR%\archive" /grant Users:(OI)(CI)M /T /Q
icacls "%INSTALL_DIR%\CSVdata" /grant Users:(OI)(CI)M /T /Q

echo ✓ Permissions configured
echo.

:: ============================================================================
:: PHASE 8: Create Desktop Shortcuts
:: ============================================================================

echo [8/8] Creating desktop shortcuts...

:: Run VBScript to create "Fischer Data Processing App" shortcut
cscript //nologo create_desktop_shortcut.vbs

:: Create "Update App" shortcut
set "DESKTOP=%USERPROFILE%\Desktop"
set "SHORTCUT=%DESKTOP%\Update Fischer App.lnk"

powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT%'); $s.TargetPath = '%INSTALL_DIR%\update_app.bat'; $s.WorkingDirectory = '%INSTALL_DIR%'; $s.IconLocation = '%INSTALL_DIR%\assets\fischer app logo ico.ico,0'; $s.Description = 'Update Fischer Data Processing App from GitHub'; $s.Save()"

echo ✓ Desktop shortcuts created
echo.

:: ============================================================================
:: Installation Complete
:: ============================================================================

echo.
echo ========================================
echo  Installation Complete!
echo ========================================
echo.
echo Desktop shortcuts created:
echo   • Fischer Data Processing App (launch)
echo   • Update Fischer App (update from GitHub)
echo.
echo Installation location:
echo   %INSTALL_DIR%
echo.
echo Next steps:
echo   1. Double-click "Fischer Data Processing App" on desktop to launch
echo   2. Upload your CSV/Excel files in the app
echo   3. Process your data!
echo.
echo To update the app in the future:
echo   Double-click "Update Fischer App" on desktop
echo.

pause
