@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: Fischer Data Processing App - Update Script
:: ============================================================================
:: Purpose: Pull latest changes from GitHub and update dependencies if needed
:: ============================================================================

title Fischer Data App - Update
color 0B
echo.
echo ========================================
echo  Fischer Data Processing App
echo  Update Utility
echo ========================================
echo.

cd /d "C:\Program Files\FischerDataApp"

if not exist ".git" (
    echo ERROR: This folder is not a Git repository!
    echo Please reinstall using install.bat
    pause
    exit /b 1
)

echo Checking for updates from GitHub...
echo.

:: Save current requirements hash
if exist requirements.txt (
    for /f %%i in ('certutil -hashfile requirements.txt MD5 ^| find /v "hash"') do set OLD_HASH=%%i
)

:: Pull latest changes
git pull origin main
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to pull updates!
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo ✓ Update downloaded successfully
echo.

:: Check if requirements changed
if exist requirements.txt (
    for /f %%i in ('certutil -hashfile requirements.txt MD5 ^| find /v "hash"') do set NEW_HASH=%%i
    if not "!OLD_HASH!"=="!NEW_HASH!" (
        echo Dependencies have changed. Updating packages...
        call .venv311\Scripts\activate.bat
        pip install -r requirements.txt
        echo ✓ Dependencies updated
    ) else (
        echo No dependency changes detected.
    )
)

:: Show what changed
echo.
echo ========================================
echo  Recent Changes:
echo ========================================
git log -3 --oneline --decorate
echo.

echo ========================================
echo  Update Complete!
echo ========================================
echo.
echo You can now launch the app normally.
echo.

pause
