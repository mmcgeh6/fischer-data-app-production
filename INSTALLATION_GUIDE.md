# Fischer Data Processing App - Installation Guide

## For Screenshare Installation Session

**Version**: 1.0
**Last Updated**: January 2026
**Target Machine**: FISCHER-DC01 (Windows Server)
**Installation Path**: `C:\Program Files\FischerDataApp\`

---

## Pre-Screenshare Checklist

### Day Before Screenshare

#### Part 1: Prepare Production Files (30 minutes)

**In your development repo**, create these files:
- [ ] `install.bat` - Main installation script
- [ ] `update_app.bat` - Update script for client
- [ ] `.env.template` - API key template
- [ ] `INSTALLATION_GUIDE.md` - This document
- [ ] Clean `requirements.txt` (remove duplicate lines 10-17)

#### Part 2: Create Production Repository (10 minutes)

1. **Create new GitHub repository**:
   - Repository name: `fischer-data-app-production`
   - Description: "Fischer Data Processing App - Production deployment for Fischer Energy"
   - Visibility: **Private** (recommended)
   - Do NOT initialize with README

2. **Create local production folder**:
   ```bash
   cd "C:\Users\minke\OneDrive\Desktop"
   mkdir fischer-data-app-production
   cd fischer-data-app-production
   git init
   git remote add origin https://github.com/mmcgeh6/fischer-data-app-production.git
   ```

3. **Copy files from development to production** (Windows):
   ```batch
   :: Core application files
   xcopy /E /I "..\Fischer Data Processing App\src" "src"
   xcopy "..\Fischer Data Processing App\launcher.py" .
   xcopy "..\Fischer Data Processing App\*.vbs" .
   xcopy "..\Fischer Data Processing App\requirements.txt" .

   :: Configuration
   xcopy /E /I "..\Fischer Data Processing App\.streamlit" ".streamlit"
   xcopy "..\Fischer Data Processing App\.gitignore" .

   :: Assets (rename to lowercase)
   xcopy /E /I "..\Fischer Data Processing App\Assets" "assets"

   :: Deployment files (NEW)
   xcopy "..\Fischer Data Processing App\install.bat" .
   xcopy "..\Fischer Data Processing App\update_app.bat" .
   xcopy "..\Fischer Data Processing App\.env.template" .
   xcopy "..\Fischer Data Processing App\INSTALLATION_GUIDE.md" .

   :: Documentation
   xcopy "..\Fischer Data Processing App\CLAUDE.md" .
   xcopy "..\Fischer Data Processing App\README.md" .

   :: Create empty folders
   mkdir logs
   mkdir archive
   mkdir CSVdata
   ```

4. **Update GITHUB_URL in install.bat**:
   - Open `install.bat` in text editor
   - Find line: `set "GITHUB_URL=https://github.com/YOUR_USERNAME/fischer-data-app-production.git"`
   - Replace `YOUR_USERNAME` with your actual GitHub username

5. **Commit and push to production repo**:
   ```bash
   git add .
   git commit -m "Initial production release - Fischer Data Processing App v12"
   git push -u origin main
   ```

### Morning of Screenshare

- [ ] Test clone production repo to verify it's accessible:
  ```bash
  git clone https://github.com/YOUR_USERNAME/fischer-data-app-production.git test-clone
  ```
- [ ] Have Claude API key ready to provide to client
- [ ] Review this installation guide
- [ ] Have GitHub URL ready: `https://github.com/YOUR_USERNAME/fischer-data-app-production`
- [ ] Test that both shortcuts work (optional, if you want to verify locally)

---

## Screenshare Installation Steps

**Total Time: 10-15 minutes**

### Step 1: Prerequisites Check (2 minutes)

Ask the client to open Command Prompt and verify:

1. **Python 3.11+ installed**:
   ```bash
   python --version
   ```
   - Should show: `Python 3.11.x` or higher
   - If not installed: Download from [python.org](https://python.org)
   - **IMPORTANT**: During Python install, check "Add Python to PATH"

2. **Git installed**:
   ```bash
   git --version
   ```
   - Should show: `git version x.x.x`
   - If not installed: Download from [git-scm.com](https://git-scm.com)

---

### Step 2: Download install.bat (1 minute)

**Option A - Direct Download** (Easiest):

1. In browser, navigate to:
   ```
   https://github.com/YOUR_USERNAME/fischer-data-app-production/blob/main/install.bat
   ```
2. Click the **"Raw"** button (top right of code view)
3. Right-click on the page → **"Save As..."**
4. Save as `install.bat` to Desktop or Downloads folder

**Option B - Clone Entire Repo** (Alternative):

```bash
cd C:\Temp
git clone https://github.com/YOUR_USERNAME/fischer-data-app-production.git
cd fischer-data-app-production
```

---

### Step 3: Run Installation (5-8 minutes)

1. **Right-click** `install.bat` → **"Run as administrator"**
   - If User Account Control (UAC) prompts, click "Yes"

2. **Follow the 8-phase installation**:

   **[1/8] Prerequisites Check** (automatic)
   - Validates admin rights, Python, and Git
   - Shows Python version detected

   **[2/8] Installation Directory** (automatic)
   - Shows: `C:\Program Files\FischerDataApp`

   **[3/8] Clone Repository** (30-60 seconds)
   - Downloads app from GitHub
   - If directory exists, asks to reinstall (choose Yes)

   **[4/8] Create Virtual Environment** (15-30 seconds)
   - Creates isolated Python environment

   **[5/8] Install Dependencies** (2-3 minutes) ← **Longest step**
   - Downloads and installs all Python packages
   - Progress bars will appear for each package

   **[6/8] Setup API Key** ← **YOU provide the key**
   - Prompt: `Enter your Claude API key (sk-ant-...): `
   - **Paste** the API key (sk-ant-api03-...)
   - Press Enter

   **[7/8] Configure Folder Permissions** (automatic)
   - Grants write access to logs/, archive/, CSVdata/

   **[8/8] Create Desktop Shortcuts** (automatic)
   - Creates "Fischer Data Processing App" shortcut
   - Creates "Update Fischer App" shortcut

3. **Installation Complete!**
   - Window will show success message
   - Press any key to close

---

### Step 4: Verification (2 minutes)

1. **Check Desktop**:
   - ✅ "Fischer Data Processing App.lnk" (with Fischer logo icon)
   - ✅ "Update Fischer App.lnk" (with Fischer logo icon)

2. **Launch the App**:
   - Double-click "Fischer Data Processing App" on desktop
   - Should see:
     - Console window minimizes to taskbar
     - Browser opens automatically to `http://localhost:5000`
     - Fischer logo and teal branding visible
     - App interface loads with 4-step workflow

3. **Test with Sample File** (if available):
   - Upload a sample CSV or Excel file
   - Click "Analyze All Files"
   - Verify AI analysis completes successfully

---

### Step 5: Update Test (1 minute)

**Test the update mechanism**:

1. On **your machine**, make a small change in the production repo:
   ```bash
   cd fischer-data-app-production
   echo "Test update" >> README.md
   git commit -am "Test update"
   git push origin main
   ```

2. On **client machine**, double-click "Update Fischer App" shortcut
   - Should show: "Checking for updates..."
   - Should show: "✓ Update downloaded successfully"
   - Should display recent changes (git log)

3. This confirms the client can receive your future updates!

---

### Step 6: Quick Walkthrough (2 minutes)

Show the client how to use the app:

1. **Data Upload**:
   - Option A: Upload files directly in the browser UI
   - Option B: Place files in `C:\Program Files\FischerDataApp\CSVdata\` folder

2. **4-Step Workflow**:
   - Step 1: Upload & Archive
   - Step 2: AI Analysis (automatic column detection)
   - Step 3: Review & Edit Configurations
   - Step 4: Process & Export (automatic)

3. **Output Files**:
   - All files saved to `archive\[Building Name]\` folder
   - Raw CSV: `{Building}_raw_merged_{timestamp}.csv`
   - Excel: `{Building}_resampled_15min_{timestamp}.xlsx`

4. **Future Updates**:
   - Double-click "Update Fischer App" on desktop
   - Wait 30 seconds
   - Relaunch app normally

---

## Troubleshooting Common Issues

### "ERROR: Administrator rights required"

**Cause**: install.bat not run as administrator
**Solution**: Right-click install.bat → "Run as administrator"

---

### "ERROR: Python not found"

**Cause**: Python not installed or not in PATH
**Solution**:
1. Download Python 3.11+ from [python.org](https://python.org)
2. During installation, **CHECK** "Add Python to PATH"
3. Restart Command Prompt and try again

---

### "ERROR: Git not found"

**Cause**: Git not installed
**Solution**:
1. Download Git from [git-scm.com](https://git-scm.com)
2. Use default installation options
3. Restart Command Prompt and try again

---

### "ERROR: Failed to clone repository"

**Possible Causes**:
- GitHub repo is private and credentials needed
- URL is incorrect
- Internet connection issue

**Solution**:
1. Verify repo URL is correct
2. If repo is private, ensure client has access or use HTTPS with credentials
3. Check internet connection

---

### Logo doesn't appear in app

**Cause**: Asset folder path issue
**Solution**: Verify that production repo has `assets/` folder (lowercase) with logo files

---

### Can't write to logs or archive folders

**Cause**: Folder permissions not granted properly
**Solution**: Run manually from Administrator Command Prompt:

```batch
cd "C:\Program Files\FischerDataApp"
icacls logs /grant Users:(OI)(CI)M /T
icacls archive /grant Users:(OI)(CI)M /T
icacls CSVdata /grant Users:(OI)(CI)M /T
```

---

### Port 5000 already in use

**Cause**: Another application using port 5000
**Solution**: The launcher automatically detects this and offers options:
- Kill existing process on port 5000
- Use alternate port (5001, 5002, etc.)
- Exit and manually resolve

---

### API key not working

**Cause**: Invalid key or incorrect format
**Solution**:
1. Open `C:\Program Files\FischerDataApp\.env` in Notepad
2. Verify format: `CLAUDE_API_KEY=sk-ant-...` (no spaces, no quotes)
3. Get new key from [console.anthropic.com](https://console.anthropic.com/)
4. Save file and restart app

---

## Post-Installation

### For Client

**Daily Use**:
- Double-click "Fischer Data Processing App" on desktop to launch
- Upload CSV/Excel files and process data
- Files automatically archived to preserve originals

**Getting Updates**:
- When notified of updates, double-click "Update Fischer App"
- Wait 30 seconds for update to complete
- Launch app normally

**Backup Important Data**:
- `C:\Program Files\FischerDataApp\archive\` (processed data)
- `C:\Program Files\FischerDataApp\.env` (API key)

---

### For You (Developer)

**Pushing Updates**:

1. Make changes in your development repo
2. Test changes locally: `streamlit run src/app_v12.py`
3. Copy changed files to production repo
4. Commit and push:
   ```bash
   cd fischer-data-app-production
   git add .
   git commit -m "Description of changes"
   git push origin main
   ```
5. Notify client: "New update available - please run Update Fischer App"

**Files That Auto-Update** (when client pulls):
- ✅ All Python code (`.py` files)
- ✅ Launcher scripts (`.vbs` files)
- ✅ Dependencies (`requirements.txt`)
- ✅ Assets (logos, icons)
- ✅ Documentation

**Files Preserved** (not updated):
- ❌ `.env` (client's API key)
- ❌ `logs/` (client's log history)
- ❌ `archive/` (client's processed data)
- ❌ `CSVdata/` (client's input files)

---

## Success Criteria

Installation is successful when:

1. ✅ Two desktop shortcuts created with Fischer logo
2. ✅ App launches in browser on port 5000
3. ✅ Logo displays correctly in app UI
4. ✅ Can upload and process sample file
5. ✅ Archive folder creates successfully
6. ✅ Update shortcut pulls from GitHub correctly
7. ✅ No permission errors in logs folder

---

## Support

For issues during installation, check:
- This troubleshooting section
- [CLAUDE.md](CLAUDE.md) for detailed architecture info
- [README.md](README.md) for general usage

---

**Original Development Repo**: Remains untouched for your experimentation
**Production Repo**: Clean, client-ready version for deployment
