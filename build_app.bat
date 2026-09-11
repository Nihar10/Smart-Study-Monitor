@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
python -m PyInstaller --noconsole --onefile --hidden-import=mediapipe --hidden-import=cvzone --hidden-import=ultralytics app.py

echo.
echo Build complete. EXE is in the dist folder.
pause
