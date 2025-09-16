@echo off
echo 🎭 면접 표정 분석 시스템 (Windows 실행)
echo ================================================

REM Python 설치 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python이 설치되지 않았습니다.
    echo 💡 Python을 설치한 후 다시 실행하세요: https://python.org
    pause
    exit /b 1
)

echo ✅ Python 설치 확인됨

REM 필요한 패키지 설치 확인
echo 📦 필요한 패키지 설치 중...
pip install mediapipe numpy matplotlib opencv-python --quiet

if errorlevel 1 (
    echo ❌ 패키지 설치에 실패했습니다.
    pause
    exit /b 1
)

echo ✅ 패키지 설치 완료

REM 실시간 분석 실행
echo 🎥 실시간 분석을 시작합니다...
echo 📝 종료하려면 'q' 키를 누르거나 Ctrl+C를 누르세요
echo.

python run_realtime_analysis.py

echo.
echo ✅ 분석이 완료되었습니다.
echo 📁 결과는 results 폴더에서 확인할 수 있습니다.
pause
