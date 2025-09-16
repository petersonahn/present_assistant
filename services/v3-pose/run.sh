#!/bin/bash

# 실시간 면접 피드백 시스템 v3-pose 서비스 실행 스크립트

echo "🎯 실시간 면접 피드백 시스템 v3-pose 서비스 시작"

# 현재 디렉토리로 이동
cd "$(dirname "$0")"

# Python 및 pip 설치 확인
if ! command -v python3 &> /dev/null; then
    echo "❌ python3이 설치되어 있지 않습니다."
    echo "   sudo apt install python3 python3-pip 실행이 필요합니다."
    exit 1
fi

if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3이 설치되어 있지 않습니다."
    echo "   sudo apt install python3-pip 실행이 필요합니다."
    exit 1
fi

# 의존성 설치 확인
echo "📦 의존성 확인 중..."
if ! python3 -c "import fastapi" &> /dev/null; then
    echo "⚠️  의존성이 설치되어 있지 않습니다. 설치를 진행합니다..."
    pip3 install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        echo "❌ 의존성 설치에 실패했습니다."
        exit 1
    fi
    echo "✅ 의존성 설치가 완료되었습니다."
else
    echo "✅ 의존성이 이미 설치되어 있습니다."
fi

# 모델 파일 확인
if [ ! -f "pose/human-pose-estimation-0001.xml" ] || [ ! -f "pose/human-pose-estimation-0001.bin" ]; then
    echo "❌ OpenVINO 모델 파일이 없습니다."
    echo "   pose/ 디렉토리에 다음 파일들이 필요합니다:"
    echo "   - human-pose-estimation-0001.xml"
    echo "   - human-pose-estimation-0001.bin"
    exit 1
fi

echo "✅ 모델 파일이 확인되었습니다."

# 서버 시작
echo "🚀 서버를 시작합니다..."
echo "   URL: http://localhost:8000"
echo "   API 문서: http://localhost:8000/docs"
echo ""
echo "종료하려면 Ctrl+C를 누르세요."
echo ""

python3 main.py
