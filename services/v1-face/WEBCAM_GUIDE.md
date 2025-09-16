# 🎥 웹캠 실시간 분석 실행 가이드

## 🏆 권장 방법: Windows에서 직접 실행

### 📋 사전 준비사항
1. ✅ 웹캠이 컴퓨터에 연결되어 있는지 확인
2. ✅ Python이 설치되어 있는지 확인
3. ✅ 다른 프로그램에서 웹캠을 사용하고 있지 않은지 확인

### 🚀 실행 방법

#### **방법 1: 원클릭 실행 (가장 간단)**
1. Windows 탐색기 열기
2. `C:\Users\201\Desktop\InterviewBuddy\services\v1-face\` 폴더로 이동
3. `run_on_windows.bat` 파일을 더블클릭
4. 자동으로 패키지 설치 및 실행됩니다!

#### **방법 2: 수동 실행**
1. **Windows PowerShell 또는 CMD 열기**
   - Windows 키 + R → "cmd" 입력 → Enter
   - 또는 Windows 키 + X → "Windows PowerShell" 선택

2. **폴더 이동**
   ```cmd
   cd C:\Users\201\Desktop\InterviewBuddy\services\v1-face
   ```

3. **Python 확인**
   ```cmd
   python --version
   ```
   만약 Python이 없다면: https://python.org 에서 설치

4. **패키지 설치**
   ```cmd
   pip install mediapipe numpy matplotlib opencv-python
   ```

5. **실시간 분석 실행**
   ```cmd
   python run_realtime_analysis.py
   ```

### 🎯 실행 화면 예시
```
🎭 실시간 면접 표정 분석 시스템
==================================================
📁 세션 폴더 생성: session_20250916_143116
✅ MediaPipe 감지기 초기화 완료
🎥 실시간 감정 분석을 시작합니다...
📝 종료하려면 'q' 키를 누르세요.
```

### 📊 실시간 분석 기능
- **실시간 감정 표시**: 화면에 현재 감정, 신뢰도, 점수 표시
- **얼굴 랜드마크**: 얼굴의 주요 포인트들을 실시간으로 표시
- **프레임별 데이터 저장**: 모든 분석 데이터를 자동 저장
- **종료 시 종합 리포트**: 전체 세션의 상세한 분석 결과 제공

### 🎮 사용 방법
1. **시작**: 프로그램 실행 후 웹캠 화면이 나타남
2. **분석**: 얼굴을 웹캠 앞에 두면 실시간으로 감정 분석
3. **종료**: 'q' 키를 누르거나 Ctrl+C로 종료
4. **결과 확인**: `results/session_YYYYMMDD_HHMMSS/` 폴더에서 상세 결과 확인

### 📁 결과 파일
```
results/session_20250916_143116/
├── frames/              # 캡처된 프레임 이미지들
├── analysis/            # 분석 데이터
└── reports/             # 종합 리포트
    ├── summary_report.json
    └── summary_report.txt
```

## 🔧 문제 해결

### ❌ 웹캠이 인식되지 않는 경우
1. **웹캠 연결 확인**
   - USB 웹캠: 케이블이 제대로 연결되었는지 확인
   - 내장 웹캠: 장치 관리자에서 카메라 상태 확인

2. **다른 프로그램에서 웹캠 사용 중인지 확인**
   - Zoom, Teams, Skype 등 종료
   - 브라우저의 카메라 사용 탭 닫기

3. **Windows 카메라 앱으로 테스트**
   - Windows 키 → "카메라" 검색 → 실행
   - 정상 작동하면 웹캠은 문제없음

### ❌ Python 오류가 발생하는 경우
1. **Python 재설치**
   - https://python.org 에서 최신 버전 다운로드
   - 설치 시 "Add Python to PATH" 체크

2. **패키지 재설치**
   ```cmd
   pip uninstall mediapipe opencv-python
   pip install mediapipe opencv-python
   ```

### ❌ 성능이 느린 경우
1. **다른 프로그램 종료**: CPU 사용량이 높은 프로그램들 종료
2. **웹캠 해상도 조정**: 고해상도 웹캠의 경우 성능에 영향을 줄 수 있음

## 🎯 면접 연습 팁
1. **좋은 조명**: 얼굴이 잘 보이도록 충분한 조명 확보
2. **안정적인 자세**: 웹캠 앞에서 안정적으로 앉기
3. **자연스러운 표정**: 과도한 표정보다는 자연스러운 표정 유지
4. **충분한 시간**: 최소 1-2분 이상 분석하여 정확한 결과 얻기

## 📞 추가 도움
문제가 지속되면 다음을 확인해주세요:
1. Windows 업데이트 상태
2. 웹캠 드라이버 최신 버전 여부
3. 바이러스 백신 프로그램의 카메라 접근 차단 여부
