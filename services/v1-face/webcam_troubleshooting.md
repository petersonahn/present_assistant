# 🎥 웹캠 문제 해결 가이드

## ❌ 현재 문제
WSL2 환경에서 웹캠에 접근할 수 없습니다.
```
[ WARN:0@0.826] global cap_v4l.cpp:913 open VIDEOIO(V4L2:/dev/video0): can't open camera by index
[ERROR:0@0.826] global obsensor_uvc_stream_channel.cpp:158 getStreamChannelGroup Camera index out of range
❌ 웹캠을 열 수 없습니다.
```

## 🔧 해결 방법들

### 방법 1: Windows에서 직접 실행 (권장)
WSL2 대신 Windows에서 직접 Python을 실행하세요.

1. **Windows PowerShell 또는 CMD 열기**
2. **Python 설치 확인**:
   ```cmd
   python --version
   pip --version
   ```

3. **필요한 패키지 설치**:
   ```cmd
   pip install mediapipe numpy matplotlib opencv-python
   ```

4. **프로젝트 폴더로 이동**:
   ```cmd
   cd C:\Users\201\Desktop\InterviewBuddy\services\v1-face
   ```

5. **실시간 분석 실행**:
   ```cmd
   python run_realtime_analysis.py
   ```

### 방법 2: WSL2에서 USB 웹캠 사용
WSL2에서 USB 웹캠을 사용하려면 추가 설정이 필요합니다.

1. **Windows에서 USB 웹캠 연결**
2. **WSL2 설정 파일 수정**:
   - `C:\Users\201\.wslconfig` 파일 생성/수정
   ```ini
   [wsl2]
   kernelCommandLine = usbcore.usbfs_memory_mb=1024
   ```

3. **WSL2 재시작**:
   ```cmd
   wsl --shutdown
   wsl
   ```

4. **웹캠 권한 확인**:
   ```bash
   ls -la /dev/video*
   ```

### 방법 3: 시뮬레이션 모드 사용 (현재 가능)
웹캠이 없는 환경에서 테스트 이미지로 시뮬레이션:

```bash
python3 simulate_realtime_analysis.py
```

### 방법 4: Docker 사용
Docker를 사용하여 웹캠에 접근:

1. **Dockerfile 생성** (이미 있음)
2. **Docker 이미지 빌드**:
   ```bash
   docker build -t emotion-analyzer .
   ```

3. **웹캠 권한으로 실행**:
   ```bash
   docker run --device=/dev/video0 -v $(pwd)/results:/app/results emotion-analyzer
   ```

## 🎯 권장 해결책

### 즉시 사용 가능한 방법:
1. **시뮬레이션 모드**: `python3 simulate_realtime_analysis.py`
2. **정적 이미지 분석**: `python3 face_lankmarker.py`

### 장기적 해결책:
1. **Windows에서 직접 실행** (가장 간단)
2. **Docker 사용** (고급 사용자용)

## 🔍 문제 진단

### 웹캠 상태 확인:
```bash
# WSL2에서
ls -la /dev/video*

# Windows에서 (PowerShell)
Get-PnpDevice | Where-Object {$_.Class -eq "Camera"}
```

### OpenCV 웹캠 테스트:
```python
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("웹캠 접근 가능")
    cap.release()
else:
    print("웹캠 접근 불가")
```

## 📞 추가 도움

문제가 지속되면:
1. Windows에서 웹캠이 정상 작동하는지 확인
2. 다른 프로그램에서 웹캠을 사용하고 있지 않은지 확인
3. 웹캠 드라이버가 최신인지 확인
