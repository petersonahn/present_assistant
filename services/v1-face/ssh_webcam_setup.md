# 🎥 SSH 환경에서 웹캠 사용 가이드

## 🔍 현재 상황
SSH 가상환경에서는 웹캠에 직접 접근할 수 없습니다. 다음과 같은 해결 방법들이 있습니다.

## 🔧 해결 방법들

### 방법 1: X11 포워딩 사용 (Linux/Mac에서 Windows로 SSH)
```bash
# SSH 연결 시 X11 포워딩 활성화
ssh -X username@hostname
# 또는
ssh -Y username@hostname

# 웹캠 권한 확인
ls -la /dev/video*

# 실행
python3 webcam_analyzer.py
```

### 방법 2: VNC 서버 사용
```bash
# VNC 서버 설치 (Ubuntu/Debian)
sudo apt update
sudo apt install x11vnc

# VNC 서버 시작
x11vnc -display :0 -auth guess -forever -loop -noxdamage -repeat -rfbauth /home/user/.vnc/passwd -rfbport 5900 -shared

# VNC 클라이언트로 연결 후 웹캠 사용
```

### 방법 3: 웹 스트리밍 방식 (권장)
웹캠을 웹 서버로 스트리밍하고 원격에서 접근하는 방법:

```python
# webcam_streamer.py 생성
import cv2
import base64
from flask import Flask, Response, render_template
import threading

app = Flask(__name__)

def get_frame():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

### 방법 4: Docker with Webcam Access
```bash
# Docker로 웹캠 접근
docker run -it --device=/dev/video0 -p 5000:5000 emotion-analyzer

# 또는 Docker Compose 사용
version: '3'
services:
  emotion-analyzer:
    build: .
    devices:
      - /dev/video0:/dev/video0
    ports:
      - "5000:5000"
    environment:
      - DISPLAY=$DISPLAY
```

### 방법 5: 웹 기반 인터페이스 (가장 실용적)
브라우저에서 웹캠을 사용하고 서버로 전송하는 방식:

```html
<!-- webcam_interface.html -->
<!DOCTYPE html>
<html>
<head>
    <title>웹캠 감정 분석</title>
</head>
<body>
    <video id="video" width="640" height="480" autoplay></video>
    <canvas id="canvas" width="640" height="480"></canvas>
    <button id="capture">캡처 및 분석</button>
    
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
            });
        
        document.getElementById('capture').addEventListener('click', () => {
            ctx.drawImage(video, 0, 0, 640, 480);
            const imageData = canvas.toDataURL('image/jpeg');
            
            // 서버로 이미지 전송
            fetch('/analyze', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({image: imageData})
            })
            .then(response => response.json())
            .then(data => {
                console.log('분석 결과:', data);
            });
        });
    </script>
</body>
</html>
```

## 🚀 즉시 사용 가능한 해결책

### 1. 웹 스트리밍 서버 생성
```bash
# Flask 설치
pip install flask

# 웹 스트리밍 서버 실행
python3 webcam_streamer.py
```

### 2. 브라우저에서 접근
```
http://your-server-ip:5000
```

### 3. 모바일 앱 사용
- IP Webcam (Android)
- EpocCam (iOS)
- DroidCam (Android/iOS)

## ⚠️ 주의사항

1. **보안**: 웹 스트리밍 시 방화벽 설정 확인
2. **성능**: 네트워크 대역폭 고려
3. **지연**: 실시간 분석 시 지연 시간 고려

## 🎯 권장 방법

SSH 환경에서는 **웹 기반 인터페이스**가 가장 실용적입니다:

1. 로컬에서 웹캠 사용
2. 브라우저에서 실시간 스트리밍
3. 서버에서 분석 처리
4. 결과를 브라우저에 표시

이 방법을 구현해드릴까요?
