#!/usr/bin/env python3
"""
간단한 웹 서버 - Flask 없이 실행
원격 서버에서 웹캠 분석을 위한 최소한의 웹 서버
"""

import http.server
import socketserver
import json
import base64
import io
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import os
from datetime import datetime
import time
from urllib.parse import parse_qs, urlparse

class EmotionAnalyzer:
    def __init__(self, model_path='./models/face_landmarker.task'):
        self.model_path = model_path
        self.detector = None
        self.initialize_detector()
    
    def initialize_detector(self):
        try:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
            print("✅ MediaPipe 감지기 초기화 완료")
            return True
        except Exception as e:
            print(f"❌ MediaPipe 초기화 실패: {e}")
            return False
    
    def calculate_distance(self, point1, point2):
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def analyze_emotion(self, image_data):
        if not self.detector:
            return {"error": "감지기가 초기화되지 않았습니다."}
        
        try:
            # Base64 이미지 디코딩
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_np = np.array(image)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
            
            # 얼굴 랜드마크 감지
            detection_result = self.detector.detect(mp_image)
            
            if not detection_result.face_landmarks:
                return {"emotion": "unknown", "confidence": 0.0, "score": 50}
            
            # 감정 분석
            landmarks = detection_result.face_landmarks[0]
            
            # 주요 랜드마크 인덱스
            LEFT_EYE_INNER, LEFT_EYE_OUTER = 133, 33
            RIGHT_EYE_INNER, RIGHT_EYE_OUTER = 362, 263
            MOUTH_LEFT, MOUTH_RIGHT = 61, 291
            MOUTH_TOP, MOUTH_BOTTOM = 13, 14
            LEFT_EYEBROW_INNER, RIGHT_EYEBROW_INNER = 70, 300
            
            # 특징 계산
            left_eye_width = self.calculate_distance(landmarks[LEFT_EYE_INNER], landmarks[LEFT_EYE_OUTER])
            right_eye_width = self.calculate_distance(landmarks[RIGHT_EYE_INNER], landmarks[RIGHT_EYE_OUTER])
            avg_eye_width = (left_eye_width + right_eye_width) / 2
            
            mouth_width = self.calculate_distance(landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT])
            
            left_eyebrow_height = landmarks[LEFT_EYEBROW_INNER].y - landmarks[LEFT_EYE_INNER].y
            right_eyebrow_height = landmarks[RIGHT_EYEBROW_INNER].y - landmarks[RIGHT_EYE_INNER].y
            avg_eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2
            
            # 감정 점수 계산
            emotion_scores = {
                "happy": 0.0, "sad": 0.0, "angry": 0.0,
                "surprised": 0.0, "neutral": 0.0, "confident": 0.0
            }
            
            # 미소 감지
            mouth_corners_up = (landmarks[MOUTH_LEFT].y < landmarks[MOUTH_TOP].y and 
                               landmarks[MOUTH_RIGHT].y < landmarks[MOUTH_TOP].y)
            if mouth_corners_up and mouth_width > 0.02:
                emotion_scores["happy"] += 0.4
                emotion_scores["confident"] += 0.2
            
            # 기타 감정 분석
            if avg_eyebrow_height < -0.01:
                emotion_scores["sad"] += 0.3
            if avg_eyebrow_height < -0.005 and avg_eye_width < 0.025:
                emotion_scores["angry"] += 0.4
            if avg_eye_width > 0.035:
                emotion_scores["surprised"] += 0.4
            if avg_eye_width > 0.03 and mouth_corners_up:
                emotion_scores["confident"] += 0.3
            if (0.02 < avg_eye_width < 0.03 and 0.01 < mouth_width < 0.02 and 
                -0.005 < avg_eyebrow_height < 0.005):
                emotion_scores["neutral"] += 0.3
            
            # 최고 점수 감정 선택
            max_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = min(1.0, emotion_scores[max_emotion])
            
            # 점수 계산
            score_map = {"confident": 90, "happy": 85, "neutral": 70, 
                        "surprised": 60, "sad": 40, "angry": 30, "unknown": 50}
            score = score_map.get(max_emotion, 50)
            
            if confidence < 0.5:
                score *= 0.8
            
            return {
                "emotion": max_emotion,
                "confidence": round(confidence, 2),
                "score": int(score)
            }
            
        except Exception as e:
            return {"error": f"분석 오류: {str(e)}"}

class WebHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.analyzer = EmotionAnalyzer()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            
            html_content = '''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎭 실시간 면접 표정 분석</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1 { text-align: center; color: #333; }
        #video { border: 2px solid #ddd; border-radius: 10px; display: block; margin: 20px auto; }
        .controls { text-align: center; margin: 20px 0; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; margin: 0 10px; border-radius: 5px; cursor: pointer; font-size: 16px; }
        button:hover { background: #0056b3; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .results { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px; }
        .result-card { background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid #007bff; }
        .emotion { font-size: 2em; font-weight: bold; margin: 10px 0; }
        .score { font-size: 3em; font-weight: bold; color: #28a745; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; text-align: center; }
        .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .status.info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 실시간 면접 표정 분석</h1>
        
        <div id="status" class="status info">웹캠을 활성화하려면 "시작" 버튼을 클릭하세요</div>
        
        <video id="video" width="640" height="480" autoplay muted></video>
        
        <div class="controls">
            <button id="startBtn" onclick="startAnalysis()">🎥 시작</button>
            <button id="stopBtn" onclick="stopAnalysis()" disabled>⏹️ 중지</button>
            <button id="analyzeBtn" onclick="analyzeFrame()" disabled>📊 분석</button>
        </div>
        
        <div class="results">
            <div class="result-card">
                <h3>😊 현재 감정</h3>
                <div id="emotion" class="emotion">-</div>
            </div>
            <div class="result-card">
                <h3>📊 신뢰도</h3>
                <div id="confidence" class="score">0%</div>
            </div>
            <div class="result-card">
                <h3>🎯 면접 점수</h3>
                <div id="score" class="score">0</div>
            </div>
            <div class="result-card">
                <h3>💡 피드백</h3>
                <div id="feedback">분석을 시작하세요</div>
            </div>
        </div>
    </div>

    <canvas id="canvas" style="display: none;"></canvas>

    <script>
        let video = document.getElementById('video');
        let canvas = document.getElementById('canvas');
        let ctx = canvas.getContext('2d');
        let stream = null;
        
        const feedback = {
            'confident': '자신감 있는 표정! 👍',
            'happy': '밝고 긍정적! 😊',
            'neutral': '차분함. 미소를 더해보세요 😐',
            'surprised': '놀란 표정. 차분해보세요 😮',
            'sad': '더 밝은 표정을 연습해보세요 😢',
            'angry': '긴장을 풀고 릴렉스하세요 😠',
            'unknown': '얼굴이 잘 보이지 않습니다 🤔'
        };
        
        function updateStatus(message, type = 'info') {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = `status ${type}`;
        }
        
        async function startAnalysis() {
            try {
                updateStatus('웹캠에 접근하는 중...', 'info');
                
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: 640, height: 480, facingMode: 'user' } 
                });
                
                video.srcObject = stream;
                
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('analyzeBtn').disabled = false;
                
                updateStatus('웹캠이 활성화되었습니다. "분석" 버튼을 눌러 감정을 분석하세요.', 'success');
                
            } catch (error) {
                console.error('웹캠 접근 오류:', error);
                updateStatus('웹캠에 접근할 수 없습니다. 권한을 확인해주세요.', 'error');
            }
        }
        
        function stopAnalysis() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('analyzeBtn').disabled = true;
            
            updateStatus('분석이 중지되었습니다', 'info');
        }
        
        function analyzeFrame() {
            if (!stream) return;
            
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);
            
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            updateStatus('분석 중...', 'info');
            
            fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    updateStatus(`오류: ${data.error}`, 'error');
                } else {
                    updateResults(data);
                    updateStatus('분석 완료!', 'success');
                }
            })
            .catch(error => {
                console.error('분석 오류:', error);
                updateStatus('서버 연결 오류', 'error');
            });
        }
        
        function updateResults(data) {
            document.getElementById('emotion').textContent = data.emotion;
            document.getElementById('confidence').textContent = Math.round(data.confidence * 100) + '%';
            document.getElementById('score').textContent = data.score;
            document.getElementById('feedback').textContent = feedback[data.emotion] || '분석 중...';
        }
        
        window.addEventListener('beforeunload', () => {
            stopAnalysis();
        });
    </script>
</body>
</html>
            '''
            self.wfile.write(html_content.encode('utf-8'))
        else:
            super().do_GET()
    
    def do_POST(self):
        if self.path == '/analyze':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                result = self.analyzer.analyze_emotion(data.get('image', ''))
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                self.wfile.write(json.dumps(result).encode('utf-8'))
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                error_response = {"error": f"서버 오류: {str(e)}"}
                self.wfile.write(json.dumps(error_response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def main():
    PORT = 8000
    
    print("🎭 간단한 웹 기반 감정 분석 서버")
    print("="*50)
    
    # MediaPipe 초기화 테스트
    analyzer = EmotionAnalyzer()
    if not analyzer.detector:
        print("❌ MediaPipe 초기화 실패. 서버를 시작할 수 없습니다.")
        return
    
    try:
        with socketserver.TCPServer(("", PORT), WebHandler) as httpd:
            print(f"✅ 서버가 포트 {PORT}에서 실행 중입니다")
            print(f"🌐 브라우저에서 접속: http://localhost:{PORT}")
            print("📝 종료하려면 Ctrl+C를 누르세요")
            print("="*50)
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n⏹️ 서버가 중지되었습니다.")
    except Exception as e:
        print(f"❌ 서버 오류: {e}")

if __name__ == "__main__":
    main()
