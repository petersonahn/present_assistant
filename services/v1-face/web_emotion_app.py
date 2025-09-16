#!/usr/bin/env python3
"""
웹 기반 실시간 감정 분석 애플리케이션
원격 서버에서 웹 서버를 띄우고, 클라이언트의 웹캠을 사용하여 감정을 분석합니다.
"""

from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import base64
import io
import time
import os
import json
from datetime import datetime
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
from PIL import Image

app = Flask(__name__)

class WebEmotionAnalyzer:
    def __init__(self, model_path='./models/face_landmarker.task'):
        """웹 기반 감정 분석기를 초기화합니다."""
        self.model_path = model_path
        self.detector = None
        self.session_data = []
        self.session_start_time = None
        self.frame_count = 0
        
        # 결과 저장 폴더 생성
        self.setup_output_folders()
        
        # MediaPipe 초기화
        self.initialize_detector()
    
    def setup_output_folders(self):
        """결과 저장을 위한 폴더 구조를 생성합니다."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_folder = f"web_session_{timestamp}"
        
        self.folders = {
            'session': f"results/{self.session_folder}",
            'frames': f"results/{self.session_folder}/frames",
            'analysis': f"results/{self.session_folder}/analysis",
            'reports': f"results/{self.session_folder}/reports"
        }
        
        for folder in self.folders.values():
            os.makedirs(folder, exist_ok=True)
        
        print(f"📁 웹 세션 폴더 생성: {self.session_folder}")
    
    def initialize_detector(self):
        """MediaPipe 얼굴 랜드마크 감지기를 초기화합니다."""
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
            self.detector = None
            return False
    
    def calculate_distance(self, point1, point2):
        """두 점 사이의 거리를 계산합니다."""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def analyze_emotion_from_landmarks(self, face_landmarks):
        """얼굴 랜드마크에서 감정을 분석합니다."""
        if not face_landmarks or len(face_landmarks) == 0:
            return {"emotion": "unknown", "confidence": 0.0, "details": {}}
        
        landmarks = face_landmarks[0]
        
        # 주요 랜드마크 인덱스
        LEFT_EYE_INNER = 133
        LEFT_EYE_OUTER = 33
        RIGHT_EYE_INNER = 362
        RIGHT_EYE_OUTER = 263
        MOUTH_LEFT = 61
        MOUTH_RIGHT = 291
        MOUTH_TOP = 13
        MOUTH_BOTTOM = 14
        LEFT_EYEBROW_INNER = 70
        RIGHT_EYEBROW_INNER = 300
        
        try:
            # 눈 크기 분석
            left_eye_width = self.calculate_distance(landmarks[LEFT_EYE_INNER], landmarks[LEFT_EYE_OUTER])
            right_eye_width = self.calculate_distance(landmarks[RIGHT_EYE_INNER], landmarks[RIGHT_EYE_OUTER])
            avg_eye_width = (left_eye_width + right_eye_width) / 2
            
            # 입 크기 분석
            mouth_width = self.calculate_distance(landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT])
            mouth_height = self.calculate_distance(landmarks[MOUTH_TOP], landmarks[MOUTH_BOTTOM])
            
            # 눈썹 높이 분석
            left_eyebrow_height = landmarks[LEFT_EYEBROW_INNER].y - landmarks[LEFT_EYE_INNER].y
            right_eyebrow_height = landmarks[RIGHT_EYEBROW_INNER].y - landmarks[RIGHT_EYE_INNER].y
            avg_eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2
            
            # 감정 분석 로직
            emotion_scores = {
                "happy": 0.0,
                "sad": 0.0,
                "angry": 0.0,
                "surprised": 0.0,
                "neutral": 0.0,
                "confident": 0.0
            }
            
            # 미소 감지
            mouth_corners_up = (landmarks[MOUTH_LEFT].y < landmarks[MOUTH_TOP].y and 
                               landmarks[MOUTH_RIGHT].y < landmarks[MOUTH_TOP].y)
            if mouth_corners_up and mouth_width > 0.02:
                emotion_scores["happy"] += 0.4
                emotion_scores["confident"] += 0.2
            
            # 슬픔 감지
            if avg_eyebrow_height < -0.01:
                emotion_scores["sad"] += 0.3
            if not mouth_corners_up and mouth_width < 0.015:
                emotion_scores["sad"] += 0.2
            
            # 화남 감지
            if avg_eyebrow_height < -0.005 and avg_eye_width < 0.025:
                emotion_scores["angry"] += 0.4
            
            # 놀람 감지
            if avg_eye_width > 0.035:
                emotion_scores["surprised"] += 0.4
            
            # 자신감 감지
            if avg_eye_width > 0.03 and mouth_corners_up:
                emotion_scores["confident"] += 0.3
            
            # 중립 감지
            if (0.02 < avg_eye_width < 0.03 and 
                0.01 < mouth_width < 0.02 and 
                -0.005 < avg_eyebrow_height < 0.005):
                emotion_scores["neutral"] += 0.3
            
            # 가장 높은 점수의 감정 선택
            max_emotion = max(emotion_scores, key=emotion_scores.get)
            max_score = emotion_scores[max_emotion]
            confidence = min(1.0, max_score)
            
            return {
                "emotion": max_emotion,
                "confidence": confidence,
                "details": {
                    "eye_width": avg_eye_width,
                    "mouth_width": mouth_width,
                    "mouth_height": mouth_height,
                    "eyebrow_height": avg_eyebrow_height,
                    "all_scores": emotion_scores
                }
            }
            
        except Exception as e:
            print(f"감정 분석 중 오류: {e}")
            return {"emotion": "unknown", "confidence": 0.0, "details": {}}
    
    def get_interview_score(self, emotion_data):
        """면접 점수를 계산합니다."""
        emotion = emotion_data["emotion"]
        confidence = emotion_data["confidence"]
        
        score_map = {
            "confident": 90,
            "happy": 85,
            "neutral": 70,
            "surprised": 60,
            "sad": 40,
            "angry": 30,
            "unknown": 50
        }
        
        base_score = score_map.get(emotion, 50)
        
        # 신뢰도에 따른 조정
        if confidence < 0.5:
            base_score *= 0.8
        
        return min(100, max(0, int(base_score)))
    
    def analyze_frame(self, image_data):
        """웹에서 전송받은 이미지를 분석합니다."""
        if not self.detector:
            return {"error": "MediaPipe 감지기가 초기화되지 않았습니다."}
        
        try:
            # Base64 이미지 디코딩
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # RGB로 변환
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # numpy 배열로 변환
            image_np = np.array(image)
            
            # MediaPipe 이미지 생성
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
            
            # 얼굴 랜드마크 감지
            detection_result = self.detector.detect(mp_image)
            
            # 감정 분석
            emotion_data = self.analyze_emotion_from_landmarks(detection_result.face_landmarks)
            score = self.get_interview_score(emotion_data)
            
            # 프레임 카운트 증가
            self.frame_count += 1
            
            # 세션 데이터 저장
            frame_data = {
                "frame_number": self.frame_count,
                "timestamp": time.time(),
                "emotion": emotion_data["emotion"],
                "confidence": emotion_data["confidence"],
                "score": score,
                "details": emotion_data["details"]
            }
            
            self.session_data.append(frame_data)
            
            # 결과 반환
            return {
                "success": True,
                "emotion": emotion_data["emotion"],
                "confidence": round(emotion_data["confidence"], 2),
                "score": score,
                "frame_count": self.frame_count,
                "details": emotion_data["details"]
            }
            
        except Exception as e:
            return {"error": f"분석 중 오류 발생: {str(e)}"}

# 전역 분석기 인스턴스
analyzer = WebEmotionAnalyzer()

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('webcam_analyzer.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """이미지 분석 API"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({"error": "이미지 데이터가 없습니다."})
        
        result = analyzer.analyze_frame(image_data)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"서버 오류: {str(e)}"})

@app.route('/session_info')
def session_info():
    """세션 정보 API"""
    return jsonify({
        "session_folder": analyzer.session_folder,
        "frame_count": analyzer.frame_count,
        "total_data": len(analyzer.session_data)
    })

if __name__ == '__main__':
    print("🎭 웹 기반 실시간 감정 분석 서버")
    print("="*50)
    
    if not analyzer.detector:
        print("❌ MediaPipe 초기화 실패. 서버를 시작할 수 없습니다.")
        exit(1)
    
    print("✅ 서버 초기화 완료")
    print("🌐 브라우저에서 접속: http://localhost:5000")
    print("📝 종료하려면 Ctrl+C를 누르세요")
    
    # Flask 앱 실행
    app.run(host='0.0.0.0', port=5000, debug=False)
