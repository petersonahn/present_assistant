#!/usr/bin/env python3
"""
WSL2 환경에서 웹캠을 사용한 실시간 감정 분석
다양한 웹캠 접근 방법을 시도합니다.
"""

import cv2
import numpy as np
import time
import os
import json
from datetime import datetime
import math
from typing import List, Dict, Tuple
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class WebcamAnalyzer:
    def __init__(self, model_path='./models/face_landmarker.task'):
        """웹캠 분석기를 초기화합니다."""
        self.model_path = model_path
        self.detector = None
        self.session_data = []
        self.session_start_time = None
        self.frame_count = 0
        self.cap = None
        
        # 결과 저장 폴더 생성
        self.setup_output_folders()
        
        # MediaPipe 초기화
        self.initialize_detector()
    
    def setup_output_folders(self):
        """결과 저장을 위한 폴더 구조를 생성합니다."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_folder = f"webcam_{timestamp}"
        
        self.folders = {
            'session': f"results/{self.session_folder}",
            'frames': f"results/{self.session_folder}/frames",
            'analysis': f"results/{self.session_folder}/analysis",
            'reports': f"results/{self.session_folder}/reports"
        }
        
        for folder in self.folders.values():
            os.makedirs(folder, exist_ok=True)
        
        print(f"📁 웹캠 세션 폴더 생성: {self.session_folder}")
    
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
        except Exception as e:
            print(f"❌ MediaPipe 초기화 실패: {e}")
            self.detector = None
    
    def find_webcam(self):
        """사용 가능한 웹캠을 찾습니다."""
        print("🔍 웹캠을 찾는 중...")
        
        # 다양한 웹캠 인덱스 시도
        for camera_index in range(5):  # 0-4까지 시도
            print(f"   카메라 {camera_index} 시도 중...")
            cap = cv2.VideoCapture(camera_index)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ 카메라 {camera_index} 발견!")
                    return cap, camera_index
                else:
                    cap.release()
            else:
                cap.release()
        
        # USB 웹캠 시도
        usb_cameras = ['/dev/video0', '/dev/video1', '/dev/video2']
        for camera_path in usb_cameras:
            if os.path.exists(camera_path):
                print(f"   USB 카메라 {camera_path} 시도 중...")
                cap = cv2.VideoCapture(camera_path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✅ USB 카메라 {camera_path} 발견!")
                        return cap, camera_path
                    else:
                        cap.release()
        
        print("❌ 사용 가능한 웹캠을 찾을 수 없습니다.")
        return None, None
    
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
        LEFT_EYEBROW_OUTER = 46
        RIGHT_EYEBROW_INNER = 300
        RIGHT_EYEBROW_OUTER = 276
        
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
    
    def draw_landmarks_and_info(self, frame, detection_result, emotion_data, score):
        """프레임에 랜드마크와 정보를 그립니다."""
        annotated_frame = frame.copy()
        
        if detection_result.face_landmarks:
            # 얼굴 랜드마크 그리기
            for face_landmarks in detection_result.face_landmarks:
                for landmark in face_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(annotated_frame, (x, y), 1, (0, 255, 0), -1)
        
        # 정보 텍스트 그리기
        emotion = emotion_data["emotion"]
        confidence = emotion_data["confidence"]
        
        # 배경 사각형
        cv2.rectangle(annotated_frame, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.rectangle(annotated_frame, (10, 10), (400, 120), (255, 255, 255), 2)
        
        # 텍스트
        cv2.putText(annotated_frame, f"Emotion: {emotion}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Confidence: {confidence:.2f}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Score: {score}/100", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Frame: {self.frame_count}", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame
    
    def save_frame_data(self, frame, emotion_data, score):
        """프레임 데이터를 저장합니다."""
        timestamp = time.time()
        
        frame_data = {
            "frame_number": self.frame_count,
            "timestamp": timestamp,
            "emotion": emotion_data["emotion"],
            "confidence": emotion_data["confidence"],
            "score": score,
            "details": emotion_data["details"]
        }
        
        self.session_data.append(frame_data)
        
        # 주기적으로 프레임 이미지 저장 (매 30프레임마다)
        if self.frame_count % 30 == 0:
            frame_filename = f"frame_{self.frame_count:06d}.jpg"
            frame_path = os.path.join(self.folders['frames'], frame_filename)
            cv2.imwrite(frame_path, frame)
    
    def run_webcam_analysis(self):
        """웹캠 분석을 실행합니다."""
        if not self.detector:
            print("❌ 감지기가 초기화되지 않았습니다.")
            return
        
        # 웹캠 찾기
        self.cap, camera_info = self.find_webcam()
        if self.cap is None:
            print("❌ 웹캠을 찾을 수 없습니다.")
            print("💡 해결 방법:")
            print("   1. 웹캠이 연결되어 있는지 확인하세요")
            print("   2. 다른 프로그램에서 웹캠을 사용하고 있지 않은지 확인하세요")
            print("   3. Windows에서 직접 실행해보세요: run_on_windows.bat")
            return
        
        print(f"🎥 웹캠 분석을 시작합니다... (카메라: {camera_info})")
        print("📝 종료하려면 'q' 키를 누르거나 Ctrl+C를 누르세요")
        
        self.session_start_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ 프레임을 읽을 수 없습니다.")
                    break
                
                self.frame_count += 1
                
                # MediaPipe용 이미지 변환
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # 얼굴 랜드마크 감지
                detection_result = self.detector.detect(mp_image)
                
                # 감정 분석
                emotion_data = self.analyze_emotion_from_landmarks(detection_result.face_landmarks)
                score = self.get_interview_score(emotion_data)
                
                # 프레임에 정보 그리기
                annotated_frame = self.draw_landmarks_and_info(frame, detection_result, emotion_data, score)
                
                # 프레임 데이터 저장
                self.save_frame_data(frame, emotion_data, score)
                
                # 화면에 표시
                try:
                    cv2.imshow('Webcam Emotion Analysis', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    print(f"GUI 표시 오류 (정상): {e}")
                    # GUI 표시 실패 시에도 계속 진행
                
                # 진행률 표시 (매 30프레임마다)
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.session_start_time
                    print(f"\r⏱️ 분석 중... {self.frame_count} 프레임 ({elapsed:.1f}초)", end="", flush=True)
                
        except KeyboardInterrupt:
            print("\n⏹️ 사용자에 의해 중단되었습니다.")
        
        finally:
            if self.cap:
                self.cap.release()
            try:
                cv2.destroyAllWindows()
            except:
                pass
            self.generate_summary_report()
    
    def generate_summary_report(self):
        """종합 분석 결과 리포트를 생성합니다."""
        if not self.session_data:
            print("❌ 분석할 데이터가 없습니다.")
            return
        
        print("\n" + "="*60)
        print("📊 종합 분석 결과 생성 중...")
        print("="*60)
        
        # 기본 통계 계산
        total_frames = len(self.session_data)
        session_duration = time.time() - self.session_start_time if self.session_start_time else 0
        
        # 감정별 통계
        emotion_counts = {}
        total_score = 0
        confidence_sum = 0
        
        for data in self.session_data:
            emotion = data["emotion"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            total_score += data["score"]
            confidence_sum += data["confidence"]
        
        # 평균 계산
        avg_score = total_score / total_frames if total_frames > 0 else 0
        avg_confidence = confidence_sum / total_frames if total_frames > 0 else 0
        
        # 가장 많이 나타난 감정
        dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "unknown"
        
        # 리포트 데이터 생성
        report = {
            "session_info": {
                "session_folder": self.session_folder,
                "start_time": datetime.fromtimestamp(self.session_start_time).isoformat() if self.session_start_time else None,
                "duration_seconds": session_duration,
                "total_frames": total_frames,
                "fps": total_frames / session_duration if session_duration > 0 else 0,
                "mode": "webcam"
            },
            "analysis_results": {
                "average_score": round(avg_score, 2),
                "average_confidence": round(avg_confidence, 2),
                "dominant_emotion": dominant_emotion,
                "emotion_distribution": emotion_counts,
                "score_range": {
                    "min": min(data["score"] for data in self.session_data),
                    "max": max(data["score"] for data in self.session_data)
                }
            },
            "recommendations": self.generate_recommendations(avg_score, dominant_emotion, emotion_counts),
            "frame_data": self.session_data
        }
        
        # JSON 파일로 저장
        report_path = os.path.join(self.folders['reports'], 'summary_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 텍스트 리포트 생성
        self.generate_text_report(report)
        
        print(f"✅ 종합 분석 결과가 저장되었습니다:")
        print(f"   📁 세션 폴더: {self.session_folder}")
        print(f"   📊 평균 점수: {avg_score:.1f}/100")
        print(f"   😊 주요 감정: {dominant_emotion}")
        print(f"   📈 신뢰도: {avg_confidence:.2f}")
        print(f"   📁 결과 위치: results/{self.session_folder}/")
    
    def generate_recommendations(self, avg_score, dominant_emotion, emotion_counts):
        """개선 권장사항을 생성합니다."""
        recommendations = []
        
        if avg_score >= 80:
            recommendations.append("훌륭한 면접 표정입니다! 현재 상태를 유지하세요.")
        elif avg_score >= 60:
            recommendations.append("좋은 표정이지만 더 개선할 여지가 있습니다.")
        else:
            recommendations.append("면접 표정을 개선할 필요가 있습니다.")
        
        if dominant_emotion == "sad":
            recommendations.append("더 밝고 긍정적인 표정을 연습해보세요.")
        elif dominant_emotion == "angry":
            recommendations.append("긴장을 풀고 차분한 표정을 연습해보세요.")
        elif dominant_emotion == "neutral":
            recommendations.append("자연스러운 미소를 추가해보세요.")
        
        if emotion_counts.get("confident", 0) < emotion_counts.get("neutral", 0):
            recommendations.append("자신감 있는 표정을 더 자주 연습해보세요.")
        
        return recommendations
    
    def generate_text_report(self, report):
        """텍스트 형태의 리포트를 생성합니다."""
        report_path = os.path.join(self.folders['reports'], 'summary_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("🎭 면접 표정 분석 종합 리포트 (웹캠)\n")
            f.write("="*60 + "\n\n")
            
            # 세션 정보
            f.write("📋 세션 정보:\n")
            f.write(f"   세션 폴더: {report['session_info']['session_folder']}\n")
            f.write(f"   분석 모드: {report['session_info']['mode']}\n")
            f.write(f"   분석 시간: {report['session_info']['duration_seconds']:.1f}초\n")
            f.write(f"   총 프레임: {report['session_info']['total_frames']}\n")
            f.write(f"   평균 FPS: {report['session_info']['fps']:.1f}\n\n")
            
            # 분석 결과
            f.write("📊 분석 결과:\n")
            f.write(f"   평균 점수: {report['analysis_results']['average_score']}/100\n")
            f.write(f"   평균 신뢰도: {report['analysis_results']['average_confidence']:.2f}\n")
            f.write(f"   주요 감정: {report['analysis_results']['dominant_emotion']}\n")
            f.write(f"   점수 범위: {report['analysis_results']['score_range']['min']}-{report['analysis_results']['score_range']['max']}\n\n")
            
            # 감정 분포
            f.write("😊 감정 분포:\n")
            for emotion, count in report['analysis_results']['emotion_distribution'].items():
                percentage = (count / report['session_info']['total_frames']) * 100
                f.write(f"   {emotion}: {count}회 ({percentage:.1f}%)\n")
            f.write("\n")
            
            # 권장사항
            f.write("💡 개선 권장사항:\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"   {i}. {rec}\n")
            f.write("\n")
            
            f.write("="*60 + "\n")
            f.write("분석 완료 시간: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("="*60 + "\n")

def main():
    """메인 실행 함수"""
    print("🎭 웹캠 면접 표정 분석 시스템")
    print("="*50)
    print("📝 이 프로그램은 웹캠을 사용하여 실시간으로 감정을 분석합니다.")
    print("="*50)
    
    # 결과 폴더 생성
    os.makedirs("results", exist_ok=True)
    
    # 분석기 초기화 및 실행
    analyzer = WebcamAnalyzer()
    analyzer.run_webcam_analysis()

if __name__ == "__main__":
    main()
