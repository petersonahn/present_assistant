#!/usr/bin/env python3
"""
시뮬레이션 모드 실시간 면접 표정 분석
웹캠이 없는 환경에서 테스트 이미지를 사용하여 시뮬레이션합니다.
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

class SimulatedRealtimeAnalyzer:
    def __init__(self, model_path='./models/face_landmarker.task', test_image='face_psy.jpg'):
        """시뮬레이션 모드 분석기를 초기화합니다."""
        self.model_path = model_path
        self.test_image_path = test_image
        self.detector = None
        self.session_data = []
        self.session_start_time = None
        self.frame_count = 0
        self.simulation_duration = 30  # 30초 시뮬레이션
        
        # 결과 저장 폴더 생성
        self.setup_output_folders()
        
        # MediaPipe 초기화
        self.initialize_detector()
    
    def setup_output_folders(self):
        """결과 저장을 위한 폴더 구조를 생성합니다."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_folder = f"simulation_{timestamp}"
        
        self.folders = {
            'session': f"results/{self.session_folder}",
            'frames': f"results/{self.session_folder}/frames",
            'analysis': f"results/{self.session_folder}/analysis",
            'reports': f"results/{self.session_folder}/reports"
        }
        
        for folder in self.folders.values():
            os.makedirs(folder, exist_ok=True)
        
        print(f"📁 시뮬레이션 세션 폴더 생성: {self.session_folder}")
    
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
    
    def add_emotion_variation(self, base_emotion_data, frame_count):
        """시뮬레이션을 위해 감정에 변화를 추가합니다."""
        # 시간에 따른 감정 변화 시뮬레이션
        time_factor = frame_count / 30.0  # 30프레임당 1초
        
        # 기본 감정 데이터 복사
        emotion_data = base_emotion_data.copy()
        details = emotion_data["details"].copy()
        all_scores = details["all_scores"].copy()
        
        # 시간에 따른 감정 변화
        if time_factor < 5:  # 처음 5초: 자신감
            all_scores["confident"] += 0.2
        elif time_factor < 10:  # 5-10초: 미소
            all_scores["happy"] += 0.3
        elif time_factor < 15:  # 10-15초: 중립
            all_scores["neutral"] += 0.2
        elif time_factor < 20:  # 15-20초: 약간의 긴장
            all_scores["surprised"] += 0.1
        else:  # 20초 이후: 다시 자신감
            all_scores["confident"] += 0.1
        
        # 가장 높은 점수의 감정 재계산
        max_emotion = max(all_scores, key=all_scores.get)
        max_score = all_scores[max_emotion]
        confidence = min(1.0, max_score)
        
        emotion_data["emotion"] = max_emotion
        emotion_data["confidence"] = confidence
        details["all_scores"] = all_scores
        emotion_data["details"] = details
        
        return emotion_data
    
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
        
        # 시뮬레이션 모드 표시
        cv2.putText(annotated_frame, "SIMULATION MODE", (frame.shape[1] - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
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
        
        # 주기적으로 프레임 이미지 저장 (매 10프레임마다)
        if self.frame_count % 10 == 0:
            frame_filename = f"frame_{self.frame_count:06d}.jpg"
            frame_path = os.path.join(self.folders['frames'], frame_filename)
            cv2.imwrite(frame_path, frame)
    
    def run_simulation(self):
        """시뮬레이션을 실행합니다."""
        if not self.detector:
            print("❌ 감지기가 초기화되지 않았습니다.")
            return
        
        # 테스트 이미지 로드
        if not os.path.exists(self.test_image_path):
            print(f"❌ 테스트 이미지를 찾을 수 없습니다: {self.test_image_path}")
            return
        
        print("🎥 시뮬레이션 모드로 분석을 시작합니다...")
        print("📝 30초간 시뮬레이션됩니다. (Ctrl+C로 중단 가능)")
        
        self.session_start_time = time.time()
        
        try:
            # 기본 이미지 로드 및 분석
            image = mp.Image.create_from_file(self.test_image_path)
            detection_result = self.detector.detect(image)
            base_emotion_data = self.analyze_emotion_from_landmarks(detection_result.face_landmarks)
            
            # OpenCV 이미지로 변환
            frame = cv2.imread(self.test_image_path)
            if frame is None:
                print("❌ 이미지를 읽을 수 없습니다.")
                return
            
            # 시뮬레이션 루프
            start_time = time.time()
            while time.time() - start_time < self.simulation_duration:
                self.frame_count += 1
                
                # 감정 변화 시뮬레이션
                emotion_data = self.add_emotion_variation(base_emotion_data, self.frame_count)
                score = self.get_interview_score(emotion_data)
                
                # 프레임에 정보 그리기
                annotated_frame = self.draw_landmarks_and_info(frame, detection_result, emotion_data, score)
                
                # 프레임 데이터 저장
                self.save_frame_data(annotated_frame, emotion_data, score)
                
                # 화면에 표시 (WSL2에서는 제한적)
                # GUI 표시를 시도하지 않고 파일로만 저장
                pass
                
                # 진행률 표시
                elapsed = time.time() - start_time
                progress = (elapsed / self.simulation_duration) * 100
                print(f"\r⏱️ 진행률: {progress:.1f}% ({elapsed:.1f}s/{self.simulation_duration}s)", end="", flush=True)
                
                time.sleep(0.1)  # 10 FPS 시뮬레이션
            
            print(f"\n✅ 시뮬레이션 완료! ({self.frame_count} 프레임 분석)")
            
        except KeyboardInterrupt:
            print("\n⏹️ 사용자에 의해 중단되었습니다.")
        
        finally:
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
                "mode": "simulation"
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
            f.write("🎭 면접 표정 분석 종합 리포트 (시뮬레이션)\n")
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
    print("🎭 시뮬레이션 모드 면접 표정 분석 시스템")
    print("="*50)
    print("📝 이 모드는 웹캠이 없는 환경에서 테스트 이미지를 사용하여")
    print("   실시간 분석을 시뮬레이션합니다.")
    print("="*50)
    
    # 결과 폴더 생성
    os.makedirs("results", exist_ok=True)
    
    # 분석기 초기화 및 실행
    analyzer = SimulatedRealtimeAnalyzer()
    analyzer.run_simulation()

if __name__ == "__main__":
    main()
