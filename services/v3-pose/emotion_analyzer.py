import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

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
        except Exception as e:
            print(f"❌ MediaPipe 감정 분석기 초기화 실패: {e}")
            self.detector = None

    def calculate_distance(self, point1, point2):
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def analyze_emotion_from_landmarks(self, face_landmarks):
        if not face_landmarks or len(face_landmarks) == 0:
            return {"emotion": "unknown", "confidence": 0.0, "details": {}}
        landmarks = face_landmarks[0]
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
            left_eye_width = self.calculate_distance(landmarks[LEFT_EYE_INNER], landmarks[LEFT_EYE_OUTER])
            right_eye_width = self.calculate_distance(landmarks[RIGHT_EYE_INNER], landmarks[RIGHT_EYE_OUTER])
            avg_eye_width = (left_eye_width + right_eye_width) / 2
            mouth_width = self.calculate_distance(landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT])
            mouth_height = self.calculate_distance(landmarks[MOUTH_TOP], landmarks[MOUTH_BOTTOM])
            left_eyebrow_height = landmarks[LEFT_EYEBROW_INNER].y - landmarks[LEFT_EYE_INNER].y
            right_eyebrow_height = landmarks[RIGHT_EYEBROW_INNER].y - landmarks[RIGHT_EYE_INNER].y
            avg_eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2
            emotion_scores = {
                "happy": 0.0,
                "sad": 0.0,
                "angry": 0.0,
                "surprised": 0.0,
                "neutral": 0.0,
                "confident": 0.0
            }
            mouth_corners_up = (landmarks[MOUTH_LEFT].y < landmarks[MOUTH_TOP].y and 
                               landmarks[MOUTH_RIGHT].y < landmarks[MOUTH_TOP].y)
            if mouth_corners_up and mouth_width > 0.02:
                emotion_scores["happy"] += 0.4
                emotion_scores["confident"] += 0.2
            if avg_eyebrow_height < -0.01:
                emotion_scores["sad"] += 0.3
            if not mouth_corners_up and mouth_width < 0.015:
                emotion_scores["sad"] += 0.2
            if avg_eyebrow_height < -0.005 and avg_eye_width < 0.025:
                emotion_scores["angry"] += 0.4
            if avg_eye_width > 0.035:
                emotion_scores["surprised"] += 0.4
            if avg_eye_width > 0.03 and mouth_corners_up:
                emotion_scores["confident"] += 0.3
            if (0.02 < avg_eye_width < 0.03 and 
                0.01 < mouth_width < 0.02 and 
                -0.005 < avg_eyebrow_height < 0.005):
                emotion_scores["neutral"] += 0.3
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

    def analyze(self, image_np):
        """OpenCV/Numpy 이미지 입력 → 감정 분석 결과 반환"""
        if self.detector is None:
            return {"emotion": "unknown", "confidence": 0.0, "details": {}, "error": "감정 분석기 미초기화"}
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        detection_result = self.detector.detect(mp_image)
        return self.analyze_emotion_from_landmarks(detection_result.face_landmarks)
