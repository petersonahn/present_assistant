import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from typing import List, Dict, Tuple

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def calculate_distance(point1, point2):
    """두 점 사이의 거리를 계산합니다."""
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def calculate_angle(p1, p2, p3):
    """세 점으로 이루어진 각도를 계산합니다."""
    a = calculate_distance(p2, p3)
    b = calculate_distance(p1, p3)
    c = calculate_distance(p1, p2)
    
    if a == 0 or b == 0 or c == 0:
        return 0
    
    cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
    cos_angle = max(-1, min(1, cos_angle))  # 범위 제한
    return math.degrees(math.acos(cos_angle))

def analyze_emotion_from_landmarks(face_landmarks):
    """얼굴 랜드마크에서 감정을 분석합니다."""
    if not face_landmarks or len(face_landmarks) == 0:
        return {"emotion": "unknown", "confidence": 0.0, "details": {}}
    
    landmarks = face_landmarks[0]  # 첫 번째 얼굴 사용
    
    # 주요 랜드마크 인덱스 (MediaPipe Face Mesh 기준)
    # 눈썹, 눈, 코, 입 관련 랜드마크
    LEFT_EYE_INNER = 133
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263
    NOSE_TIP = 1
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14
    LEFT_EYEBROW_INNER = 70
    LEFT_EYEBROW_OUTER = 46
    RIGHT_EYEBROW_INNER = 300
    RIGHT_EYEBROW_OUTER = 276
    
    try:
        # 눈 크기 분석 (눈을 뜨고 있는지)
        left_eye_width = calculate_distance(landmarks[LEFT_EYE_INNER], landmarks[LEFT_EYE_OUTER])
        right_eye_width = calculate_distance(landmarks[RIGHT_EYE_INNER], landmarks[RIGHT_EYE_OUTER])
        avg_eye_width = (left_eye_width + right_eye_width) / 2
        
        # 입 크기 분석
        mouth_width = calculate_distance(landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT])
        mouth_height = calculate_distance(landmarks[MOUTH_TOP], landmarks[MOUTH_BOTTOM])
        
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
        
        # 미소 감지 (입꼬리가 올라감)
        mouth_corners_up = (landmarks[MOUTH_LEFT].y < landmarks[MOUTH_TOP].y and 
                           landmarks[MOUTH_RIGHT].y < landmarks[MOUTH_TOP].y)
        if mouth_corners_up and mouth_width > 0.02:  # 입이 열려있고 미소
            emotion_scores["happy"] += 0.4
            emotion_scores["confident"] += 0.2
        
        # 슬픔 감지 (눈썹이 내려가고 입꼬리가 내려감)
        if avg_eyebrow_height < -0.01:  # 눈썹이 내려감
            emotion_scores["sad"] += 0.3
        if not mouth_corners_up and mouth_width < 0.015:  # 입이 작고 내려감
            emotion_scores["sad"] += 0.2
        
        # 화남 감지 (눈썹이 내려가고 눈이 좁아짐)
        if avg_eyebrow_height < -0.005 and avg_eye_width < 0.025:  # 눈썹 내려가고 눈 좁음
            emotion_scores["angry"] += 0.4
        
        # 놀람 감지 (눈이 크게 뜨임)
        if avg_eye_width > 0.035:  # 눈이 크게 뜨임
            emotion_scores["surprised"] += 0.4
        
        # 자신감 감지 (눈이 크게 뜨이고 미소)
        if avg_eye_width > 0.03 and mouth_corners_up:
            emotion_scores["confident"] += 0.3
        
        # 중립 감지 (기본 상태)
        if (0.02 < avg_eye_width < 0.03 and 
            0.01 < mouth_width < 0.02 and 
            -0.005 < avg_eyebrow_height < 0.005):
            emotion_scores["neutral"] += 0.3
        
        # 가장 높은 점수의 감정 선택
        max_emotion = max(emotion_scores, key=emotion_scores.get)
        max_score = emotion_scores[max_emotion]
        
        # 신뢰도 계산 (0-1 범위)
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
        print(f"감정 분석 중 오류 발생: {e}")
        return {"emotion": "unknown", "confidence": 0.0, "details": {}}

def get_interview_feedback(emotion_data):
    """면접 피드백을 제공합니다."""
    emotion = emotion_data["emotion"]
    confidence = emotion_data["confidence"]
    
    feedback = {
        "overall_score": 0,
        "feedback": "",
        "suggestions": []
    }
    
    if emotion == "confident":
        feedback["overall_score"] = 90
        feedback["feedback"] = "자신감 있는 표정입니다! 좋은 면접 태도입니다."
        feedback["suggestions"] = ["계속 이 표정을 유지하세요."]
        
    elif emotion == "happy":
        feedback["overall_score"] = 85
        feedback["feedback"] = "긍정적이고 밝은 표정입니다."
        feedback["suggestions"] = ["자연스러운 미소를 유지하세요."]
        
    elif emotion == "neutral":
        feedback["overall_score"] = 70
        feedback["feedback"] = "차분한 표정입니다."
        feedback["suggestions"] = ["조금 더 밝은 표정을 연습해보세요.", "자연스러운 미소를 추가해보세요."]
        
    elif emotion == "surprised":
        feedback["overall_score"] = 60
        feedback["feedback"] = "놀란 표정이 보입니다."
        feedback["suggestions"] = ["더 차분하고 안정적인 표정을 연습해보세요."]
        
    elif emotion == "sad":
        feedback["overall_score"] = 40
        feedback["feedback"] = "우울하거나 피곤해 보입니다."
        feedback["suggestions"] = ["충분한 휴식을 취하세요.", "긍정적인 마음가짐을 가져보세요."]
        
    elif emotion == "angry":
        feedback["overall_score"] = 30
        feedback["feedback"] = "화가 나거나 긴장된 표정입니다."
        feedback["suggestions"] = ["심호흡을 하고 긴장을 풀어보세요.", "긍정적인 생각을 해보세요."]
        
    else:
        feedback["overall_score"] = 50
        feedback["feedback"] = "표정을 명확히 분석하기 어렵습니다."
        feedback["suggestions"] = ["더 명확한 표정을 연습해보세요."]
    
    # 신뢰도에 따른 조정
    if confidence < 0.5:
        feedback["suggestions"].append("더 명확한 표정을 만들어보세요.")
    
    return feedback

def create_emotion_visualization(emotion_data, feedback):
    """감정 분석 결과를 시각화합니다."""
    emotion = emotion_data['emotion']
    confidence = emotion_data['confidence']
    score = feedback['overall_score']
    
    # 감정별 색상 매핑
    emotion_colors = {
        'happy': '#FFD700',      # 금색
        'confident': '#32CD32',  # 라임그린
        'neutral': '#87CEEB',    # 하늘색
        'surprised': '#FFA500',  # 오렌지
        'sad': '#4169E1',        # 로얄블루
        'angry': '#DC143C',      # 크림슨
        'unknown': '#808080'     # 회색
    }
    
    # 감정 점수 막대 그래프 생성
    emotions = ['happy', 'confident', 'neutral', 'surprised', 'sad', 'angry']
    scores = [emotion_data['details']['all_scores'].get(emotion, 0) for emotion in emotions]
    colors = [emotion_colors[emotion] for emotion in emotions]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 감정 점수 막대 그래프
    bars = ax1.bar(emotions, scores, color=colors, alpha=0.7)
    ax1.set_title('감정 분석 점수', fontsize=14, fontweight='bold')
    ax1.set_ylabel('점수')
    ax1.set_ylim(0, 1)
    
    # 막대 위에 점수 표시
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.2f}', ha='center', va='bottom')
    
    # 전체 점수 원형 그래프
    ax2.pie([score, 100-score], labels=['점수', '남은 점수'], 
            colors=[emotion_colors.get(emotion, '#808080'), '#E0E0E0'],
            autopct='%1.0f%%', startangle=90)
    ax2.set_title(f'면접 표정 점수: {score}/100', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('emotion_analysis_result.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 감정 분석 시각화가 'emotion_analysis_result.png'에 저장되었습니다.")

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()


def main():
    # STEP 1: Import the necessary modules.
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    # STEP 2: Create an FaceLandmarker object.
    base_options = python.BaseOptions(model_asset_path='./models/face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    image = mp.Image.create_from_file("face.jpeg")

    # STEP 4: Detect face landmarks from the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    
    # STEP 6: Analyze emotion from face landmarks
    print("\n" + "="*50)
    print("🎭 면접 표정 분석 결과")
    print("="*50)
    
    emotion_data = analyze_emotion_from_landmarks(detection_result.face_landmarks)
    
    print(f"📊 감정 분석:")
    print(f"   감정: {emotion_data['emotion']}")
    print(f"   신뢰도: {emotion_data['confidence']:.2f}")
    
    if emotion_data['details']:
        details = emotion_data['details']
        print(f"   눈 크기: {details['eye_width']:.4f}")
        print(f"   입 크기: {details['mouth_width']:.4f}")
        print(f"   눈썹 높이: {details['eyebrow_height']:.4f}")
    
    # STEP 7: Get interview feedback
    feedback = get_interview_feedback(emotion_data)
    
    print(f"\n📝 면접 피드백:")
    print(f"   점수: {feedback['overall_score']}/100")
    print(f"   평가: {feedback['feedback']}")
    print(f"   조언:")
    for i, suggestion in enumerate(feedback['suggestions'], 1):
        print(f"   {i}. {suggestion}")
    
    # STEP 8: Create emotion visualization
    try:
        create_emotion_visualization(emotion_data, feedback)
    except Exception as e:
        print(f"시각화 생성 중 오류: {e}")
    
    # STEP 9: Save the result image
    output_path = "face_landmarks_result.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    print(f"\n💾 얼굴 랜드마크 결과가 '{output_path}'에 저장되었습니다.")
    
    # Optional: Try to display if GUI is available
    try:
        cv2.imshow("Face Landmarks", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"GUI 표시 불가 (WSL2 환경): {e}")
        print("결과 이미지는 파일로 저장되었습니다.")
    
    print("="*50)

if __name__ == "__main__":
    main()