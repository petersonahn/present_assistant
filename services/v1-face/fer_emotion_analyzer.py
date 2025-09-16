import cv2
from fer import FER

def analyze_emotion_with_fer(image):
    """
    FER 라이브러리를 이용해 입력 이미지에서 감정(표정)을 분석합니다.
    Args:
        image (np.ndarray): BGR(OpenCV) 이미지
    Returns:
        dominant_emotion (str): 가장 높은 확률의 감정
        emotions (dict): 감정별 확률 딕셔너리
    """
    detector = FER()
    results = detector.detect_emotions(image)
    if results:
        emotions = results[0]['emotions']
        dominant_emotion = max(emotions, key=emotions.get)
        return dominant_emotion, emotions
    else:
        return "unknown", {}

if __name__ == "__main__":
    # 예시: 이미지 파일에서 감정 분석
    img_path = "face.jpeg"  # 분석할 이미지 파일 경로
    img = cv2.imread(img_path)
    if img is None:
        print(f"이미지 파일을 찾을 수 없습니다: {img_path}")
    else:
        dominant_emotion, emotions = analyze_emotion_with_fer(img)
        print("주 감정:", dominant_emotion)
        print("감정 확률:", emotions)

    # 예시: 웹캠 실시간 감정 분석
    print("\n웹캠에서 실시간 감정 분석을 시작합니다. (q 키로 종료)")
    cap = cv2.VideoCapture(0)
    detector = FER()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = detector.detect_emotions(frame)
        if results:
            emotions = results[0]['emotions']
            dominant_emotion = max(emotions, key=emotions.get)
            cv2.putText(frame, f"{dominant_emotion}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print("감정:", dominant_emotion, emotions)
        cv2.imshow("FER Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
