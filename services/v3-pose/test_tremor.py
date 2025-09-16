"""
떨림 감지 기능 테스트 스크립트
"""

import cv2
import time
from pose_estimator import create_pose_estimator

def test_tremor_detection():
    """떨림 감지 테스트"""
    print("🧪 떨림 감지 테스트 시작...")
    print("📷 웹캠을 시작합니다. 손을 떨어보세요!")
    print("⌨️  ESC 키를 누르면 종료됩니다.")
    
    # 포즈 추정기 생성
    estimator = create_pose_estimator()
    
    # 웹캠 시작
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다.")
        return
    
    frame_count = 0
    tremor_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임을 읽을 수 없습니다.")
            break
        
        frame_count += 1
        
        try:
            # 포즈 추정 수행
            result = estimator.estimate_pose(frame)
            keypoints = result['keypoints']
            analysis = result['analysis']
            
            # 떨림 감지 결과
            tremor_detected = analysis['tremor_detected']
            if tremor_detected:
                tremor_count += 1
            
            # 화면에 결과 표시
            display_frame = frame.copy()
            
            # 상태 텍스트
            status_text = "🔴 떨림 감지됨!" if tremor_detected else "✅ 안정 상태"
            color = (0, 0, 255) if tremor_detected else (0, 255, 0)
            
            cv2.putText(display_frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # 통계 정보
            elapsed_time = time.time() - start_time
            tremor_rate = (tremor_count / frame_count * 100) if frame_count > 0 else 0
            
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Tremor Rate: {tremor_rate:.1f}%", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Keypoints: {len(keypoints)}", (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Score: {analysis['posture_score']}/100", (10, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 키포인트 정보 (디버깅용)
            tremor_points = ['l_wrist', 'r_wrist', 'l_elbow', 'r_elbow']
            detected_tremor_points = [kp['name'] for kp in keypoints if kp['name'] in tremor_points]
            
            if detected_tremor_points:
                points_text = f"Tremor Points: {', '.join(detected_tremor_points)}"
                cv2.putText(display_frame, points_text, (10, 190),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # 떨림 감지 시 특별 효과
            if tremor_detected:
                # 화면 테두리를 빨간색으로
                cv2.rectangle(display_frame, (0, 0), 
                             (display_frame.shape[1]-1, display_frame.shape[0]-1), 
                             (0, 0, 255), 5)
            
            cv2.imshow('Tremor Detection Test', display_frame)
            
            # 콘솔에도 결과 출력 (매 10프레임마다)
            if frame_count % 10 == 0:
                print(f"프레임 {frame_count}: {'떨림 감지' if tremor_detected else '안정'} | "
                      f"키포인트: {len(keypoints)}개 | 점수: {analysis['posture_score']}")
            
        except Exception as e:
            print(f"❌ 처리 오류: {e}")
        
        # ESC 키로 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # 결과 요약
    print(f"\n📊 테스트 결과:")
    print(f"총 프레임: {frame_count}")
    print(f"떨림 감지 프레임: {tremor_count}")
    print(f"떨림 감지 비율: {tremor_rate:.1f}%")
    print(f"테스트 시간: {elapsed_time:.1f}초")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_tremor_detection()
