"""
포즈 감지 API 테스트 클라이언트
"""

import requests
import cv2
import base64
import json
from typing import Dict
import time

class PoseAPIClient:
    """포즈 감지 API 클라이언트"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self) -> Dict:
        """헬스체크"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def analyze_image_file(self, image_path: str) -> Dict:
        """이미지 파일로 포즈 분석"""
        with open(image_path, 'rb') as f:
            files = {'file': ('image.jpg', f, 'image/jpeg')}
            response = requests.post(f"{self.base_url}/pose/analyze", files=files)
        
        return response.json()
    
    def analyze_image_base64(self, image_path: str, include_result_image: bool = True) -> Dict:
        """Base64로 인코딩된 이미지로 포즈 분석"""
        # 이미지를 Base64로 인코딩
        with open(image_path, 'rb') as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        data = {
            "image": f"data:image/jpeg;base64,{image_base64}",
            "include_result_image": include_result_image
        }
        
        response = requests.post(f"{self.base_url}/pose/analyze_base64", json=data)
        return response.json()
    
    def analyze_webcam_frame(self, include_result_image: bool = False) -> Dict:
        """웹캠 프레임으로 포즈 분석"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            return {"error": "웹캠을 열 수 없습니다"}
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return {"error": "프레임을 읽을 수 없습니다"}
        
        # 프레임을 Base64로 인코딩
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        data = {
            "image": f"data:image/jpeg;base64,{image_base64}",
            "include_result_image": include_result_image
        }
        
        response = requests.post(f"{self.base_url}/pose/analyze_base64", json=data)
        return response.json()
    
    def get_keypoint_info(self) -> Dict:
        """키포인트 정보 가져오기"""
        response = requests.get(f"{self.base_url}/pose/keypoints")
        return response.json()


def test_api():
    """API 테스트 함수"""
    client = PoseAPIClient()
    
    print("🏥 헬스체크...")
    health = client.health_check()
    print(f"상태: {health}")
    
    if not health.get('model_loaded', False):
        print("❌ 모델이 로드되지 않았습니다. 서버를 확인하세요.")
        return
    
    print("\n📋 키포인트 정보...")
    keypoint_info = client.get_keypoint_info()
    print(f"키포인트 개수: {keypoint_info.get('total_keypoints', 0)}")
    print(f"키포인트 이름: {keypoint_info.get('keypoint_names', [])[:5]}...")  # 처음 5개만 출력
    
    print("\n📷 웹캠으로 포즈 분석 테스트...")
    try:
        start_time = time.time()
        result = client.analyze_webcam_frame(include_result_image=False)
        end_time = time.time()
        
        if result.get('success'):
            data = result['data']
            print(f"✅ 분석 성공! (소요시간: {end_time - start_time:.2f}초)")
            print(f"감지된 키포인트: {data['keypoint_count']}개")
            print(f"자세 점수: {data['analysis']['posture_score']}/100")
            print("피드백:")
            for feedback in data['analysis']['feedback']:
                print(f"  - {feedback}")
        else:
            print(f"❌ 분석 실패: {result}")
            
    except Exception as e:
        print(f"❌ 웹캠 테스트 실패: {e}")


def realtime_demo():
    """실시간 데모"""
    client = PoseAPIClient()
    
    print("실시간 포즈 분석 데모 시작... (ESC로 종료)")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            # 프레임을 Base64로 인코딩
            _, buffer = cv2.imencode('.jpg', frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            data = {
                "image": f"data:image/jpeg;base64,{image_base64}",
                "include_result_image": False
            }
            
            # API 호출
            response = requests.post(f"{client.base_url}/pose/analyze_base64", json=data)
            result = response.json()
            
            if result.get('success'):
                analysis = result['data']['analysis']
                keypoints = result['data']['keypoints']
                
                # 면접 모드: 시각화 비활성화 - 콘솔에만 결과 출력
                print(f"키포인트: {len(keypoints)}개, 점수: {analysis['posture_score']}/100")
            
        except Exception as e:
            print(f"분석 오류: {e}")
        
        cv2.imshow('Real-time Pose Analysis', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("포즈 감지 API 테스트 클라이언트")
    print("1. API 테스트")
    print("2. 실시간 데모")
    
    choice = input("선택하세요 (1 또는 2): ").strip()
    
    if choice == "1":
        test_api()
    elif choice == "2":
        realtime_demo()
    else:
        print("잘못된 선택입니다.")
