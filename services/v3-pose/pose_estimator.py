"""
Human Pose Estimation using OpenVINO
실시간 면접 피드백 시스템 - 포즈 감지 모듈
"""

import cv2
import numpy as np
from openvino.runtime import Core
from typing import List, Tuple, Dict, Optional
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HumanPoseEstimator:
    """OpenVINO 기반 인간 포즈 추정 클래스"""
    
    # COCO 포즈 키포인트 인덱스 (18개 관절)
    POSE_PAIRS = [
        (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
        (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
        (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)
    ]
    
    # 키포인트 이름 매핑
    KEYPOINT_NAMES = [
        "nose", "neck", "r_shoulder", "r_elbow", "r_wrist",
        "l_shoulder", "l_elbow", "l_wrist", "r_hip", "r_knee",
        "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye",
        "l_eye", "r_ear", "l_ear"
    ]
    
    def __init__(self, model_xml_path: str, model_bin_path: str, device: str = "CPU"):
        """
        포즈 추정기 초기화
        
        Args:
            model_xml_path: OpenVINO XML 모델 파일 경로
            model_bin_path: OpenVINO BIN 모델 파일 경로 
            device: 추론 디바이스 ("CPU", "GPU", "MYRIAD" 등)
        """
        self.device = device
        self.model_xml_path = model_xml_path
        self.model_bin_path = model_bin_path
        
        # OpenVINO 코어 및 모델 로드
        self.core = Core()
        self.model = self.core.read_model(model_xml_path)
        self.compiled_model = self.core.compile_model(self.model, device)
        
        # 입출력 정보 가져오기
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
        # 입력 크기 정보
        self.input_shape = self.input_layer.shape
        self.input_height = self.input_shape[2]  # 256
        self.input_width = self.input_shape[3]   # 456
        
        logger.info(f"모델 로드 완료: {model_xml_path}")
        logger.info(f"입력 크기: {self.input_width}x{self.input_height}")
        logger.info(f"디바이스: {device}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        이미지 전처리 (최적화됨)
        
        Args:
            image: 입력 이미지 (BGR 형식)
            
        Returns:
            전처리된 이미지 텐서
        """
        # 크기 조정을 먼저 수행 (작은 이미지에서 색상 변환이 더 빠름)
        resized_image = cv2.resize(image, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        
        # BGR을 RGB로 변환하면서 동시에 정규화
        # OpenCV는 BGR이므로 채널 순서를 바꿔서 정규화
        input_tensor = np.empty((1, 3, self.input_height, self.input_width), dtype=np.float32)
        
        # B, G, R 채널을 R, G, B 순서로 변환하면서 정규화 (메모리 효율적)
        input_tensor[0, 0] = resized_image[:, :, 2] / 255.0  # R
        input_tensor[0, 1] = resized_image[:, :, 1] / 255.0  # G  
        input_tensor[0, 2] = resized_image[:, :, 0] / 255.0  # B
        
        return input_tensor
    
    def postprocess_output(self, output: np.ndarray, original_shape: Tuple[int, int]) -> List[Dict]:
        """
        모델 출력 후처리 (최적화됨)
        
        Args:
            output: 모델 출력 텐서
            original_shape: 원본 이미지 크기 (height, width)
            
        Returns:
            감지된 포즈 키포인트 리스트
        """
        original_height, original_width = original_shape
        
        # 출력 형태: [1, 38, 32, 57] (PAFs + keypoints heatmaps)
        # 키포인트 히트맵 추출 (채널 19-37, 인덱스 19:38)
        keypoint_heatmaps = output[0, 19:38]  # 배치 차원 제거와 동시에 슬라이싱
        
        # 스케일 팩터 미리 계산
        heatmap_height, heatmap_width = keypoint_heatmaps.shape[1:]
        scale_x = original_width / heatmap_width
        scale_y = original_height / heatmap_height
        
        keypoints = []
        confidence_threshold = 0.05  # 임계값을 낮춰서 더 많은 키포인트 감지
        
        # 벡터화된 처리로 최적화
        for i in range(1, len(keypoint_heatmaps)):  # 배경(0번) 제외
            heatmap = keypoint_heatmaps[i]
            
            # argmax를 사용해서 최대값 위치 찾기 (더 빠름)
            max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            max_val = heatmap[max_idx]
            
            # 신뢰도 임계값 체크
            if max_val > confidence_threshold:
                # 좌표 변환 (정수 연산 최소화)
                x = int(max_idx[1] * scale_x)
                y = int(max_idx[0] * scale_y)
                
                keypoints.append({
                    'id': i - 1,  # 0-17 인덱스로 조정
                    'name': self.KEYPOINT_NAMES[i - 1],
                    'x': x,
                    'y': y,
                    'confidence': float(max_val)
                })
        
        return keypoints
    
    def estimate_pose(self, image: np.ndarray) -> Dict:
        """
        이미지에서 포즈 추정 수행
        
        Args:
            image: 입력 이미지 (BGR 형식)
            
        Returns:
            포즈 추정 결과 딕셔너리
        """
        original_height, original_width = image.shape[:2]
        
        # 전처리
        input_tensor = self.preprocess_image(image)
        
        # 추론 실행
        result = self.compiled_model([input_tensor])
        output = result[self.output_layer]
        
        # 후처리
        keypoints = self.postprocess_output(output, (original_height, original_width))
        
        # 포즈 분석
        pose_analysis = self.analyze_pose(keypoints)
        
        return {
            'keypoints': keypoints,
            'analysis': pose_analysis,
            'image_shape': (original_height, original_width)
        }
    
    def analyze_pose(self, keypoints: List[Dict]) -> Dict:
        """
        포즈 분석 및 피드백 생성
        
        Args:
            keypoints: 감지된 키포인트 리스트
            
        Returns:
            포즈 분석 결과
        """
        analysis = {
            'posture_score': 0,
            'shoulder_balance': 'unknown',
            'head_position': 'unknown',
            'arm_position': 'unknown',
            'feedback': []
        }
        
        # 키포인트 개수 확인
        if not keypoints or len(keypoints) == 0:
            analysis['feedback'].append('키포인트를 감지할 수 없습니다. 카메라에 전신이 보이도록 조정해주세요')
            return analysis
        
        # 키포인트를 이름으로 매핑 (신뢰도 0.2 이상만 사용 - 더 관대하게)
        kp_dict = {kp['name']: kp for kp in keypoints if kp.get('confidence', 0) > 0.2}
        
        logger.info(f"전체 키포인트 개수: {len(keypoints)}")
        logger.info(f"신뢰도 0.2 이상 키포인트: {list(kp_dict.keys())}")
        logger.info(f"키포인트 신뢰도: {[(kp['name'], round(kp['confidence'], 3)) for kp in keypoints[:5]]}")  # 상위 5개만
        
        try:
            # 기본 점수 (키포인트가 감지되면 20점)
            analysis['posture_score'] = 20
            
            # 어깨 균형 체크
            if 'l_shoulder' in kp_dict and 'r_shoulder' in kp_dict:
                left_shoulder = kp_dict['l_shoulder']
                right_shoulder = kp_dict['r_shoulder']
                
                shoulder_diff = abs(left_shoulder['y'] - right_shoulder['y'])
                
                if shoulder_diff < 30:  # 임계값을 30으로 증가 (더 관대하게)
                    analysis['shoulder_balance'] = 'balanced'
                    analysis['posture_score'] += 25
                    analysis['feedback'].append('어깨 위치가 균형잡혀 있어요 ✓')
                else:
                    analysis['shoulder_balance'] = 'unbalanced'
                    analysis['posture_score'] += 10  # 불균형이어도 일부 점수 부여
                    analysis['feedback'].append('어깨를 수평으로 맞춰보세요 ⚠')
            elif 'neck' in kp_dict:
                # 어깨가 없어도 목이 있으면 부분 점수
                analysis['shoulder_balance'] = 'partial'
                analysis['posture_score'] += 15
                analysis['feedback'].append('어깨 키포인트를 감지하지 못했습니다')
            
            # 목/머리 위치 체크 (nose 대신 더 많은 얼굴 키포인트 활용)
            head_detected = False
            if 'neck' in kp_dict:
                head_points = [kp for name, kp in kp_dict.items() if name in ['nose', 'l_eye', 'r_eye']]
                
                if head_points:
                    neck = kp_dict['neck']
                    # 가장 신뢰도 높은 얼굴 키포인트 사용
                    head_point = max(head_points, key=lambda x: x.get('confidence', 0))
                    
                    # 목과 얼굴의 수직 정렬 체크
                    head_tilt = abs(neck['x'] - head_point['x'])
                    
                    if head_tilt < 40:  # 임계값을 40으로 증가
                        analysis['head_position'] = 'straight'
                        analysis['posture_score'] += 25
                        analysis['feedback'].append('머리 위치가 바른 자세예요 ✓')
                    else:
                        analysis['head_position'] = 'tilted'
                        analysis['posture_score'] += 10
                        analysis['feedback'].append('머리를 곧게 세워보세요 ⚠')
                    head_detected = True
            
            if not head_detected:
                analysis['feedback'].append('머리 위치를 분석할 수 없습니다')
            
            # 팔 위치 체크 (더 관대한 조건)
            arm_positions = []
            arms_detected = 0
            
            for side in ['l', 'r']:
                shoulder_key = f'{side}_shoulder'
                elbow_key = f'{side}_elbow'
                wrist_key = f'{side}_wrist'
                
                if shoulder_key in kp_dict:
                    shoulder = kp_dict[shoulder_key]
                    arms_detected += 1
                    
                    if elbow_key in kp_dict:
                        elbow = kp_dict[elbow_key]
                        
                        # 팔꿈치가 어깨보다 많이 위에 있지 않은지 체크 (더 관대하게)
                        if elbow['y'] >= shoulder['y'] - 50:  # 50픽셀 여유
                            arm_positions.append('natural')
                        else:
                            arm_positions.append('raised')
                    else:
                        # 팔꿈치가 없어도 어깨가 있으면 부분적으로 자연스럽다고 가정
                        arm_positions.append('natural')
            
            if arms_detected > 0:
                if len(arm_positions) > 0:
                    natural_count = arm_positions.count('natural')
                    if natural_count == len(arm_positions):
                        analysis['arm_position'] = 'natural'
                        analysis['posture_score'] += 25
                        analysis['feedback'].append('팔 자세가 자연스러워요 ✓')
                    elif natural_count > 0:
                        analysis['arm_position'] = 'partial'
                        analysis['posture_score'] += 15
                        analysis['feedback'].append('일부 팔 자세가 자연스러워요 👌')
                    else:
                        analysis['arm_position'] = 'raised'
                        analysis['posture_score'] += 5
                        analysis['feedback'].append('팔을 자연스럽게 내려보세요 ⚠')
            else:
                analysis['feedback'].append('팔 키포인트를 감지하지 못했습니다')
            
            # 점수 상한 설정
            analysis['posture_score'] = min(100, analysis['posture_score'])
            
            # 전체적인 피드백
            if analysis['posture_score'] >= 70:
                analysis['feedback'].insert(0, '전반적으로 좋은 자세입니다! 👍')
            elif analysis['posture_score'] >= 40:
                analysis['feedback'].insert(0, '자세가 괜찮습니다. 조금만 더 개선해보세요 👌')
            elif analysis['posture_score'] >= 20:
                analysis['feedback'].insert(0, '키포인트가 감지되었습니다. 자세를 개선해보세요 📐')
            else:
                analysis['feedback'].insert(0, '카메라 위치를 조정하여 전신이 잘 보이도록 해주세요 📷')
                
        except Exception as e:
            logger.error(f"포즈 분석 중 오류: {e}")
            analysis['feedback'].append('포즈 분석 중 오류가 발생했습니다')
            analysis['posture_score'] = 0
        
        return analysis
    
    def draw_pose(self, image: np.ndarray, keypoints: List[Dict]) -> np.ndarray:
        """
        이미지에 포즈 키포인트와 스켈레톤 그리기
        
        Args:
            image: 원본 이미지
            keypoints: 키포인트 리스트
            
        Returns:
            포즈가 그려진 이미지
        """
        result_image = image.copy()
        
        # 키포인트를 인덱스로 매핑
        kp_dict = {kp['id']: kp for kp in keypoints}
        
        # 스켈레톤 연결선 그리기
        for pair in self.POSE_PAIRS:
            if pair[0] in kp_dict and pair[1] in kp_dict:
                point1 = kp_dict[pair[0]]
                point2 = kp_dict[pair[1]]
                
                cv2.line(result_image, 
                        (point1['x'], point1['y']), 
                        (point2['x'], point2['y']), 
                        (0, 255, 0), 2)
        
        # 키포인트 그리기
        for kp in keypoints:
            cv2.circle(result_image, (kp['x'], kp['y']), 5, (0, 0, 255), -1)
            cv2.putText(result_image, f"{kp['name']}", 
                       (kp['x'] + 5, kp['y'] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return result_image


def create_pose_estimator() -> HumanPoseEstimator:
    """포즈 추정기 인스턴스 생성"""
    model_xml = "pose/human-pose-estimation-0001.xml"
    model_bin = "pose/human-pose-estimation-0001.bin"
    
    return HumanPoseEstimator(model_xml, model_bin, device="CPU")


if __name__ == "__main__":
    # 테스트 코드
    estimator = create_pose_estimator()
    
    # 웹캠으로 테스트
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 포즈 추정
        result = estimator.estimate_pose(frame)
        
        # 결과 그리기
        pose_image = estimator.draw_pose(frame, result['keypoints'])
        
        # 피드백 표시
        y_offset = 30
        for feedback in result['analysis']['feedback']:
            cv2.putText(pose_image, feedback, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        cv2.imshow('Pose Estimation', pose_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
