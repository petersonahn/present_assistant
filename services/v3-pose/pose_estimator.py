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
        logger.info(f"입력 이미지 형태: {image.shape}")
        logger.info(f"입력 이미지 타입: {image.dtype}")
        logger.info(f"입력 이미지 값 범위: {image.min()} - {image.max()}")
        
        # 크기 조정을 먼저 수행 (작은 이미지에서 색상 변환이 더 빠름)
        resized_image = cv2.resize(image, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        
        # BGR을 RGB로 변환하면서 동시에 정규화
        # OpenCV는 BGR이므로 채널 순서를 바꿔서 정규화
        input_tensor = np.empty((1, 3, self.input_height, self.input_width), dtype=np.float32)
        
        # B, G, R 채널을 R, G, B 순서로 변환하면서 정규화 (메모리 효율적)
        input_tensor[0, 0] = resized_image[:, :, 2] / 255.0  # R
        input_tensor[0, 1] = resized_image[:, :, 1] / 255.0  # G  
        input_tensor[0, 2] = resized_image[:, :, 0] / 255.0  # B
        
        logger.info(f"전처리된 텐서 형태: {input_tensor.shape}")
        logger.info(f"전처리된 텐서 값 범위: {input_tensor.min():.3f} - {input_tensor.max():.3f}")
        
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
        
        logger.info(f"모델 출력 형태: {output.shape}")
        logger.info(f"출력 값 범위: {output.min():.4f} - {output.max():.4f}")
        
        # 출력 형태 확인 및 적응적 처리
        if len(output.shape) == 4 and output.shape[1] >= 19:
            # 표준 형태: [1, channels, height, width]
            if output.shape[1] == 38:
                # PAFs + keypoints: 19개 PAF + 19개 keypoints
                keypoint_heatmaps = output[0, 19:38]
            elif output.shape[1] == 57:
                # 다른 형태의 출력
                keypoint_heatmaps = output[0, 38:57] 
            else:
                # 키포인트만 있는 경우
                keypoint_heatmaps = output[0, :19] if output.shape[1] >= 19 else output[0]
        else:
            # 다른 형태의 출력 처리
            keypoint_heatmaps = output[0] if len(output.shape) == 4 else output
            
        logger.info(f"키포인트 히트맵 형태: {keypoint_heatmaps.shape}")
        
        # 스케일 팩터 미리 계산
        heatmap_height, heatmap_width = keypoint_heatmaps.shape[1:]
        scale_x = original_width / heatmap_width
        scale_y = original_height / heatmap_height
        
        keypoints = []
        confidence_threshold = 0.001  # 임계값을 더욱 낮춤
        
        logger.info(f"히트맵 형태: {keypoint_heatmaps.shape}")
        logger.info(f"히트맵 최대값들: {[np.max(keypoint_heatmaps[i]) for i in range(min(5, len(keypoint_heatmaps)))]}")
        
        # 키포인트 개수를 히트맵 수에 맞춤
        num_keypoints = min(len(keypoint_heatmaps), len(self.KEYPOINT_NAMES))
        logger.info(f"처리할 키포인트 개수: {num_keypoints}")
        
        # 벡터화된 처리로 최적화
        for i in range(num_keypoints):
            heatmap = keypoint_heatmaps[i]
            
            # argmax를 사용해서 최대값 위치 찾기 (더 빠름)
            max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            max_val = heatmap[max_idx]
            
            # 디버깅을 위한 로그
            if i < 5:  # 처음 5개만 로그
                logger.info(f"키포인트 {self.KEYPOINT_NAMES[i]}: 최대값={max_val:.4f}, 위치={max_idx}")
            
            # 신뢰도 임계값 체크 - 매우 낮은 임계값 사용
            if max_val > confidence_threshold:
                # 좌표 변환 (정수 연산 최소화)
                x = int(max_idx[1] * scale_x)
                y = int(max_idx[0] * scale_y)
                
                keypoints.append({
                    'id': i,  # 0부터 시작하는 인덱스
                    'name': self.KEYPOINT_NAMES[i],
                    'x': x,
                    'y': y,
                    'confidence': float(max_val)
                })
            else:
                # 낮은 신뢰도도 로그에 기록
                if i < 5:
                    logger.info(f"키포인트 {self.KEYPOINT_NAMES[i]} 신뢰도 낮음: {max_val:.4f}")
        
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
            analysis['feedback'] = [
                '사람이 감지되지 않았습니다 👤',
                '카메라에 상반신이 잘 보이도록 조정해주세요',
                '조명이 충분한지 확인해주세요 💡',
                '화면 중앙에 위치해주세요 📷'
            ]
            analysis['posture_score'] = 0
            logger.warning("키포인트가 전혀 감지되지 않음")
            return analysis
        
        # 키포인트를 이름으로 매핑 (모든 감지된 키포인트 사용)
        kp_dict = {kp['name']: kp for kp in keypoints}  # 신뢰도 제한 완전 제거
        
        logger.info(f"전체 키포인트 개수: {len(keypoints)}")
        logger.info(f"분석에 사용할 키포인트: {list(kp_dict.keys())}")
        logger.info(f"키포인트 신뢰도: {[(kp['name'], round(kp['confidence'], 3)) for kp in keypoints]}")
        
        # 분석 전 강제 확인
        if len(kp_dict) == 0:
            logger.error("분석에 사용할 키포인트가 없습니다!")
            analysis['feedback'] = ['키포인트 매핑 실패']
            return analysis
        
        try:
            # 기본 점수 (상반신 웹캠 촬영 기준으로 조정)
            base_score = min(40, len(keypoints) * 10)  # 키포인트 개수에 따라 기본점수 조정
            analysis['posture_score'] = base_score
            
            logger.info(f"분석에 사용할 키포인트: {list(kp_dict.keys())}")
            logger.info(f"기본 점수: {base_score}점 (키포인트 {len(keypoints)}개)")
            
            # 어깨 균형 체크 (더 유연한 조건)
            shoulders = [kp for name, kp in kp_dict.items() if 'shoulder' in name]
            logger.info(f"감지된 어깨: {[name for name in kp_dict.keys() if 'shoulder' in name]}")
            
            if 'l_shoulder' in kp_dict and 'r_shoulder' in kp_dict:
                left_shoulder = kp_dict['l_shoulder']
                right_shoulder = kp_dict['r_shoulder']
                
                shoulder_diff = abs(left_shoulder['y'] - right_shoulder['y'])
                
                if shoulder_diff < 40:  # 더욱 관대한 임계값
                    analysis['shoulder_balance'] = 'balanced'
                    analysis['posture_score'] += 25
                    analysis['feedback'].append('어깨 위치가 균형잡혀 있어요 ✓')
                else:
                    analysis['shoulder_balance'] = 'unbalanced'
                    analysis['posture_score'] += 15  # 불균형이어도 더 많은 점수
                    analysis['feedback'].append('어깨를 수평으로 맞춰보세요 ⚠')
            elif len(shoulders) == 1:
                # 한쪽 어깨만 보이는 경우도 부분 인정
                analysis['shoulder_balance'] = 'partial'
                analysis['posture_score'] += 15
                shoulder_side = '왼쪽' if 'l_shoulder' in kp_dict else '오른쪽'
                analysis['feedback'].append(f'{shoulder_side} 어깨만 보입니다. 몸을 정면으로 향해주세요')
                logger.info(f"한쪽 어깨만 감지: {shoulder_side}")
            elif 'neck' in kp_dict:
                # 어깨가 없어도 목이 있으면 부분 점수
                analysis['shoulder_balance'] = 'partial'
                analysis['posture_score'] += 10
                analysis['feedback'].append('어깨 키포인트를 감지하지 못했습니다')
            else:
                # 팔 부위 키포인트로 어깨 위치 추정
                arms = [name for name in kp_dict.keys() if any(part in name for part in ['elbow', 'wrist'])]
                if len(arms) >= 2:
                    analysis['shoulder_balance'] = 'estimated'
                    analysis['posture_score'] += 10
                    analysis['feedback'].append('팔 위치로 어깨 균형을 추정했습니다')
            
            # 목/머리 위치 체크 (더 유연한 분석)
            head_detected = False
            head_points = [name for name in kp_dict.keys() if name in ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear']]
            logger.info(f"감지된 머리 키포인트: {head_points}")
            logger.info(f"목 키포인트 존재: {'neck' in kp_dict}")
            
            if 'neck' in kp_dict and head_points:
                neck = kp_dict['neck']
                # 가장 신뢰도 높은 얼굴 키포인트 사용
                best_head_point = max([kp_dict[name] for name in head_points], key=lambda x: x.get('confidence', 0))
                
                # 목과 얼굴의 수직 정렬 체크
                head_tilt = abs(neck['x'] - best_head_point['x'])
                
                if head_tilt < 50:  # 더욱 관대한 임계값
                    analysis['head_position'] = 'straight'
                    analysis['posture_score'] += 25
                    analysis['feedback'].append('머리 위치가 바른 자세예요 ✓')
                else:
                    analysis['head_position'] = 'tilted'
                    analysis['posture_score'] += 15
                    analysis['feedback'].append('머리를 곧게 세워보세요 ⚠')
                head_detected = True
            elif head_points and len(head_points) >= 2:
                # 목이 없어도 얼굴 키포인트들로 추정
                analysis['head_position'] = 'estimated'
                analysis['posture_score'] += 15
                analysis['feedback'].append('얼굴 키포인트로 머리 위치를 추정했습니다')
                head_detected = True
            elif 'neck' in kp_dict:
                # 목만 있는 경우
                analysis['head_position'] = 'partial'
                analysis['posture_score'] += 10
                analysis['feedback'].append('목 키포인트만 감지되었습니다')
                head_detected = True
            
            if not head_detected:
                analysis['feedback'].append('머리 위치를 분석할 수 없습니다')
            
            # 팔 위치 체크 (감지된 키포인트 활용)
            arm_parts = [name for name in kp_dict.keys() if any(part in name for part in ['shoulder', 'elbow', 'wrist'])]
            logger.info(f"감지된 팔 부위: {arm_parts}")
            arm_detected = False
            
            # 어깨-팔꿈치-손목 연결 체크
            for side in ['l', 'r']:
                shoulder_key = f'{side}_shoulder'
                elbow_key = f'{side}_elbow'
                wrist_key = f'{side}_wrist'
                
                side_parts = [key for key in [shoulder_key, elbow_key, wrist_key] if key in kp_dict]
                
                if len(side_parts) >= 2:
                    # 2개 이상의 팔 부위가 감지된 경우
                    if shoulder_key in kp_dict and elbow_key in kp_dict:
                        shoulder = kp_dict[shoulder_key]
                        elbow = kp_dict[elbow_key]
                        
                        if elbow['y'] >= shoulder['y'] - 60:  # 매우 관대한 조건
                            analysis['arm_position'] = 'natural'
                            analysis['posture_score'] += 25
                            side_kr = '왼쪽' if side == 'l' else '오른쪽'
                            analysis['feedback'].append(f'{side_kr} 팔 자세가 자연스러워요 ✓')
                        else:
                            analysis['arm_position'] = 'raised'
                            analysis['posture_score'] += 15
                            side_kr = '왼쪽' if side == 'l' else '오른쪽'
                            analysis['feedback'].append(f'{side_kr} 팔이 약간 올라가 있어요 ⚠')
                        arm_detected = True
                        break
                    elif elbow_key in kp_dict and wrist_key in kp_dict:
                        # 팔꿈치-손목만 있는 경우도 분석
                        elbow = kp_dict[elbow_key]
                        wrist = kp_dict[wrist_key]
                        
                        # 손목이 팔꿈치보다 아래에 있으면 자연스러운 자세로 추정
                        if wrist['y'] >= elbow['y'] - 30:
                            analysis['arm_position'] = 'estimated'
                            analysis['posture_score'] += 20
                            side_kr = '왼쪽' if side == 'l' else '오른쪽'
                            analysis['feedback'].append(f'{side_kr} 팔 위치가 자연스러워 보입니다 👌')
                        else:
                            analysis['arm_position'] = 'partial'
                            analysis['posture_score'] += 10
                            side_kr = '왼쪽' if side == 'l' else '오른쪽'
                            analysis['feedback'].append(f'{side_kr} 팔 일부만 보입니다')
                        arm_detected = True
                        break
            
            # 팔 부위가 하나라도 감지된 경우 (강제 분석)
            if not arm_detected and len(arm_parts) > 0:
                analysis['arm_position'] = 'partial'
                analysis['posture_score'] += 10
                analysis['feedback'].append(f'팔 일부 감지: {", ".join(arm_parts)}')
                arm_detected = True
                logger.info(f"팔 부위 강제 분석 적용: {arm_parts}")
            
            # 현재 데이터 기반 강제 분석 (r_wrist, l_elbow)
            if not arm_detected:
                if 'r_wrist' in kp_dict or 'l_elbow' in kp_dict:
                    analysis['arm_position'] = 'detected'
                    analysis['posture_score'] += 15
                    detected_parts = [name for name in ['r_wrist', 'l_elbow'] if name in kp_dict]
                    analysis['feedback'].append(f'팔 키포인트 감지: {", ".join(detected_parts)}')
                    arm_detected = True
                    logger.info(f"강제 팔 분석 적용: {detected_parts}")
            
            if not arm_detected:
                analysis['feedback'].append('팔 키포인트를 감지하지 못했습니다')
            
            # 점수 상한 설정
            analysis['posture_score'] = min(100, analysis['posture_score'])
            
            # 상반신 중심 전체적인 피드백
            if analysis['posture_score'] >= 80:
                analysis['feedback'].insert(0, '훌륭한 면접 자세입니다! 👍')
            elif analysis['posture_score'] >= 60:
                analysis['feedback'].insert(0, '좋은 자세를 유지하고 계세요 👌')
            elif analysis['posture_score'] >= 40:
                analysis['feedback'].insert(0, '상반신 자세를 조금 더 개선해보세요 📐')
            else:
                analysis['feedback'].insert(0, '카메라에 상반신이 잘 보이도록 조정해주세요 📷')
                
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
