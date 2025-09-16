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
        
        # 떨림 감지를 위한 이전 프레임 키포인트 저장
        self.prev_keypoints = {}
        self.keypoint_history = {}  # 최근 5프레임 저장
        self.tremor_threshold = 8.0  # 떨림 감지 임계값 (픽셀) - 더 민감하게 조정
        self.tremor_frame_count = 0  # 떨림 감지 프레임 카운터
        
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
        
        # 포즈 분석 (떨림 감지 포함)
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
            'head_tilt': 'unknown',  # 머리 위치 → 고개 기울임으로 명확화
            'arm_position': 'unknown',
            'tremor_detected': False,  # 떨림 감지 추가
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
                
                # 웹캠 각도를 고려한 어깨 균형 분석
                shoulder_diff = abs(left_shoulder['y'] - right_shoulder['y'])
                
                # 어깨 높이 차이와 함께 전체적인 자세도 고려
                shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
                shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])
                
                # 어깨 펴짐 정도 분석 (목과의 관계)
                if 'neck' in kp_dict:
                    neck = kp_dict['neck']
                    # 목이 어깨보다 앞으로 나와있는 정도 체크
                    neck_forward = neck['y'] - min(left_shoulder['y'], right_shoulder['y'])
                    
                    if neck_forward > 30:  # 목이 어깨보다 많이 앞으로 나온 경우
                        analysis['feedback'].append('어깨를 뒤로 펴고 가슴을 내밀어보세요 💪')
                        analysis['posture_score'] -= 5  # 구부정한 자세 감점
                    elif neck_forward < -10:  # 너무 뒤로 젖힌 경우
                        analysis['feedback'].append('자연스럽게 어깨 힘을 빼보세요 😊')
                    else:
                        analysis['feedback'].append('당당한 자세를 유지하고 계세요 👍')
                
                if shoulder_diff < 50:  # 웹캠 각도 고려하여 더욱 관대하게
                    analysis['shoulder_balance'] = 'balanced'
                    analysis['posture_score'] += 25
                    analysis['feedback'].append('어깨 균형이 좋습니다 ✓')
                elif shoulder_diff < 80:  # 약간의 차이는 허용
                    analysis['shoulder_balance'] = 'fair'
                    analysis['posture_score'] += 20
                    analysis['feedback'].append('어깨 균형이 양호합니다 👌')
                else:
                    analysis['shoulder_balance'] = 'unbalanced'
                    analysis['posture_score'] += 10
                    analysis['feedback'].append('어깨 높이를 맞춰보세요 ⚠')
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
            
            # 고개 기울임 분석 (좌우 기울임 감지)
            head_detected = False
            face_points = [name for name in kp_dict.keys() if name in ['l_eye', 'r_eye', 'l_ear', 'r_ear']]
            logger.info(f"감지된 얼굴 키포인트: {face_points}")
            
            # 양쪽 눈이 모두 감지된 경우 고개 기울임 분석
            if 'l_eye' in kp_dict and 'r_eye' in kp_dict:
                left_eye = kp_dict['l_eye']
                right_eye = kp_dict['r_eye']
                
                # 두 눈의 높이 차이로 고개 기울임 계산
                eye_height_diff = abs(left_eye['y'] - right_eye['y'])
                eye_distance = abs(left_eye['x'] - right_eye['x'])
                
                # 기울임 각도 계산 (각도가 클수록 고개가 많이 기울어짐)
                if eye_distance > 0:
                    tilt_ratio = eye_height_diff / eye_distance
                    
                    if tilt_ratio < 0.15:  # 약 8도 미만
                        analysis['head_tilt'] = 'straight'
                        analysis['posture_score'] += 25
                        analysis['feedback'].append('고개를 바르게 들고 계세요 ✓')
                    elif tilt_ratio < 0.3:  # 약 17도 미만
                        analysis['head_tilt'] = 'slightly_tilted'
                        analysis['posture_score'] += 15
                        analysis['feedback'].append('고개가 약간 기울어져 있어요 ⚠')
                    else:
                        analysis['head_tilt'] = 'tilted'
                        analysis['posture_score'] += 5
                        analysis['feedback'].append('고개를 바로 세워보세요 📐')
                    head_detected = True
                    
            # 한쪽 눈만 감지된 경우
            elif 'l_eye' in kp_dict or 'r_eye' in kp_dict:
                analysis['head_tilt'] = 'partial'
                analysis['posture_score'] += 10
                analysis['feedback'].append('정면을 향해 주세요')
                head_detected = True
                
            # 목만 감지된 경우
            elif 'neck' in kp_dict:
                analysis['head_tilt'] = 'neck_only'
                analysis['posture_score'] += 5
                analysis['feedback'].append('얼굴이 잘 보이도록 조정해주세요')
                head_detected = True
            
            if not head_detected:
                analysis['feedback'].append('고개 기울임을 분석할 수 없습니다')
            
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
            
            # 떨림 감지 분석
            tremor_detected = self.detect_tremor(keypoints)
            analysis['tremor_detected'] = tremor_detected
            
            if tremor_detected:
                analysis['posture_score'] -= 10  # 떨림 감지 시 감점
                analysis['feedback'].append('긴장을 풀고 자연스럽게 앉아보세요 🧘‍♀️')
            
            # 점수 상한 설정
            analysis['posture_score'] = min(100, max(0, analysis['posture_score']))  # 0-100 범위
            
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
    
    def detect_tremor(self, keypoints: List[Dict]) -> bool:
        """
        키포인트 위치 변화를 통한 떨림 감지 (개선된 버전)
        
        Args:
            keypoints: 현재 프레임의 키포인트 리스트
            
        Returns:
            떨림 감지 여부
        """
        # 떨림 감지 대상 키포인트 (손목을 주로, 팔꿈치와 어깨도 포함)
        tremor_points = ['l_wrist', 'r_wrist', 'l_elbow', 'r_elbow', 'l_shoulder', 'r_shoulder']
        
        current_kp = {kp['name']: (kp['x'], kp['y']) for kp in keypoints if kp['name'] in tremor_points}
        
        # 디버깅: 감지된 키포인트 출력 (매 10프레임마다만)
        if hasattr(self, '_tremor_debug_counter'):
            self._tremor_debug_counter += 1
        else:
            self._tremor_debug_counter = 1
            
        if self._tremor_debug_counter % 10 == 0:
            logger.info(f"떨림 감지 대상 키포인트: {list(current_kp.keys())}")
        
        # 첫 번째 프레임이거나 키포인트가 부족한 경우
        if len(current_kp) < 1 or not self.prev_keypoints:  # 조건 완화: 1개만 있어도 분석
            self.prev_keypoints = current_kp
            logger.info("첫 프레임 또는 키포인트 부족 - 떨림 감지 건너뜀")
            return False
        
        tremor_detected = False
        total_movement = 0
        movement_count = 0
        high_movement_count = 0  # 높은 움직임을 보이는 키포인트 수
        
        # 각 키포인트의 이동량 계산
        for name, (x, y) in current_kp.items():
            if name in self.prev_keypoints:
                prev_x, prev_y = self.prev_keypoints[name]
                movement = ((x - prev_x) ** 2 + (y - prev_y) ** 2) ** 0.5
                total_movement += movement
                movement_count += 1
                
                # 디버깅: 각 키포인트의 움직임 출력 (큰 움직임만)
                if movement > 5.0:  # 5픽셀 이상의 움직임만 로그
                    logger.info(f"{name} 움직임: {movement:.2f}px")
                
                # 즉시 떨림 감지 (단일 프레임에서 큰 움직임)
                if movement > self.tremor_threshold:
                    high_movement_count += 1
                    logger.info(f"⚠️ {name}에서 큰 움직임 감지: {movement:.2f}px")
                
                # 키포인트 히스토리 업데이트
                if name not in self.keypoint_history:
                    self.keypoint_history[name] = []
                
                self.keypoint_history[name].append((x, y))
                # 최근 3프레임만 유지 (더 빠른 반응)
                if len(self.keypoint_history[name]) > 3:
                    self.keypoint_history[name].pop(0)
                
                # 최근 프레임들의 변화량 분석 (조건 완화)
                if len(self.keypoint_history[name]) >= 2:  # 2프레임만 있어도 분석
                    recent_movements = []
                    for i in range(1, len(self.keypoint_history[name])):
                        prev_pos = self.keypoint_history[name][i-1]
                        curr_pos = self.keypoint_history[name][i]
                        move = ((curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2) ** 0.5
                        recent_movements.append(move)
                    
                    # 평균 이동량이 임계값을 초과하는 경우 (조건 완화)
                    avg_movement = sum(recent_movements) / len(recent_movements)
                    max_movement = max(recent_movements)
                    
                    # 더 관대한 조건: 평균이 임계값 초과하거나 최대값이 임계값*1.5 초과
                    if avg_movement > self.tremor_threshold * 0.7 or max_movement > self.tremor_threshold:
                        tremor_detected = True
                        logger.info(f"🔴 {name}에서 떨림 패턴 감지 - 평균: {avg_movement:.2f}, 최대: {max_movement:.2f}")
        
        # 전체 평균 이동량 분석 (조건 완화)
        if movement_count > 0:
            avg_total_movement = total_movement / movement_count
            
            # 큰 움직임이 있을 때만 로그 출력
            if avg_total_movement > 3.0:
                logger.info(f"전체 평균 움직임: {avg_total_movement:.2f}px")
            
            # 더 민감한 전체 떨림 감지
            if avg_total_movement > self.tremor_threshold * 0.8:
                tremor_detected = True
                logger.info(f"🔴 전체 떨림 감지: 평균 이동량 {avg_total_movement:.2f}px")
        
        # 다중 키포인트에서 동시에 큰 움직임이 있는 경우
        if high_movement_count >= 2:
            tremor_detected = True
            logger.info(f"🔴 다중 키포인트 떨림 감지: {high_movement_count}개 키포인트에서 큰 움직임")
        
        # 떨림 감지 결과 로그 (상태 변화시만)
        prev_tremor_state = getattr(self, '_prev_tremor_detected', False)
        if tremor_detected != prev_tremor_state:
            if tremor_detected:
                self.tremor_frame_count = 1
                logger.info(f"🚨 떨림 감지 시작!")
            else:
                logger.info(f"✅ 떨림 종료 - 안정 상태로 복귀")
                self.tremor_frame_count = 0
        elif tremor_detected:
            self.tremor_frame_count += 1
            # 연속 떨림 감지시 5프레임마다만 로그
            if self.tremor_frame_count % 5 == 0:
                logger.info(f"🚨 지속적인 떨림 (연속 {self.tremor_frame_count}프레임)")
        
        self._prev_tremor_detected = tremor_detected
        
        # 현재 키포인트를 다음 프레임을 위해 저장
        self.prev_keypoints = current_kp
        
        return tremor_detected
    
    def draw_pose(self, image: np.ndarray, keypoints: List[Dict]) -> np.ndarray:
        """
        면접 모드에서는 시각화 비활성화 - 원본 이미지만 반환
        
        Args:
            image: 원본 이미지
            keypoints: 키포인트 리스트 (사용하지 않음)
            
        Returns:
            원본 이미지 복사본
        """
        return image.copy()


def create_pose_estimator() -> HumanPoseEstimator:
    """포즈 추정기 인스턴스 생성"""
    model_xml = "pose/human-pose-estimation-0001.xml"
    model_bin = "pose/human-pose-estimation-0001.bin"
    
    return HumanPoseEstimator(model_xml, model_bin, device="CPU")


if __name__ == "__main__":
    # 테스트 코드 (시각화 완전 비활성화)
    estimator = create_pose_estimator()
    
    # 웹캠으로 테스트
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 포즈 추정
        result = estimator.estimate_pose(frame)
        
        # 면접 모드: 시각화 완전 비활성화 - 원본 프레임만 표시
        pose_image = frame.copy()
        
        # 콘솔에만 결과 출력 (시각적 방해 없이)
        print(f"키포인트: {len(result['keypoints'])}개, 점수: {result['analysis']['posture_score']}/100")
        
        cv2.imshow('Pose Estimation - Interview Mode', pose_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
