"""
실시간 오디오 캡처 모듈
sounddevice를 사용한 마이크 입력 처리 및 VAD 구현
"""

import sounddevice as sd
import numpy as np
import threading
import queue
import time
import logging
from typing import Optional, Callable, Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import deque
import scipy.signal

from config.audio_config import get_config

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """오디오 청크 데이터 구조"""
    data: np.ndarray
    timestamp: float
    sample_rate: int
    is_speech: bool = False
    volume_level: float = 0.0
    duration: float = 0.0


class VoiceActivityDetector:
    """음성 활동 감지 (VAD) 클래스"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.threshold = self.config.audio.VAD_THRESHOLD
        self.min_duration = self.config.audio.VAD_MIN_DURATION
        self.silence_duration = self.config.audio.VAD_SILENCE_DURATION
        
        # 상태 추적
        self.speech_start_time = None
        self.silence_start_time = None
        self.is_speaking = False
        
        # 에너지 기반 VAD를 위한 버퍼
        self.energy_buffer = deque(maxlen=10)
        self.energy_threshold = None
        
        # 적응형 임계값을 위한 통계
        self.noise_floor = 0.0
        self.adaptation_rate = 0.01
    
    def _calculate_energy(self, audio: np.ndarray) -> float:
        """오디오 에너지 계산"""
        return np.sqrt(np.mean(audio ** 2))
    
    def _calculate_zero_crossing_rate(self, audio: np.ndarray) -> float:
        """영교차율 계산 (음성/비음성 구분에 유용)"""
        signs = np.sign(audio)
        return np.mean(np.abs(np.diff(signs))) / 2.0
    
    def _update_noise_floor(self, energy: float):
        """노이즈 플로어 적응적 업데이트"""
        if not self.is_speaking:
            self.noise_floor = (1 - self.adaptation_rate) * self.noise_floor + \
                              self.adaptation_rate * energy
    
    def detect(self, audio: np.ndarray, timestamp: float) -> Tuple[bool, Dict[str, Any]]:
        """
        음성 활동 감지
        
        Args:
            audio: 오디오 데이터
            timestamp: 타임스탬프
            
        Returns:
            (is_speech, vad_info)
        """
        # 에너지 계산
        energy = self._calculate_energy(audio)
        zcr = self._calculate_zero_crossing_rate(audio)
        
        # 에너지 버퍼 업데이트
        self.energy_buffer.append(energy)
        
        # 적응형 임계값 설정
        if self.energy_threshold is None and len(self.energy_buffer) >= 5:
            self.energy_threshold = np.mean(list(self.energy_buffer)) * 2.0
        
        # 노이즈 플로어 업데이트
        self._update_noise_floor(energy)
        
        # 동적 임계값 계산
        dynamic_threshold = max(self.threshold, self.noise_floor * 3.0)
        
        # 음성 활동 판단 (에너지 + ZCR 조합)
        energy_condition = energy > dynamic_threshold
        zcr_condition = zcr > 0.01  # 음성은 일반적으로 높은 ZCR을 가짐
        
        current_speech = energy_condition and zcr_condition
        
        # 상태 머신 업데이트
        current_time = timestamp
        
        if current_speech and not self.is_speaking:
            # 음성 시작 감지
            if self.speech_start_time is None:
                self.speech_start_time = current_time
            elif current_time - self.speech_start_time >= self.min_duration:
                self.is_speaking = True
                self.silence_start_time = None
                
        elif not current_speech and self.is_speaking:
            # 침묵 시작 감지
            if self.silence_start_time is None:
                self.silence_start_time = current_time
            elif current_time - self.silence_start_time >= self.silence_duration:
                self.is_speaking = False
                self.speech_start_time = None
                
        elif current_speech and self.is_speaking:
            # 음성 지속 중
            self.silence_start_time = None
            
        elif not current_speech and not self.is_speaking:
            # 침묵 지속 중
            self.speech_start_time = None
        
        # VAD 정보 반환
        vad_info = {
            'energy': energy,
            'zcr': zcr,
            'threshold': dynamic_threshold,
            'noise_floor': self.noise_floor,
            'is_speaking': self.is_speaking,
            'speech_probability': min(1.0, energy / dynamic_threshold) if dynamic_threshold > 0 else 0.0
        }
        
        return self.is_speaking, vad_info


class AudioDeviceManager:
    """오디오 디바이스 관리 클래스"""
    
    @staticmethod
    def list_devices() -> List[Dict[str, Any]]:
        """사용 가능한 오디오 디바이스 목록 반환"""
        try:
            devices = sd.query_devices()
            device_list = []
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:  # 입력 가능한 디바이스만
                    device_list.append({
                        'id': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate'],
                        'hostapi': device['hostapi']
                    })
            
            return device_list
            
        except Exception as e:
            logger.error(f"디바이스 목록 조회 실패: {e}")
            return []
    
    @staticmethod
    def get_default_device() -> Optional[int]:
        """기본 입력 디바이스 ID 반환"""
        try:
            return sd.default.device[0]  # 입력 디바이스
        except Exception as e:
            logger.error(f"기본 디바이스 조회 실패: {e}")
            return None
    
    @staticmethod
    def test_device(device_id: int, duration: float = 1.0) -> bool:
        """디바이스 테스트"""
        try:
            config = get_config()
            test_audio = sd.rec(
                int(duration * config.audio.SAMPLE_RATE),
                samplerate=config.audio.SAMPLE_RATE,
                channels=config.audio.CHANNELS,
                device=device_id,
                dtype=config.audio.DTYPE
            )
            sd.wait()
            return len(test_audio) > 0 and np.any(test_audio != 0)
            
        except Exception as e:
            logger.error(f"디바이스 {device_id} 테스트 실패: {e}")
            return False


class RealTimeAudioCapture:
    """실시간 오디오 캡처 클래스"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        
        # 오디오 스트림 설정
        self.stream = None
        self.is_recording = False
        
        # 데이터 큐 및 콜백
        self.audio_queue = queue.Queue(maxsize=self.config.audio.MAX_QUEUE_SIZE)
        self.callbacks: List[Callable[[AudioChunk], None]] = []
        
        # VAD 및 전처리
        self.vad = VoiceActivityDetector(self.config)
        self.noise_reducer = NoiseReducer(self.config)
        
        # 버퍼 관리
        self.audio_buffer = deque(maxlen=self.config.audio.BUFFER_SIZE * 
                                 self.config.audio.SAMPLE_RATE)
        
        # 스레드 관리
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        # 통계 정보
        self.stats = {
            'total_chunks': 0,
            'speech_chunks': 0,
            'dropped_chunks': 0,
            'average_volume': 0.0,
            'last_speech_time': None
        }
    
    def add_callback(self, callback: Callable[[AudioChunk], None]):
        """오디오 청크 처리 콜백 추가"""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[AudioChunk], None]):
        """콜백 제거"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def _audio_callback(self, indata: np.ndarray, frames: int, 
                       time_info: Dict, status: sd.CallbackFlags):
        """오디오 스트림 콜백"""
        if status:
            logger.warning(f"오디오 콜백 상태: {status}")
        
        try:
            # 오디오 데이터 복사 (원본 데이터 보호)
            audio_data = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
            
            # 큐에 데이터 추가 (논블로킹)
            timestamp = time.time()
            
            if not self.audio_queue.full():
                self.audio_queue.put_nowait((audio_data, timestamp))
            else:
                self.stats['dropped_chunks'] += 1
                logger.warning("오디오 큐 가득참 - 청크 드롭됨")
                
        except Exception as e:
            logger.error(f"오디오 콜백 오류: {e}")
    
    def _process_audio_chunks(self):
        """오디오 청크 처리 스레드"""
        while not self.stop_event.is_set():
            try:
                # 큐에서 데이터 가져오기 (타임아웃 설정)
                audio_data, timestamp = self.audio_queue.get(timeout=0.1)
                
                # 노이즈 제거
                if self.config.audio.NOISE_REDUCTION:
                    audio_data = self.noise_reducer.reduce_noise(audio_data)
                
                # 볼륨 계산
                volume_level = np.sqrt(np.mean(audio_data ** 2))
                
                # VAD 수행
                is_speech, vad_info = self.vad.detect(audio_data, timestamp)
                
                # 오디오 청크 생성
                chunk = AudioChunk(
                    data=audio_data,
                    timestamp=timestamp,
                    sample_rate=self.config.audio.SAMPLE_RATE,
                    is_speech=is_speech,
                    volume_level=volume_level,
                    duration=len(audio_data) / self.config.audio.SAMPLE_RATE
                )
                
                # 통계 업데이트
                self.stats['total_chunks'] += 1
                if is_speech:
                    self.stats['speech_chunks'] += 1
                    self.stats['last_speech_time'] = timestamp
                
                self.stats['average_volume'] = (
                    0.9 * self.stats['average_volume'] + 0.1 * volume_level
                )
                
                # 버퍼에 추가
                self.audio_buffer.extend(audio_data)
                
                # 콜백 호출
                for callback in self.callbacks:
                    try:
                        callback(chunk)
                    except Exception as e:
                        logger.error(f"콜백 실행 오류: {e}")
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"오디오 처리 오류: {e}")
    
    def start_recording(self, device_id: Optional[int] = None) -> bool:
        """녹음 시작"""
        if self.is_recording:
            logger.warning("이미 녹음 중입니다")
            return False
        
        try:
            # 디바이스 설정
            if device_id is None:
                device_id = self.config.audio.DEFAULT_DEVICE
            
            # 오디오 스트림 파라미터
            audio_params = self.config.get_audio_params()
            audio_params['device'] = device_id
            
            # 스트림 생성
            self.stream = sd.InputStream(
                callback=self._audio_callback,
                **audio_params
            )
            
            # 스트림 시작
            self.stream.start()
            self.is_recording = True
            
            # 처리 스레드 시작
            self.stop_event.clear()
            self.processing_thread = threading.Thread(
                target=self._process_audio_chunks,
                daemon=True
            )
            self.processing_thread.start()
            
            logger.info(f"오디오 녹음 시작 - 디바이스: {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"녹음 시작 실패: {e}")
            self.is_recording = False
            return False
    
    def stop_recording(self):
        """녹음 중지"""
        if not self.is_recording:
            return
        
        try:
            self.is_recording = False
            
            # 스트림 중지
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            
            # 처리 스레드 중지
            self.stop_event.set()
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1.0)
            
            logger.info("오디오 녹음 중지")
            
        except Exception as e:
            logger.error(f"녹음 중지 오류: {e}")
    
    def get_recent_audio(self, duration: float) -> Optional[np.ndarray]:
        """최근 오디오 데이터 반환"""
        if not self.audio_buffer:
            return None
        
        sample_count = int(duration * self.config.audio.SAMPLE_RATE)
        sample_count = min(sample_count, len(self.audio_buffer))
        
        if sample_count <= 0:
            return None
        
        # 최근 데이터 추출
        recent_audio = np.array(list(self.audio_buffer)[-sample_count:])
        return recent_audio
    
    def get_stats(self) -> Dict[str, Any]:
        """녹음 통계 반환"""
        stats = self.stats.copy()
        stats['is_recording'] = self.is_recording
        stats['queue_size'] = self.audio_queue.qsize()
        stats['buffer_size'] = len(self.audio_buffer)
        return stats
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_recording()


class NoiseReducer:
    """노이즈 제거 클래스"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.noise_gate_threshold = self.config.audio.NOISE_GATE_THRESHOLD
        
        # 적응형 필터를 위한 버퍼
        self.noise_profile = None
        self.adaptation_frames = 0
        self.max_adaptation_frames = 50
    
    def reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """노이즈 제거 수행"""
        # 노이즈 게이트 적용
        audio = self._apply_noise_gate(audio)
        
        # 적응형 노이즈 제거 (선택사항)
        if self.config.audio.NOISE_REDUCTION:
            audio = self._adaptive_noise_reduction(audio)
        
        return audio
    
    def _apply_noise_gate(self, audio: np.ndarray) -> np.ndarray:
        """노이즈 게이트 적용"""
        # RMS 기반 게이팅
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms < self.noise_gate_threshold:
            return audio * 0.1  # 노이즈 레벨 감소
        
        return audio
    
    def _adaptive_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """적응형 노이즈 제거"""
        if self.noise_profile is None and self.adaptation_frames < self.max_adaptation_frames:
            # 초기 노이즈 프로파일 구축
            if self.noise_profile is None:
                self.noise_profile = np.abs(np.fft.fft(audio))
            else:
                current_spectrum = np.abs(np.fft.fft(audio))
                self.noise_profile = 0.9 * self.noise_profile + 0.1 * current_spectrum
            
            self.adaptation_frames += 1
            return audio
        
        if self.noise_profile is not None:
            # 스펙트럴 서브트랙션
            spectrum = np.fft.fft(audio)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)
            
            # 노이즈 제거 (간단한 스펙트럴 서브트랙션)
            clean_magnitude = magnitude - 0.5 * self.noise_profile
            clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)
            
            # 역변환
            clean_spectrum = clean_magnitude * np.exp(1j * phase)
            clean_audio = np.real(np.fft.ifft(clean_spectrum))
            
            return clean_audio
        
        return audio


# 편의 함수들
def create_audio_capture(config=None) -> RealTimeAudioCapture:
    """오디오 캡처 인스턴스 생성"""
    return RealTimeAudioCapture(config)

def list_audio_devices() -> List[Dict[str, Any]]:
    """오디오 디바이스 목록 반환"""
    return AudioDeviceManager.list_devices()

def test_audio_device(device_id: int) -> bool:
    """오디오 디바이스 테스트"""
    return AudioDeviceManager.test_device(device_id)
