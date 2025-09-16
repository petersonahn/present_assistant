"""
오디오 처리 설정 파일
실시간 음성 분석을 위한 모든 설정값들을 관리
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AudioConfig:
    """오디오 캡처 설정"""
    # 기본 오디오 설정
    SAMPLE_RATE: int = 16000  # 16kHz (Whisper 권장)
    CHANNELS: int = 1  # 모노
    DTYPE: str = 'float32'  # 32-bit float
    BLOCK_SIZE: int = 1024  # 블록 크기
    
    # 실시간 처리 설정
    CHUNK_DURATION: float = 1.0  # 1초 단위 청크
    CHUNK_SIZE: int = SAMPLE_RATE  # 1초 = 16000 샘플
    OVERLAP_SIZE: int = SAMPLE_RATE // 4  # 0.25초 오버랩
    
    # VAD (Voice Activity Detection) 설정
    VAD_THRESHOLD: float = 0.01  # 음성 활동 감지 임계값
    VAD_MIN_DURATION: float = 0.1  # 최소 음성 지속 시간 (초)
    VAD_SILENCE_DURATION: float = 0.5  # 침묵 감지 시간 (초)
    
    # 노이즈 제거 설정
    NOISE_REDUCTION: bool = True
    NOISE_GATE_THRESHOLD: float = 0.005  # 노이즈 게이트 임계값
    
    # 디바이스 설정
    DEFAULT_DEVICE: Optional[int] = None  # None = 기본 디바이스
    DEVICE_LATENCY: str = 'low'  # 'low', 'high'
    
    # 버퍼 설정
    BUFFER_SIZE: int = 5  # 최대 5초 버퍼
    MAX_QUEUE_SIZE: int = 10  # 최대 큐 크기


@dataclass 
class LibrosaConfig:
    """Librosa 음성 분석 설정"""
    # STFT 설정
    N_FFT: int = 2048  # FFT 윈도우 크기
    HOP_LENGTH: int = 512  # 홉 길이
    WIN_LENGTH: int = 2048  # 윈도우 길이
    WINDOW: str = 'hann'  # 윈도우 함수
    
    # 피치 분석 설정
    F_MIN: float = 50.0  # 최소 주파수 (Hz)
    F_MAX: float = 400.0  # 최대 주파수 (Hz) - 인간 음성 범위
    PITCH_THRESHOLD: float = 0.1  # 피치 감지 임계값
    
    # 스펙트로그램 설정
    N_MELS: int = 128  # 멜 필터 개수
    MEL_FMIN: float = 0.0  # 멜 최소 주파수
    MEL_FMAX: Optional[float] = None  # 멜 최대 주파수 (None = sr/2)
    
    # 템포 분석 설정
    TEMPO_MIN: float = 60.0  # 최소 BPM
    TEMPO_MAX: float = 200.0  # 최대 BPM
    
    # 음성 특징 추출 설정
    MFCC_N: int = 13  # MFCC 계수 개수
    CHROMA_N: int = 12  # 크로마 특징 개수
    SPECTRAL_CENTROID: bool = True  # 스펙트럴 중심 계산
    ZERO_CROSSING_RATE: bool = True  # 영교차율 계산


@dataclass
class WhisperConfig:
    """Whisper 음성 인식 설정"""
    # 모델 설정
    MODEL_NAME: str = "openai/whisper-small"  # Hugging Face 모델
    MODEL_SIZE: str = "small"  # tiny, base, small, medium, large
    LANGUAGE: str = "auto"  # 자동 감지 또는 'ko', 'en'
    
    # 처리 설정
    CHUNK_LENGTH_S: float = 30.0  # 최대 청크 길이 (초)
    STRIDE_LENGTH_S: float = 5.0  # 스트라이드 길이 (초)
    
    # 품질 설정
    RETURN_TIMESTAMPS: bool = True  # 타임스탬프 반환
    RETURN_LANGUAGE: bool = True  # 언어 정보 반환
    RETURN_CONFIDENCE: bool = True  # 신뢰도 점수 반환
    
    # GPU/CPU 설정
    DEVICE: str = "auto"  # "auto", "cpu", "cuda"
    TORCH_DTYPE: str = "float16"  # "float16", "float32"
    
    # 배치 처리 설정
    BATCH_SIZE: int = 1  # 실시간 처리를 위해 1
    
    # 후처리 설정
    NORMALIZE: bool = True  # 텍스트 정규화
    REMOVE_PUNCTUATION: bool = False  # 구두점 제거 여부


@dataclass
class EmotionConfig:
    """감정 분석 설정"""
    # 모델 설정
    MODEL_NAME: str = "monologg/koelectra-base-v3-discriminator"  # 한국어 감정 분석
    FALLBACK_MODEL: str = "cardiffnlp/twitter-roberta-base-emotion-multilingual-latest"
    
    # 감정 클래스
    EMOTION_LABELS: Dict[str, str] = None  # 초기화에서 설정
    
    # 처리 설정
    MAX_LENGTH: int = 512  # 최대 토큰 길이
    TRUNCATION: bool = True
    PADDING: bool = True
    
    # 신뢰도 설정
    MIN_CONFIDENCE: float = 0.5  # 최소 신뢰도
    
    # 배치 처리
    BATCH_SIZE: int = 4
    
    def __post_init__(self):
        if self.EMOTION_LABELS is None:
            self.EMOTION_LABELS = {
                "positive": "긍정적",
                "negative": "부정적", 
                "neutral": "중립적",
                "joy": "기쁨",
                "sadness": "슬픔",
                "anger": "분노",
                "fear": "두려움",
                "surprise": "놀람"
            }


@dataclass
class PerformanceConfig:
    """성능 최적화 설정"""
    # 멀티프로세싱
    USE_MULTIPROCESSING: bool = True
    MAX_WORKERS: int = 4
    
    # 메모리 관리
    MAX_MEMORY_MB: int = 1024  # 최대 메모리 사용량 (MB)
    GARBAGE_COLLECTION: bool = True  # 가비지 컬렉션 활성화
    
    # 캐싱
    ENABLE_CACHING: bool = True
    CACHE_SIZE: int = 100  # 캐시 항목 수
    CACHE_TTL: int = 300  # 캐시 TTL (초)
    
    # GPU 설정
    USE_GPU: bool = True  # GPU 사용 여부
    GPU_MEMORY_FRACTION: float = 0.7  # GPU 메모리 사용 비율
    
    # 실시간 처리 설정
    REAL_TIME_FACTOR: float = 1.0  # 실시간 배수 (1.0 = 실시간)
    MAX_LATENCY_MS: int = 100  # 최대 지연시간 (밀리초)


class SpeechConfig:
    """통합 음성 분석 설정 클래스"""
    
    def __init__(self):
        self.audio = AudioConfig()
        self.librosa = LibrosaConfig()
        self.whisper = WhisperConfig()
        self.emotion = EmotionConfig()
        self.performance = PerformanceConfig()
        
        # 환경 변수에서 설정 오버라이드
        self._load_from_env()
    
    def _load_from_env(self):
        """환경 변수에서 설정값 로드"""
        # 오디오 설정
        if os.getenv("AUDIO_SAMPLE_RATE"):
            self.audio.SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE"))
        
        if os.getenv("AUDIO_CHUNK_DURATION"):
            self.audio.CHUNK_DURATION = float(os.getenv("AUDIO_CHUNK_DURATION"))
            
        # Whisper 설정
        if os.getenv("WHISPER_MODEL"):
            self.whisper.MODEL_SIZE = os.getenv("WHISPER_MODEL")
            
        if os.getenv("WHISPER_LANGUAGE"):
            self.whisper.LANGUAGE = os.getenv("WHISPER_LANGUAGE")
            
        # GPU 설정
        if os.getenv("USE_GPU"):
            self.performance.USE_GPU = os.getenv("USE_GPU").lower() == "true"
            
        if os.getenv("DEVICE"):
            self.whisper.DEVICE = os.getenv("DEVICE")
    
    def get_audio_params(self) -> Dict[str, Any]:
        """오디오 캡처 파라미터 반환"""
        return {
            'samplerate': self.audio.SAMPLE_RATE,
            'channels': self.audio.CHANNELS,
            'dtype': self.audio.DTYPE,
            'blocksize': self.audio.BLOCK_SIZE,
            'latency': self.audio.DEVICE_LATENCY,
            'device': self.audio.DEFAULT_DEVICE
        }
    
    def get_librosa_params(self) -> Dict[str, Any]:
        """Librosa 분석 파라미터 반환"""
        return {
            'sr': self.audio.SAMPLE_RATE,
            'n_fft': self.librosa.N_FFT,
            'hop_length': self.librosa.HOP_LENGTH,
            'win_length': self.librosa.WIN_LENGTH,
            'window': self.librosa.WINDOW,
            'n_mels': self.librosa.N_MELS,
            'fmin': self.librosa.MEL_FMIN,
            'fmax': self.librosa.MEL_FMAX
        }
    
    def get_whisper_params(self) -> Dict[str, Any]:
        """Whisper 모델 파라미터 반환"""
        return {
            'model': self.whisper.MODEL_SIZE,
            'language': self.whisper.LANGUAGE,
            'return_timestamps': self.whisper.RETURN_TIMESTAMPS,
            'chunk_length_s': self.whisper.CHUNK_LENGTH_S,
            'stride_length_s': self.whisper.STRIDE_LENGTH_S
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """모든 설정을 딕셔너리로 변환"""
        return {
            'audio': self.audio.__dict__,
            'librosa': self.librosa.__dict__, 
            'whisper': self.whisper.__dict__,
            'emotion': self.emotion.__dict__,
            'performance': self.performance.__dict__
        }


# 전역 설정 인스턴스
config = SpeechConfig()

# 편의 함수들
def get_config() -> SpeechConfig:
    """설정 인스턴스 반환"""
    return config

def update_config(**kwargs):
    """설정 업데이트"""
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

def reset_config():
    """설정 초기화"""
    global config
    config = SpeechConfig()
