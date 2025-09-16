"""
음성 분석 모듈
librosa를 사용한 실시간 음성 특징 추출 및 분석
"""

import librosa
import numpy as np
import scipy.signal
import scipy.stats
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
from collections import deque
import time

from config.audio_config import get_config

logger = logging.getLogger(__name__)


@dataclass
class SpeechFeatures:
    """음성 특징 데이터 구조"""
    # 기본 정보
    timestamp: float
    duration: float
    sample_rate: int
    
    # 음성 품질 지표
    speaking_rate: float  # 말하기 속도 (WPM)
    volume_level: float  # 음량 레벨
    pitch_mean: float  # 평균 피치
    pitch_std: float  # 피치 변화량
    pitch_range: float  # 피치 범위
    
    # 음성 명료도
    clarity_score: float  # 명료도 점수
    articulation_rate: float  # 조음 속도
    pause_ratio: float  # 침묵 비율
    
    # 고급 특징
    spectral_centroid: float  # 스펙트럴 중심
    spectral_rolloff: float  # 스펙트럴 롤오프
    zero_crossing_rate: float  # 영교차율
    mfcc_features: List[float]  # MFCC 특징
    chroma_features: List[float]  # 크로마 특징
    
    # 감정 관련 특징
    energy: float  # 에너지
    formant_frequencies: List[float]  # 포먼트 주파수
    jitter: float  # 지터 (피치 변동)
    shimmer: float  # 시머 (진폭 변동)
    
    # 면접 적합성 점수
    interview_score: float  # 종합 점수 (0-100)
    confidence_level: float  # 자신감 레벨
    nervousness_indicator: float  # 긴장도 지표


class PitchAnalyzer:
    """피치 분석 클래스"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.sample_rate = self.config.audio.SAMPLE_RATE
        self.f_min = self.config.librosa.F_MIN
        self.f_max = self.config.librosa.F_MAX
        
        # 피치 추적을 위한 버퍼
        self.pitch_history = deque(maxlen=50)
    
    def extract_pitch(self, audio: np.ndarray) -> Dict[str, float]:
        """피치 특징 추출"""
        try:
            # Librosa를 사용한 피치 추출
            pitches, magnitudes = librosa.piptrack(
                y=audio,
                sr=self.sample_rate,
                fmin=self.f_min,
                fmax=self.f_max,
                threshold=self.config.librosa.PITCH_THRESHOLD
            )
            
            # 가장 강한 피치 선택
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if not pitch_values:
                return {
                    'pitch_mean': 0.0,
                    'pitch_std': 0.0,
                    'pitch_range': 0.0,
                    'jitter': 0.0
                }
            
            pitch_array = np.array(pitch_values)
            
            # 피치 통계 계산
            pitch_mean = np.mean(pitch_array)
            pitch_std = np.std(pitch_array)
            pitch_range = np.max(pitch_array) - np.min(pitch_array)
            
            # 지터 계산 (피치 변동성)
            jitter = self._calculate_jitter(pitch_array)
            
            # 히스토리 업데이트
            self.pitch_history.extend(pitch_values)
            
            return {
                'pitch_mean': float(pitch_mean),
                'pitch_std': float(pitch_std),
                'pitch_range': float(pitch_range),
                'jitter': float(jitter)
            }
            
        except Exception as e:
            logger.error(f"피치 추출 오류: {e}")
            return {
                'pitch_mean': 0.0,
                'pitch_std': 0.0,
                'pitch_range': 0.0,
                'jitter': 0.0
            }
    
    def _calculate_jitter(self, pitch_values: np.ndarray) -> float:
        """지터 계산 (연속 피치 값의 변동)"""
        if len(pitch_values) < 2:
            return 0.0
        
        # 연속 값들 간의 차이 계산
        diffs = np.abs(np.diff(pitch_values))
        mean_pitch = np.mean(pitch_values)
        
        if mean_pitch == 0:
            return 0.0
        
        # 정규화된 지터
        jitter = np.mean(diffs) / mean_pitch
        return float(jitter)


class SpectralAnalyzer:
    """스펙트럴 특징 분석 클래스"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.sample_rate = self.config.audio.SAMPLE_RATE
        self.n_fft = self.config.librosa.N_FFT
        self.hop_length = self.config.librosa.HOP_LENGTH
        self.n_mels = self.config.librosa.N_MELS
        self.n_mfcc = self.config.librosa.MFCC_N
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """스펙트럴 특징 추출"""
        try:
            # 스펙트로그램 계산
            stft = librosa.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.config.librosa.WIN_LENGTH,
                window=self.config.librosa.WINDOW
            )
            
            magnitude = np.abs(stft)
            
            # 스펙트럴 중심 (음성의 밝기 지표)
            spectral_centroid = librosa.feature.spectral_centroid(
                S=magnitude,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )[0]
            
            # 스펙트럴 롤오프 (고주파 성분 지표)
            spectral_rolloff = librosa.feature.spectral_rolloff(
                S=magnitude,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )[0]
            
            # 영교차율 (음성/무성음 구분)
            zcr = librosa.feature.zero_crossing_rate(
                audio,
                hop_length=self.hop_length
            )[0]
            
            # MFCC 특징 (음성 인식에 중요)
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # 크로마 특징 (음조 정보)
            chroma = librosa.feature.chroma_stft(
                S=magnitude,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # 멜 스펙트로그램
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            return {
                'spectral_centroid': float(np.mean(spectral_centroid)),
                'spectral_rolloff': float(np.mean(spectral_rolloff)),
                'zero_crossing_rate': float(np.mean(zcr)),
                'mfcc_features': [float(x) for x in np.mean(mfcc, axis=1)],
                'chroma_features': [float(x) for x in np.mean(chroma, axis=1)],
                'mel_spectrogram': mel_spec,
                'energy': float(np.sum(magnitude ** 2))
            }
            
        except Exception as e:
            logger.error(f"스펙트럴 특징 추출 오류: {e}")
            return {
                'spectral_centroid': 0.0,
                'spectral_rolloff': 0.0,
                'zero_crossing_rate': 0.0,
                'mfcc_features': [0.0] * self.n_mfcc,
                'chroma_features': [0.0] * 12,
                'energy': 0.0
            }


class FormantAnalyzer:
    """포먼트 분석 클래스"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.sample_rate = self.config.audio.SAMPLE_RATE
    
    def extract_formants(self, audio: np.ndarray, n_formants: int = 3) -> List[float]:
        """포먼트 주파수 추출"""
        try:
            # 선형 예측 계수 (LPC) 사용
            # 음성 길이에 따른 LPC 차수 결정
            lpc_order = min(int(2 + self.sample_rate / 1000), len(audio) - 1)
            
            if lpc_order < 2:
                return [0.0] * n_formants
            
            # LPC 계수 계산
            lpc_coeffs = librosa.lpc(audio, order=lpc_order)
            
            # 근을 구하여 포먼트 추출
            roots = np.roots(lpc_coeffs)
            
            # 복소수 근에서 주파수 계산
            angles = np.angle(roots)
            frequencies = angles * self.sample_rate / (2 * np.pi)
            
            # 양수 주파수만 선택하고 정렬
            positive_freqs = frequencies[frequencies > 0]
            positive_freqs = np.sort(positive_freqs)
            
            # 포먼트 주파수 범위 필터링 (일반적으로 200-3000Hz)
            formant_freqs = positive_freqs[
                (positive_freqs >= 200) & (positive_freqs <= 3000)
            ]
            
            # 요청된 개수만큼 반환
            result = list(formant_freqs[:n_formants])
            while len(result) < n_formants:
                result.append(0.0)
            
            return result[:n_formants]
            
        except Exception as e:
            logger.error(f"포먼트 추출 오류: {e}")
            return [0.0] * n_formants


class SpeechRateAnalyzer:
    """말하기 속도 분석 클래스"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.sample_rate = self.config.audio.SAMPLE_RATE
        
        # 음절 감지를 위한 파라미터
        self.min_syllable_duration = 0.1  # 최소 음절 길이 (초)
        self.syllable_threshold = 0.02  # 음절 감지 임계값
    
    def analyze_speech_rate(self, audio: np.ndarray, duration: float) -> Dict[str, float]:
        """말하기 속도 분석"""
        try:
            # 에너지 기반 음절 감지
            syllable_count = self._count_syllables(audio)
            
            # 침묵 구간 감지
            silence_ratio = self._calculate_silence_ratio(audio)
            
            # 말하기 속도 계산 (분당 음절 수)
            if duration > 0:
                syllables_per_minute = (syllable_count / duration) * 60
                # 한국어 기준: 1음절 ≈ 0.6단어
                words_per_minute = syllables_per_minute * 0.6
            else:
                syllables_per_minute = 0.0
                words_per_minute = 0.0
            
            # 조음 속도 (실제 말하는 시간만 고려)
            actual_speech_time = duration * (1 - silence_ratio)
            if actual_speech_time > 0:
                articulation_rate = (syllable_count / actual_speech_time) * 60
            else:
                articulation_rate = 0.0
            
            return {
                'speaking_rate': float(words_per_minute),
                'articulation_rate': float(articulation_rate),
                'syllable_count': int(syllable_count),
                'pause_ratio': float(silence_ratio)
            }
            
        except Exception as e:
            logger.error(f"말하기 속도 분석 오류: {e}")
            return {
                'speaking_rate': 0.0,
                'articulation_rate': 0.0,
                'syllable_count': 0,
                'pause_ratio': 0.0
            }
    
    def _count_syllables(self, audio: np.ndarray) -> int:
        """에너지 기반 음절 카운팅"""
        # 에너지 계산
        frame_length = int(0.025 * self.sample_rate)  # 25ms 프레임
        hop_length = int(0.01 * self.sample_rate)  # 10ms 홉
        
        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy.append(np.sum(frame ** 2))
        
        energy = np.array(energy)
        
        # 에너지 평활화
        energy = scipy.signal.medfilt(energy, kernel_size=5)
        
        # 피크 감지로 음절 카운팅
        peaks, _ = scipy.signal.find_peaks(
            energy,
            height=self.syllable_threshold * np.max(energy),
            distance=int(self.min_syllable_duration * 100)  # 10ms 단위
        )
        
        return len(peaks)
    
    def _calculate_silence_ratio(self, audio: np.ndarray) -> float:
        """침묵 비율 계산"""
        # 프레임 단위 에너지 계산
        frame_length = int(0.025 * self.sample_rate)
        hop_length = int(0.01 * self.sample_rate)
        
        silence_frames = 0
        total_frames = 0
        
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy = np.sum(frame ** 2)
            
            if energy < self.syllable_threshold:
                silence_frames += 1
            total_frames += 1
        
        if total_frames == 0:
            return 0.0
        
        return silence_frames / total_frames


class SpeechQualityAnalyzer:
    """음성 품질 분석 클래스"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.sample_rate = self.config.audio.SAMPLE_RATE
    
    def analyze_quality(self, audio: np.ndarray, spectral_features: Dict) -> Dict[str, float]:
        """음성 품질 분석"""
        try:
            # 명료도 점수 계산
            clarity_score = self._calculate_clarity(audio, spectral_features)
            
            # 시머 계산 (진폭 변동)
            shimmer = self._calculate_shimmer(audio)
            
            # 신호 대 잡음비 추정
            snr = self._estimate_snr(audio)
            
            # 하모닉-노이즈 비율
            hnr = self._calculate_hnr(audio)
            
            return {
                'clarity_score': float(clarity_score),
                'shimmer': float(shimmer),
                'snr': float(snr),
                'hnr': float(hnr)
            }
            
        except Exception as e:
            logger.error(f"음성 품질 분석 오류: {e}")
            return {
                'clarity_score': 0.0,
                'shimmer': 0.0,
                'snr': 0.0,
                'hnr': 0.0
            }
    
    def _calculate_clarity(self, audio: np.ndarray, spectral_features: Dict) -> float:
        """명료도 점수 계산"""
        # 스펙트럴 특징 기반 명료도 평가
        spectral_centroid = spectral_features.get('spectral_centroid', 0)
        zcr = spectral_features.get('zero_crossing_rate', 0)
        energy = spectral_features.get('energy', 0)
        
        # 정규화된 점수 계산 (0-1 범위)
        centroid_score = min(1.0, spectral_centroid / 2000.0)  # 2kHz 기준
        zcr_score = min(1.0, zcr * 100)  # ZCR 정규화
        energy_score = min(1.0, np.log10(energy + 1) / 10.0)  # 로그 스케일
        
        # 가중 평균
        clarity = 0.4 * centroid_score + 0.3 * zcr_score + 0.3 * energy_score
        return clarity
    
    def _calculate_shimmer(self, audio: np.ndarray) -> float:
        """시머 계산 (진폭 변동)"""
        # 프레임별 RMS 계산
        frame_length = int(0.025 * self.sample_rate)
        hop_length = int(0.01 * self.sample_rate)
        
        rms_values = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            rms = np.sqrt(np.mean(frame ** 2))
            if rms > 0:
                rms_values.append(rms)
        
        if len(rms_values) < 2:
            return 0.0
        
        rms_array = np.array(rms_values)
        
        # 연속 프레임 간 RMS 차이
        diffs = np.abs(np.diff(rms_array))
        mean_rms = np.mean(rms_array)
        
        if mean_rms == 0:
            return 0.0
        
        shimmer = np.mean(diffs) / mean_rms
        return shimmer
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """신호 대 잡음비 추정"""
        # 간단한 SNR 추정: 신호 에너지 vs 배경 노이즈
        signal_power = np.mean(audio ** 2)
        
        # 하위 10% 에너지를 노이즈로 간주
        sorted_power = np.sort(audio ** 2)
        noise_power = np.mean(sorted_power[:len(sorted_power) // 10])
        
        if noise_power == 0:
            return 100.0  # 매우 높은 SNR
        
        snr_db = 10 * np.log10(signal_power / noise_power)
        return max(0.0, snr_db)
    
    def _calculate_hnr(self, audio: np.ndarray) -> float:
        """하모닉-노이즈 비율 계산"""
        try:
            # 자기상관 기반 HNR 계산
            autocorr = np.correlate(audio, audio, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]
            
            # 피크 찾기
            if len(autocorr) < 2:
                return 0.0
            
            # 첫 번째 피크 (기본 주기)
            peak_idx = np.argmax(autocorr[1:]) + 1
            
            if peak_idx < len(autocorr):
                harmonic_power = autocorr[peak_idx]
                total_power = autocorr[0]
                
                if total_power > harmonic_power > 0:
                    hnr = 10 * np.log10(harmonic_power / (total_power - harmonic_power))
                    return max(0.0, hnr)
            
            return 0.0
            
        except Exception:
            return 0.0


class InterviewScoreCalculator:
    """면접 적합성 점수 계산 클래스"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        
        # 점수 가중치 설정
        self.weights = {
            'speaking_rate': 0.25,      # 말하기 속도
            'clarity': 0.20,            # 명료도
            'pitch_stability': 0.15,    # 피치 안정성
            'volume_consistency': 0.15, # 음량 일관성
            'pause_appropriateness': 0.10, # 적절한 휴지
            'confidence': 0.15          # 자신감
        }
    
    def calculate_interview_score(self, features: Dict[str, Any]) -> Dict[str, float]:
        """종합 면접 점수 계산"""
        try:
            # 각 항목별 점수 계산
            speaking_rate_score = self._score_speaking_rate(features.get('speaking_rate', 0))
            clarity_score = features.get('clarity_score', 0) * 100
            pitch_score = self._score_pitch_stability(
                features.get('pitch_std', 0),
                features.get('pitch_mean', 0)
            )
            volume_score = self._score_volume_consistency(features.get('volume_level', 0))
            pause_score = self._score_pause_appropriateness(features.get('pause_ratio', 0))
            confidence_score = self._calculate_confidence(features)
            
            # 가중 평균으로 종합 점수 계산
            total_score = (
                self.weights['speaking_rate'] * speaking_rate_score +
                self.weights['clarity'] * clarity_score +
                self.weights['pitch_stability'] * pitch_score +
                self.weights['volume_consistency'] * volume_score +
                self.weights['pause_appropriateness'] * pause_score +
                self.weights['confidence'] * confidence_score
            )
            
            # 긴장도 지표 계산
            nervousness = self._calculate_nervousness(features)
            
            return {
                'interview_score': float(min(100.0, max(0.0, total_score))),
                'confidence_level': float(confidence_score),
                'nervousness_indicator': float(nervousness),
                'sub_scores': {
                    'speaking_rate': float(speaking_rate_score),
                    'clarity': float(clarity_score),
                    'pitch_stability': float(pitch_score),
                    'volume_consistency': float(volume_score),
                    'pause_appropriateness': float(pause_score)
                }
            }
            
        except Exception as e:
            logger.error(f"면접 점수 계산 오류: {e}")
            return {
                'interview_score': 0.0,
                'confidence_level': 0.0,
                'nervousness_indicator': 0.0,
                'sub_scores': {}
            }
    
    def _score_speaking_rate(self, wpm: float) -> float:
        """말하기 속도 점수 (최적: 150-180 WPM)"""
        if 150 <= wpm <= 180:
            return 100.0
        elif 120 <= wpm < 150:
            return 80.0 + (wpm - 120) * 20 / 30
        elif 180 < wpm <= 220:
            return 80.0 + (220 - wpm) * 20 / 40
        elif 100 <= wpm < 120:
            return 60.0 + (wpm - 100) * 20 / 20
        elif 220 < wpm <= 260:
            return 60.0 + (260 - wpm) * 20 / 40
        else:
            return max(0.0, 60.0 - abs(wpm - 150) * 0.5)
    
    def _score_pitch_stability(self, pitch_std: float, pitch_mean: float) -> float:
        """피치 안정성 점수"""
        if pitch_mean == 0:
            return 0.0
        
        # 정규화된 피치 변동성
        cv = pitch_std / pitch_mean if pitch_mean > 0 else 0
        
        # 적절한 변동성: 0.1-0.3 (너무 단조롭지도, 변동이 심하지도 않게)
        if 0.1 <= cv <= 0.3:
            return 100.0
        elif cv < 0.1:
            return 70.0 + cv * 300  # 단조로움 페널티
        else:
            return max(0.0, 100.0 - (cv - 0.3) * 200)  # 과도한 변동 페널티
    
    def _score_volume_consistency(self, volume: float) -> float:
        """음량 일관성 점수"""
        # 적절한 음량 범위: 0.01-0.1
        if 0.01 <= volume <= 0.1:
            return 100.0
        elif volume < 0.01:
            return volume * 5000  # 너무 조용함
        else:
            return max(0.0, 100.0 - (volume - 0.1) * 500)  # 너무 큼
    
    def _score_pause_appropriateness(self, pause_ratio: float) -> float:
        """휴지 적절성 점수"""
        # 적절한 휴지 비율: 10-30%
        if 0.1 <= pause_ratio <= 0.3:
            return 100.0
        elif pause_ratio < 0.1:
            return 50.0 + pause_ratio * 500  # 휴지 부족
        else:
            return max(0.0, 100.0 - (pause_ratio - 0.3) * 200)  # 휴지 과다
    
    def _calculate_confidence(self, features: Dict[str, Any]) -> float:
        """자신감 레벨 계산"""
        # 여러 특징을 종합하여 자신감 평가
        volume = features.get('volume_level', 0)
        pitch_mean = features.get('pitch_mean', 0)
        speaking_rate = features.get('speaking_rate', 0)
        clarity = features.get('clarity_score', 0)
        
        # 자신감 지표들
        volume_confidence = min(100.0, volume * 1000)  # 적절한 음량
        pitch_confidence = min(100.0, pitch_mean / 2.0) if pitch_mean > 0 else 0  # 안정적인 피치
        rate_confidence = self._score_speaking_rate(speaking_rate)  # 적절한 속도
        clarity_confidence = clarity * 100  # 명료도
        
        # 가중 평균
        confidence = (
            0.3 * volume_confidence +
            0.2 * pitch_confidence +
            0.3 * rate_confidence +
            0.2 * clarity_confidence
        )
        
        return confidence
    
    def _calculate_nervousness(self, features: Dict[str, Any]) -> float:
        """긴장도 지표 계산"""
        # 긴장의 징후들
        jitter = features.get('jitter', 0)
        shimmer = features.get('shimmer', 0)
        pitch_std = features.get('pitch_std', 0)
        speaking_rate = features.get('speaking_rate', 0)
        
        # 긴장 지표들 (높을수록 긴장)
        jitter_nervousness = min(100.0, jitter * 1000)
        shimmer_nervousness = min(100.0, shimmer * 500)
        pitch_nervousness = min(100.0, pitch_std / 10.0)
        
        # 말하기 속도 기반 긴장도 (너무 빠르거나 느림)
        rate_nervousness = 0.0
        if speaking_rate > 200 or speaking_rate < 100:
            rate_nervousness = min(100.0, abs(speaking_rate - 150) / 2.0)
        
        # 평균 긴장도
        nervousness = (
            0.3 * jitter_nervousness +
            0.2 * shimmer_nervousness +
            0.3 * pitch_nervousness +
            0.2 * rate_nervousness
        )
        
        return nervousness


class RealTimeSpeechAnalyzer:
    """실시간 음성 분석 통합 클래스"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        
        # 분석기 인스턴스들
        self.pitch_analyzer = PitchAnalyzer(config)
        self.spectral_analyzer = SpectralAnalyzer(config)
        self.formant_analyzer = FormantAnalyzer(config)
        self.rate_analyzer = SpeechRateAnalyzer(config)
        self.quality_analyzer = SpeechQualityAnalyzer(config)
        self.score_calculator = InterviewScoreCalculator(config)
        
        # 분석 히스토리
        self.analysis_history = deque(maxlen=100)
    
    def analyze_audio_chunk(self, audio: np.ndarray, timestamp: float) -> SpeechFeatures:
        """오디오 청크 종합 분석"""
        try:
            duration = len(audio) / self.config.audio.SAMPLE_RATE
            
            # 각 분석기로 특징 추출
            pitch_features = self.pitch_analyzer.extract_pitch(audio)
            spectral_features = self.spectral_analyzer.extract_spectral_features(audio)
            formant_features = self.formant_analyzer.extract_formants(audio)
            rate_features = self.rate_analyzer.analyze_speech_rate(audio, duration)
            quality_features = self.quality_analyzer.analyze_quality(audio, spectral_features)
            
            # 모든 특징 통합
            all_features = {
                **pitch_features,
                **spectral_features,
                **rate_features,
                **quality_features,
                'volume_level': float(np.sqrt(np.mean(audio ** 2))),
                'formant_frequencies': formant_features
            }
            
            # 면접 점수 계산
            score_features = self.score_calculator.calculate_interview_score(all_features)
            
            # SpeechFeatures 객체 생성
            features = SpeechFeatures(
                timestamp=timestamp,
                duration=duration,
                sample_rate=self.config.audio.SAMPLE_RATE,
                speaking_rate=all_features.get('speaking_rate', 0.0),
                volume_level=all_features.get('volume_level', 0.0),
                pitch_mean=all_features.get('pitch_mean', 0.0),
                pitch_std=all_features.get('pitch_std', 0.0),
                pitch_range=all_features.get('pitch_range', 0.0),
                clarity_score=all_features.get('clarity_score', 0.0),
                articulation_rate=all_features.get('articulation_rate', 0.0),
                pause_ratio=all_features.get('pause_ratio', 0.0),
                spectral_centroid=all_features.get('spectral_centroid', 0.0),
                spectral_rolloff=all_features.get('spectral_rolloff', 0.0),
                zero_crossing_rate=all_features.get('zero_crossing_rate', 0.0),
                mfcc_features=all_features.get('mfcc_features', []),
                chroma_features=all_features.get('chroma_features', []),
                energy=all_features.get('energy', 0.0),
                formant_frequencies=formant_features,
                jitter=all_features.get('jitter', 0.0),
                shimmer=all_features.get('shimmer', 0.0),
                interview_score=score_features.get('interview_score', 0.0),
                confidence_level=score_features.get('confidence_level', 0.0),
                nervousness_indicator=score_features.get('nervousness_indicator', 0.0)
            )
            
            # 히스토리에 추가
            self.analysis_history.append(features)
            
            return features
            
        except Exception as e:
            logger.error(f"음성 분석 오류: {e}")
            # 기본값으로 SpeechFeatures 반환
            return SpeechFeatures(
                timestamp=timestamp,
                duration=len(audio) / self.config.audio.SAMPLE_RATE,
                sample_rate=self.config.audio.SAMPLE_RATE,
                speaking_rate=0.0,
                volume_level=0.0,
                pitch_mean=0.0,
                pitch_std=0.0,
                pitch_range=0.0,
                clarity_score=0.0,
                articulation_rate=0.0,
                pause_ratio=0.0,
                spectral_centroid=0.0,
                spectral_rolloff=0.0,
                zero_crossing_rate=0.0,
                mfcc_features=[],
                chroma_features=[],
                energy=0.0,
                formant_frequencies=[],
                jitter=0.0,
                shimmer=0.0,
                interview_score=0.0,
                confidence_level=0.0,
                nervousness_indicator=0.0
            )
    
    def get_analysis_summary(self, duration: float = 30.0) -> Dict[str, Any]:
        """최근 분석 결과 요약"""
        if not self.analysis_history:
            return {}
        
        # 최근 duration 초간의 데이터 필터링
        current_time = time.time()
        recent_analyses = [
            analysis for analysis in self.analysis_history
            if current_time - analysis.timestamp <= duration
        ]
        
        if not recent_analyses:
            return {}
        
        # 평균 통계 계산
        summary = {
            'avg_speaking_rate': np.mean([a.speaking_rate for a in recent_analyses]),
            'avg_volume_level': np.mean([a.volume_level for a in recent_analyses]),
            'avg_pitch_mean': np.mean([a.pitch_mean for a in recent_analyses]),
            'avg_clarity_score': np.mean([a.clarity_score for a in recent_analyses]),
            'avg_interview_score': np.mean([a.interview_score for a in recent_analyses]),
            'avg_confidence_level': np.mean([a.confidence_level for a in recent_analyses]),
            'avg_nervousness': np.mean([a.nervousness_indicator for a in recent_analyses]),
            'total_chunks': len(recent_analyses),
            'analysis_period': duration
        }
        
        return summary
    
    def to_dict(self, features: SpeechFeatures) -> Dict[str, Any]:
        """SpeechFeatures를 딕셔너리로 변환"""
        return asdict(features)


# 편의 함수
def create_speech_analyzer(config=None) -> RealTimeSpeechAnalyzer:
    """음성 분석기 인스턴스 생성"""
    return RealTimeSpeechAnalyzer(config)
