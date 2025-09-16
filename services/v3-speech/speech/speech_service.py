"""
통합 음성 분석 서비스
모든 음성 처리 기능을 통합하여 관리하는 메인 서비스 클래스
"""

import asyncio
import threading
import time
import logging
import queue
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor
import gc

from .audio_capture import RealTimeAudioCapture, AudioChunk, create_audio_capture
from .audio_analyzer import RealTimeSpeechAnalyzer, SpeechFeatures, create_speech_analyzer
from .speech_recognizer import RealTimeSpeechRecognizer, TranscriptionResult, create_speech_recognizer
from .emotion_detector import RealTimeEmotionDetector, EmotionResult, create_emotion_detector
from config.audio_config import get_config

logger = logging.getLogger(__name__)


@dataclass
class SpeechAnalysisResult:
    """통합 음성 분석 결과"""
    timestamp: float
    
    # 오디오 정보
    audio_info: Dict[str, Any]
    
    # 음성 특징
    speech_features: Optional[SpeechFeatures] = None
    
    # 음성 인식 결과
    transcription: Optional[TranscriptionResult] = None
    
    # 감정 분석 결과
    emotion: Optional[EmotionResult] = None
    
    # 종합 분석
    overall_score: float = 0.0
    interview_feedback: Dict[str, Any] = None
    
    # 처리 성능
    processing_time: float = 0.0
    is_complete: bool = False


class SpeechAnalysisCallback:
    """음성 분석 콜백 인터페이스"""
    
    def on_audio_chunk(self, chunk: AudioChunk):
        """오디오 청크 수신 시 호출"""
        pass
    
    def on_speech_features(self, features: SpeechFeatures):
        """음성 특징 분석 완료 시 호출"""
        pass
    
    def on_transcription(self, transcription: TranscriptionResult):
        """음성 인식 완료 시 호출"""
        pass
    
    def on_emotion_analysis(self, emotion: EmotionResult):
        """감정 분석 완료 시 호출"""
        pass
    
    def on_complete_analysis(self, result: SpeechAnalysisResult):
        """전체 분석 완료 시 호출"""
        pass
    
    def on_error(self, error: Exception):
        """오류 발생 시 호출"""
        pass


class RealTimeSpeechService:
    """실시간 음성 분석 통합 서비스"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        
        # 서비스 컴포넌트들
        self.audio_capture = None
        self.speech_analyzer = None
        self.speech_recognizer = None
        self.emotion_detector = None
        
        # 상태 관리
        self.is_running = False
        self.is_initialized = False
        
        # 콜백 관리
        self.callbacks: List[SpeechAnalysisCallback] = []
        
        # 결과 큐 및 히스토리
        self.result_queue = queue.Queue(maxsize=50)
        self.analysis_history = deque(maxlen=200)
        
        # 스레드 풀
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.performance.MAX_WORKERS
        )
        
        # 동기화 객체
        self.stop_event = threading.Event()
        self.processing_lock = threading.Lock()
        
        # 통계 및 성능 모니터링
        self.stats = {
            'session_start_time': None,
            'total_chunks_processed': 0,
            'successful_transcriptions': 0,
            'successful_emotions': 0,
            'average_processing_time': 0.0,
            'current_session_duration': 0.0,
            'error_count': 0,
            'last_error_time': None
        }
        
        # 실시간 상태
        self.current_state = {
            'is_speaking': False,
            'current_volume': 0.0,
            'speech_rate': 0.0,
            'dominant_emotion': 'neutral',
            'confidence_level': 0.0,
            'overall_quality': 0.0
        }
        
        # 버퍼 관리
        self.audio_buffer_for_transcription = deque(maxlen=50)  # 음성 인식용
        self.transcription_buffer = deque(maxlen=20)  # 텍스트 버퍼
    
    async def initialize(self) -> bool:
        """서비스 초기화"""
        if self.is_initialized:
            logger.warning("이미 초기화된 서비스입니다")
            return True
        
        try:
            logger.info("음성 분석 서비스 초기화 시작...")
            
            # 각 컴포넌트 초기화
            initialization_tasks = []
            
            # 오디오 캡처 초기화
            self.audio_capture = create_audio_capture(self.config)
            initialization_tasks.append(self._initialize_audio_capture())
            
            # 음성 분석기 초기화
            self.speech_analyzer = create_speech_analyzer(self.config)
            initialization_tasks.append(self._initialize_speech_analyzer())
            
            # 음성 인식기 초기화
            self.speech_recognizer = create_speech_recognizer(self.config)
            initialization_tasks.append(self._initialize_speech_recognizer())
            
            # 감정 분석기 초기화
            self.emotion_detector = create_emotion_detector(self.config)
            initialization_tasks.append(self._initialize_emotion_detector())
            
            # 병렬 초기화
            results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
            
            # 초기화 결과 확인
            success_count = sum(1 for result in results if result is True)
            
            if success_count >= 2:  # 최소 2개 컴포넌트 성공
                self.is_initialized = True
                logger.info(f"음성 분석 서비스 초기화 완료 ({success_count}/4 컴포넌트 성공)")
                return True
            else:
                logger.error(f"음성 분석 서비스 초기화 실패 ({success_count}/4 컴포넌트 성공)")
                return False
                
        except Exception as e:
            logger.error(f"서비스 초기화 오류: {e}")
            return False
    
    async def _initialize_audio_capture(self) -> bool:
        """오디오 캡처 초기화"""
        try:
            # 콜백 등록
            self.audio_capture.add_callback(self._on_audio_chunk)
            logger.info("오디오 캡처 초기화 완료")
            return True
        except Exception as e:
            logger.error(f"오디오 캡처 초기화 실패: {e}")
            return False
    
    async def _initialize_speech_analyzer(self) -> bool:
        """음성 분석기 초기화"""
        try:
            # 음성 분석기는 별도 초기화 불필요
            logger.info("음성 분석기 초기화 완료")
            return True
        except Exception as e:
            logger.error(f"음성 분석기 초기화 실패: {e}")
            return False
    
    async def _initialize_speech_recognizer(self) -> bool:
        """음성 인식기 초기화"""
        try:
            success = self.speech_recognizer.start()
            if success:
                logger.info("음성 인식기 초기화 완료")
                return True
            else:
                logger.error("음성 인식기 초기화 실패")
                return False
        except Exception as e:
            logger.error(f"음성 인식기 초기화 실패: {e}")
            return False
    
    async def _initialize_emotion_detector(self) -> bool:
        """감정 분석기 초기화"""
        try:
            success = self.emotion_detector.initialize()
            if success:
                logger.info("감정 분석기 초기화 완료")
                return True
            else:
                logger.error("감정 분석기 초기화 실패")
                return False
        except Exception as e:
            logger.error(f"감정 분석기 초기화 실패: {e}")
            return False
    
    def add_callback(self, callback: SpeechAnalysisCallback):
        """콜백 추가"""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: SpeechAnalysisCallback):
        """콜백 제거"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def start_realtime_analysis(self, device_id: Optional[int] = None) -> bool:
        """실시간 분석 시작"""
        if not self.is_initialized:
            logger.error("서비스가 초기화되지 않았습니다")
            return False
        
        if self.is_running:
            logger.warning("이미 실시간 분석이 실행 중입니다")
            return False
        
        try:
            # 통계 초기화
            self.stats['session_start_time'] = time.time()
            self.stop_event.clear()
            
            # 실제 sounddevice 오디오 캡처 시작
            if self.audio_capture:
                logger.info("실제 sounddevice 오디오 캡처 시작")
                if not self.audio_capture.start_recording(device_id):
                    logger.error("오디오 캡처 시작 실패")
                    return False
            else:
                logger.error("오디오 캡처 모듈이 초기화되지 않았습니다")
                return False
            
            self.is_running = True
            logger.info("실시간 음성 분석 시작")
            
            # 콜백 알림
            for callback in self.callbacks:
                try:
                    callback.on_audio_chunk(None)  # 시작 알림
                except Exception as e:
                    logger.error(f"콜백 오류: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"실시간 분석 시작 오류: {e}")
            logger.error(f"에러 타입: {type(e)}")
            import traceback
            logger.error(f"전체 스택 트레이스: {traceback.format_exc()}")
            self.is_running = False
            raise e  # 에러를 다시 발생시켜 main.py에서 정확한 메시지 확인
    
    def stop_realtime_analysis(self):
        """실시간 분석 중지"""
        if not self.is_running:
            return
        
        try:
            self.is_running = False
            self.stop_event.set()
            
            # 오디오 캡처 중지 (Mock 모드에서는 스킵)
            if self.audio_capture and not hasattr(self, 'mock_thread'):
                try:
                    self.audio_capture.stop_recording()
                except Exception as e:
                    logger.warning(f"오디오 캡처 중지 실패 (Mock 모드에서는 정상): {e}")
            
            
            # 음성 인식기 중지
            if self.speech_recognizer:
                self.speech_recognizer.stop()
            
            # 스레드 풀 정리
            self.thread_pool.shutdown(wait=False)
            
            # 통계 업데이트
            if self.stats['session_start_time']:
                self.stats['current_session_duration'] = time.time() - self.stats['session_start_time']
            
            logger.info("실시간 음성 분석 중지")
            
        except Exception as e:
            logger.error(f"실시간 분석 중지 오류: {e}")
    
    def _on_audio_chunk(self, chunk: AudioChunk):
        """오디오 청크 처리 (메인 콜백)"""
        if not self.is_running or self.stop_event.is_set():
            return
        
        try:
            # 통계 업데이트
            self.stats['total_chunks_processed'] += 1
            
            # 현재 상태 업데이트
            self.current_state['is_speaking'] = chunk.is_speech
            self.current_state['current_volume'] = chunk.volume_level
            
            # 콜백 알림
            for callback in self.callbacks:
                try:
                    callback.on_audio_chunk(chunk)
                except Exception as e:
                    logger.error(f"오디오 청크 콜백 오류: {e}")
            
            # 음성 활동이 감지된 경우에만 처리
            if chunk.is_speech:
                # 비동기 처리를 위해 스레드 풀에 제출
                self.thread_pool.submit(self._process_audio_chunk, chunk)
            
        except Exception as e:
            logger.error(f"오디오 청크 처리 오류: {e}")
            self.stats['error_count'] += 1
            self.stats['last_error_time'] = time.time()
    
    def _process_audio_chunk(self, chunk: AudioChunk):
        """오디오 청크 종합 처리"""
        start_time = time.time()
        
        try:
            with self.processing_lock:
                # 분석 결과 초기화
                result = SpeechAnalysisResult(
                    timestamp=chunk.timestamp,
                    audio_info={
                        'duration': chunk.duration,
                        'volume_level': chunk.volume_level,
                        'is_speech': chunk.is_speech,
                        'sample_rate': chunk.sample_rate
                    }
                )
                
                # 1. 음성 특징 분석 (실시간)
                if self.speech_analyzer:
                    try:
                        features = self.speech_analyzer.analyze_audio_chunk(
                            chunk.data, chunk.timestamp
                        )
                        result.speech_features = features
                        
                        # 현재 상태 업데이트
                        self.current_state['speech_rate'] = features.speaking_rate
                        self.current_state['overall_quality'] = features.interview_score
                        
                        # 콜백 알림
                        for callback in self.callbacks:
                            try:
                                callback.on_speech_features(features)
                            except Exception as e:
                                logger.error(f"음성 특징 콜백 오류: {e}")
                                
                    except Exception as e:
                        logger.error(f"음성 특징 분석 오류: {e}")
                
                # 2. 음성 인식을 위한 버퍼 관리
                if self.speech_recognizer:
                    try:
                        # 음성 인식기에 오디오 추가
                        self.speech_recognizer.add_audio_chunk(
                            chunk.data, chunk.timestamp, chunk.sample_rate
                        )
                        
                        # 최신 인식 결과 확인
                        transcription = self.speech_recognizer.get_latest_result()
                        if transcription:
                            result.transcription = transcription
                            self.stats['successful_transcriptions'] += 1
                            
                            # 콜백 알림
                            for callback in self.callbacks:
                                try:
                                    callback.on_transcription(transcription)
                                except Exception as e:
                                    logger.error(f"음성 인식 콜백 오류: {e}")
                            
                            # 감정 분석을 위해 텍스트 버퍼에 추가
                            self.transcription_buffer.append(transcription)
                            
                    except Exception as e:
                        logger.error(f"음성 인식 처리 오류: {e}")
                
                # 3. 감정 분석 (텍스트가 있을 때)
                if self.emotion_detector and result.transcription:
                    try:
                        emotion_result = self.emotion_detector.analyze_text(
                            result.transcription.text,
                            result.transcription.language,
                            chunk.timestamp
                        )
                        result.emotion = emotion_result
                        self.stats['successful_emotions'] += 1
                        
                        # 현재 상태 업데이트
                        self.current_state['dominant_emotion'] = emotion_result.primary_emotion
                        self.current_state['confidence_level'] = emotion_result.confidence_level
                        
                        # 콜백 알림
                        for callback in self.callbacks:
                            try:
                                callback.on_emotion_analysis(emotion_result)
                            except Exception as e:
                                logger.error(f"감정 분석 콜백 오류: {e}")
                                
                    except Exception as e:
                        logger.error(f"감정 분석 처리 오류: {e}")
                
                # 4. 종합 분석 및 피드백 생성
                result = self._generate_comprehensive_analysis(result)
                
                # 처리 시간 기록
                result.processing_time = time.time() - start_time
                result.is_complete = True
                
                # 통계 업데이트
                self._update_processing_stats(result.processing_time)
                
                # 결과 저장
                self.analysis_history.append(result)
                
                # 결과 큐에 추가 (논블로킹)
                if not self.result_queue.full():
                    self.result_queue.put_nowait(result)
                
                # 완료 콜백 알림
                for callback in self.callbacks:
                    try:
                        callback.on_complete_analysis(result)
                    except Exception as e:
                        logger.error(f"완료 분석 콜백 오류: {e}")
                
        except Exception as e:
            logger.error(f"오디오 청크 종합 처리 오류: {e}")
            self.stats['error_count'] += 1
            
            # 오류 콜백 알림
            for callback in self.callbacks:
                try:
                    callback.on_error(e)
                except Exception as e2:
                    logger.error(f"오류 콜백 오류: {e2}")
    
    def _generate_comprehensive_analysis(self, result: SpeechAnalysisResult) -> SpeechAnalysisResult:
        """종합 분석 및 피드백 생성"""
        try:
            # 기본 점수 계산
            scores = []
            
            # 음성 특징 점수
            if result.speech_features:
                scores.append(result.speech_features.interview_score)
            
            # 감정 분석 점수
            if result.emotion:
                emotion_score = (
                    result.emotion.confidence_level * 40 +
                    (1.0 - result.emotion.nervousness_level) * 30 +
                    result.emotion.positivity_score * 30
                )
                scores.append(emotion_score)
            
            # 음성 인식 품질 점수
            if result.transcription:
                transcription_score = result.transcription.confidence * 100
                scores.append(transcription_score)
            
            # 종합 점수 계산
            if scores:
                result.overall_score = sum(scores) / len(scores)
            else:
                result.overall_score = 0.0
            
            # 면접 피드백 생성
            result.interview_feedback = self._generate_interview_feedback(result)
            
            return result
            
        except Exception as e:
            logger.error(f"종합 분석 생성 오류: {e}")
            result.overall_score = 0.0
            result.interview_feedback = {"error": "분석 생성 실패"}
            return result
    
    def _generate_interview_feedback(self, result: SpeechAnalysisResult) -> Dict[str, Any]:
        """면접 피드백 생성"""
        feedback = {
            'timestamp': result.timestamp,
            'overall_score': result.overall_score,
            'categories': {},
            'recommendations': [],
            'strengths': [],
            'areas_for_improvement': []
        }
        
        try:
            # 음성 품질 피드백
            if result.speech_features:
                speech_feedback = self._analyze_speech_quality(result.speech_features)
                feedback['categories']['speech_quality'] = speech_feedback
            
            # 감정 상태 피드백
            if result.emotion:
                emotion_feedback = self._analyze_emotional_state(result.emotion)
                feedback['categories']['emotional_state'] = emotion_feedback
            
            # 의사소통 피드백
            if result.transcription:
                communication_feedback = self._analyze_communication(result.transcription)
                feedback['categories']['communication'] = communication_feedback
            
            # 전체적인 추천사항 생성
            feedback['recommendations'] = self._generate_recommendations(result)
            
            return feedback
            
        except Exception as e:
            logger.error(f"면접 피드백 생성 오류: {e}")
            return feedback
    
    def _analyze_speech_quality(self, features: SpeechFeatures) -> Dict[str, Any]:
        """음성 품질 분석"""
        return {
            'score': features.interview_score,
            'speaking_rate': {
                'value': features.speaking_rate,
                'assessment': self._assess_speaking_rate(features.speaking_rate)
            },
            'clarity': {
                'value': features.clarity_score,
                'assessment': self._assess_clarity(features.clarity_score)
            },
            'volume': {
                'value': features.volume_level,
                'assessment': self._assess_volume(features.volume_level)
            }
        }
    
    def _analyze_emotional_state(self, emotion: EmotionResult) -> Dict[str, Any]:
        """감정 상태 분석"""
        return {
            'primary_emotion': emotion.primary_emotion,
            'confidence_level': emotion.confidence_level,
            'nervousness_level': emotion.nervousness_level,
            'positivity_score': emotion.positivity_score,
            'stress_indicator': emotion.stress_indicator,
            'assessment': self._assess_emotional_state(emotion)
        }
    
    def _analyze_communication(self, transcription: TranscriptionResult) -> Dict[str, Any]:
        """의사소통 분석"""
        return {
            'text_length': len(transcription.text),
            'confidence': transcription.confidence,
            'language': transcription.language,
            'word_count': len(transcription.text.split()) if transcription.text else 0,
            'assessment': self._assess_communication_quality(transcription)
        }
    
    def _assess_speaking_rate(self, rate: float) -> str:
        """말하기 속도 평가"""
        if 150 <= rate <= 180:
            return "적절함"
        elif rate < 120:
            return "너무 느림"
        elif rate > 200:
            return "너무 빠름"
        else:
            return "보통"
    
    def _assess_clarity(self, clarity: float) -> str:
        """명료도 평가"""
        if clarity > 0.8:
            return "매우 명료함"
        elif clarity > 0.6:
            return "명료함"
        elif clarity > 0.4:
            return "보통"
        else:
            return "개선 필요"
    
    def _assess_volume(self, volume: float) -> str:
        """음량 평가"""
        if 0.02 <= volume <= 0.08:
            return "적절함"
        elif volume < 0.02:
            return "너무 조용함"
        else:
            return "너무 큼"
    
    def _assess_emotional_state(self, emotion: EmotionResult) -> str:
        """감정 상태 평가"""
        if emotion.confidence_level > 0.7:
            return "자신감 있음"
        elif emotion.nervousness_level > 0.6:
            return "긴장 상태"
        elif emotion.stress_indicator > 0.5:
            return "스트레스 상태"
        else:
            return "안정적"
    
    def _assess_communication_quality(self, transcription: TranscriptionResult) -> str:
        """의사소통 품질 평가"""
        if transcription.confidence > 0.8:
            return "명확한 발음"
        elif transcription.confidence > 0.6:
            return "양호한 발음"
        else:
            return "발음 개선 필요"
    
    def _generate_recommendations(self, result: SpeechAnalysisResult) -> List[str]:
        """추천사항 생성"""
        recommendations = []
        
        # 음성 특징 기반 추천
        if result.speech_features:
            if result.speech_features.speaking_rate > 200:
                recommendations.append("말하기 속도를 조금 늦춰보세요")
            elif result.speech_features.speaking_rate < 120:
                recommendations.append("좀 더 빠르게 말해보세요")
            
            if result.speech_features.clarity_score < 0.5:
                recommendations.append("더 명확하게 발음해보세요")
            
            if result.speech_features.volume_level < 0.02:
                recommendations.append("목소리를 조금 더 크게 해보세요")
        
        # 감정 상태 기반 추천
        if result.emotion:
            if result.emotion.nervousness_level > 0.6:
                recommendations.append("긴장을 완화하기 위해 심호흡을 해보세요")
            
            if result.emotion.confidence_level < 0.4:
                recommendations.append("더 자신감 있게 말해보세요")
            
            if result.emotion.positivity_score < 0.3:
                recommendations.append("더 긍정적인 표현을 사용해보세요")
        
        if not recommendations:
            recommendations.append("좋은 상태를 유지하고 있습니다!")
        
        return recommendations
    
    def _update_processing_stats(self, processing_time: float):
        """처리 통계 업데이트"""
        total_processed = self.stats['total_chunks_processed']
        current_avg = self.stats['average_processing_time']
        
        # 이동 평균 계산
        self.stats['average_processing_time'] = (
            (current_avg * (total_processed - 1) + processing_time) / total_processed
        )
    
    def get_latest_result(self) -> Optional[SpeechAnalysisResult]:
        """최신 분석 결과 반환"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_all_pending_results(self) -> List[SpeechAnalysisResult]:
        """모든 대기 중인 결과 반환"""
        results = []
        while True:
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        return results
    
    def get_current_state(self) -> Dict[str, Any]:
        """현재 실시간 상태 반환"""
        state = self.current_state.copy()
        state['is_running'] = self.is_running
        state['session_duration'] = (
            time.time() - self.stats['session_start_time']
            if self.stats['session_start_time'] else 0.0
        )
        return state
    
    def get_session_summary(self, duration: float = 60.0) -> Dict[str, Any]:
        """세션 요약 반환"""
        if not self.analysis_history:
            return {}
        
        # 최근 duration 초간의 데이터 필터링
        current_time = time.time()
        recent_results = [
            result for result in self.analysis_history
            if current_time - result.timestamp <= duration
        ]
        
        if not recent_results:
            return {}
        
        # 통계 계산
        overall_scores = [r.overall_score for r in recent_results if r.overall_score > 0]
        speech_rates = []
        emotions = []
        
        for result in recent_results:
            if result.speech_features:
                speech_rates.append(result.speech_features.speaking_rate)
            if result.emotion:
                emotions.append(result.emotion.primary_emotion)
        
        summary = {
            'period': duration,
            'total_samples': len(recent_results),
            'average_overall_score': np.mean(overall_scores) if overall_scores else 0.0,
            'average_speech_rate': np.mean(speech_rates) if speech_rates else 0.0,
            'dominant_emotions': dict(Counter(emotions)) if emotions else {},
            'performance_metrics': {
                'average_processing_time': self.stats['average_processing_time'],
                'error_rate': self.stats['error_count'] / max(1, self.stats['total_chunks_processed'])
            }
        }
        
        return summary
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """종합 통계 반환"""
        stats = self.stats.copy()
        
        # 컴포넌트별 통계 추가
        if self.audio_capture:
            stats['audio_capture'] = self.audio_capture.get_stats()
        
        if self.speech_recognizer:
            stats['speech_recognizer'] = self.speech_recognizer.get_stats()
        
        if self.emotion_detector:
            stats['emotion_detector'] = self.emotion_detector.get_stats()
        
        # 현재 상태 추가
        stats['current_state'] = self.get_current_state()
        
        return stats
    
    def cleanup(self):
        """리소스 정리"""
        try:
            # 실시간 분석 중지
            self.stop_realtime_analysis()
            
            # 컴포넌트 정리
            if self.speech_recognizer:
                self.speech_recognizer.stop()
            
            # 메모리 정리
            self.analysis_history.clear()
            self.transcription_buffer.clear()
            
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    break
            
            # 가비지 컬렉션
            if self.config.performance.GARBAGE_COLLECTION:
                gc.collect()
            
            logger.info("음성 분석 서비스 정리 완료")
            
        except Exception as e:
            logger.error(f"서비스 정리 오류: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# 편의 함수
def create_speech_service(config=None) -> RealTimeSpeechService:
    """음성 분석 서비스 인스턴스 생성"""
    return RealTimeSpeechService(config)

async def test_speech_service(duration: float = 10.0) -> bool:
    """음성 분석 서비스 테스트"""
    try:
        service = create_speech_service()
        
        # 초기화
        if not await service.initialize():
            logger.error("서비스 초기화 실패")
            return False
        
        # 실시간 분석 시작
        if not service.start_realtime_analysis():
            logger.error("실시간 분석 시작 실패")
            return False
        
        logger.info(f"{duration}초 동안 테스트 실행...")
        
        # 테스트 실행
        start_time = time.time()
        while time.time() - start_time < duration:
            result = service.get_latest_result()
            if result:
                logger.info(f"분석 결과: 점수={result.overall_score:.1f}")
            
            await asyncio.sleep(0.5)
        
        # 정리
        service.stop_realtime_analysis()
        service.cleanup()
        
        logger.info("음성 분석 서비스 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"서비스 테스트 오류: {e}")
        return False
