"""
감정 분석 모듈
한국어 및 다국어 감정 분석을 위한 Hugging Face 모델 사용
"""

import torch
import numpy as np
import threading
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque, Counter
import time
import re
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline,
    TextClassificationPipeline
)
import warnings

from config.audio_config import get_config

logger = logging.getLogger(__name__)

# Transformers 경고 억제
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


@dataclass
class EmotionResult:
    """감정 분석 결과 데이터 구조"""
    text: str
    primary_emotion: str
    emotion_scores: Dict[str, float]
    confidence: float
    timestamp: float
    processing_time: float
    language: str = "ko"
    
    # 면접 관련 감정 지표
    nervousness_level: float = 0.0
    confidence_level: float = 0.0
    positivity_score: float = 0.0
    stress_indicator: float = 0.0


class KoreanEmotionAnalyzer:
    """한국어 감정 분석 클래스"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = self._get_device()
        self.model_loaded = False
        self.model_lock = threading.Lock()
        
        # 감정 매핑
        self.emotion_mapping = {
            # KoELECTRA 기반 모델의 감정 라벨들
            "LABEL_0": "neutral",    # 중립
            "LABEL_1": "positive",   # 긍정
            "LABEL_2": "negative",   # 부정
            "LABEL_3": "joy",        # 기쁨
            "LABEL_4": "sadness",    # 슬픔
            "LABEL_5": "anger",      # 분노
            "LABEL_6": "fear",       # 두려움
            "LABEL_7": "surprise",   # 놀람
            # 직접 매핑
            "neutral": "neutral",
            "positive": "positive",
            "negative": "negative",
            "joy": "joy",
            "sadness": "sadness", 
            "anger": "anger",
            "fear": "fear",
            "surprise": "surprise"
        }
        
        # 한국어 감정 표현 사전
        self.korean_emotion_keywords = {
            "positive": ["좋다", "행복", "기쁘다", "만족", "훌륭", "완벽", "최고", "감사"],
            "negative": ["나쁘다", "슬프다", "화나다", "실망", "짜증", "스트레스", "걱정", "불안"],
            "neutral": ["그냥", "보통", "평범", "일반적", "그저", "단순히"],
            "joy": ["웃음", "즐거움", "신나다", "재미있다", "흥미롭다", "활기"],
            "sadness": ["우울", "슬픔", "눈물", "아쉽다", "안타깝다", "서글프다"],
            "anger": ["화", "분노", "짜증", "열받다", "빡치다", "억울하다"],
            "fear": ["무섭다", "두렵다", "걱정", "불안", "떨리다", "긴장"],
            "surprise": ["놀랍다", "신기하다", "예상외", "뜻밖", "깜짝", "어머"]
        }
    
    def _get_device(self) -> str:
        """사용할 디바이스 결정"""
        if torch.cuda.is_available() and self.config.performance.USE_GPU:
            return "cuda"
        return "cpu"
    
    def load_model(self, force_reload: bool = False) -> bool:
        """한국어 감정 분석 모델 로드 (오프라인 우선)"""
        with self.model_lock:
            if self.model_loaded and not force_reload:
                return True
            
            try:
                model_name = self.config.emotion.MODEL_NAME
                logger.info(f"한국어 감정 분석 모델 로딩 중: {model_name}")
                
                # 1차 시도: 로컬 캐시에서 로드
                try:
                    # 토크나이저 로드 (로컬 캐시만)
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name, 
                        local_files_only=True
                    )
                    
                    # 모델 로드 (로컬 캐시만)
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        local_files_only=True
                    )
                    
                    # 디바이스로 이동
                    self.model = self.model.to(self.device)
                    
                    # 파이프라인 생성
                    self.pipeline = TextClassificationPipeline(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=0 if self.device == "cuda" else -1,
                        return_all_scores=True
                    )
                    
                    self.model_loaded = True
                    logger.info("로컬 캐시에서 한국어 감정 분석 모델 로딩 완료")
                    return True
                    
                except Exception as local_error:
                    logger.warning(f"로컬 캐시 로드 실패: {local_error}")
                    # 백업 모델 시도
                    return self._load_fallback_model()
                
            except Exception as e:
                logger.error(f"한국어 감정 분석 모델 로딩 실패: {e}")
                # Mock 모델로 대체
                return self._load_mock_model()
    
    def _load_fallback_model(self) -> bool:
        """백업 모델 로드 (오프라인 우선)"""
        try:
            fallback_model = self.config.emotion.FALLBACK_MODEL
            logger.info(f"백업 모델 로딩 시도: {fallback_model}")
            
            # 로컬 캐시에서 백업 모델 시도
            try:
                self.pipeline = pipeline(
                    "text-classification",
                    model=fallback_model,
                    device=0 if self.device == "cuda" else -1,
                    return_all_scores=True,
                    local_files_only=True
                )
                
                self.model_loaded = True
                logger.info("로컬 캐시에서 백업 감정 분석 모델 로딩 완료")
                return True
                
            except Exception as local_error:
                logger.warning(f"백업 모델 로컬 캐시 실패: {local_error}")
                # Mock 모델로 대체
                return self._load_mock_model()
            
        except Exception as e:
            logger.error(f"백업 모델 로딩 실패: {e}")
            return self._load_mock_model()
    
    def _load_mock_model(self) -> bool:
        """Mock 감정 분석 모델 로드"""
        try:
            class MockEmotionPipeline:
                def __call__(self, text, **kwargs):
                    """Mock 감정 분석 결과 반환"""
                    # 키워드 기반 간단한 감정 분석
                    text_lower = text.lower()
                    
                    if any(word in text_lower for word in ['좋', '기쁘', '행복', '만족', '훌륭']):
                        return [[
                            {'label': 'positive', 'score': 0.8},
                            {'label': 'neutral', 'score': 0.15},
                            {'label': 'negative', 'score': 0.05}
                        ]]
                    elif any(word in text_lower for word in ['나쁘', '슬프', '화', '짜증', '스트레스']):
                        return [[
                            {'label': 'negative', 'score': 0.7},
                            {'label': 'neutral', 'score': 0.2},
                            {'label': 'positive', 'score': 0.1}
                        ]]
                    else:
                        return [[
                            {'label': 'neutral', 'score': 0.6},
                            {'label': 'positive', 'score': 0.25},
                            {'label': 'negative', 'score': 0.15}
                        ]]
            
            self.pipeline = MockEmotionPipeline()
            self.model_loaded = True
            logger.info("Mock 감정 분석 모델 로딩 완료")
            return True
            
        except Exception as e:
            logger.error(f"Mock 모델 생성 실패: {e}")
            self.model_loaded = False
            return False
    
    def analyze_emotion(self, text: str, timestamp: float = None) -> EmotionResult:
        """텍스트 감정 분석"""
        if not self.model_loaded:
            return self._create_empty_result(text, timestamp)
        
        start_time = time.time()
        
        try:
            # 텍스트 전처리
            processed_text = self._preprocess_text(text)
            
            if not processed_text:
                return self._create_empty_result(text, timestamp)
            
            # 감정 분석 수행
            with self.model_lock:
                results = self.pipeline(processed_text)
            
            # 결과 처리
            emotion_scores = {}
            max_score = 0.0
            primary_emotion = "neutral"
            
            for result in results[0]:  # 첫 번째 결과 (단일 텍스트)
                label = result['label']
                score = result['score']
                
                # 감정 라벨 매핑
                emotion = self.emotion_mapping.get(label, label.lower())
                emotion_scores[emotion] = score
                
                if score > max_score:
                    max_score = score
                    primary_emotion = emotion
            
            # 키워드 기반 보정
            keyword_emotions = self._analyze_keywords(processed_text)
            emotion_scores = self._combine_scores(emotion_scores, keyword_emotions)
            
            # 최종 주요 감정 결정
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[primary_emotion]
            
            # 면접 관련 지표 계산
            interview_metrics = self._calculate_interview_metrics(
                processed_text, emotion_scores, primary_emotion
            )
            
            processing_time = time.time() - start_time
            
            return EmotionResult(
                text=text,
                primary_emotion=primary_emotion,
                emotion_scores=emotion_scores,
                confidence=confidence,
                timestamp=timestamp or time.time(),
                processing_time=processing_time,
                language="ko",
                **interview_metrics
            )
            
        except Exception as e:
            logger.error(f"감정 분석 오류: {e}")
            return self._create_empty_result(text, timestamp)
    
    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        if not text:
            return ""
        
        # 공백 정리
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 너무 짧은 텍스트 필터링
        if len(text) < 2:
            return ""
        
        # 최대 길이 제한
        max_length = self.config.emotion.MAX_LENGTH
        if len(text) > max_length:
            text = text[:max_length]
        
        return text
    
    def _analyze_keywords(self, text: str) -> Dict[str, float]:
        """키워드 기반 감정 분석"""
        keyword_scores = {emotion: 0.0 for emotion in self.korean_emotion_keywords}
        
        text_lower = text.lower()
        
        for emotion, keywords in self.korean_emotion_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1.0
            
            # 정규화
            if keywords:
                keyword_scores[emotion] = score / len(keywords)
        
        return keyword_scores
    
    def _combine_scores(self, model_scores: Dict[str, float], 
                       keyword_scores: Dict[str, float]) -> Dict[str, float]:
        """모델 점수와 키워드 점수 결합"""
        combined_scores = {}
        
        # 모든 감정에 대해 결합
        all_emotions = set(model_scores.keys()) | set(keyword_scores.keys())
        
        for emotion in all_emotions:
            model_score = model_scores.get(emotion, 0.0)
            keyword_score = keyword_scores.get(emotion, 0.0)
            
            # 가중 평균 (모델 70%, 키워드 30%)
            combined_score = 0.7 * model_score + 0.3 * keyword_score
            combined_scores[emotion] = combined_score
        
        # 정규화
        total_score = sum(combined_scores.values())
        if total_score > 0:
            combined_scores = {k: v / total_score for k, v in combined_scores.items()}
        
        return combined_scores
    
    def _calculate_interview_metrics(self, text: str, emotion_scores: Dict[str, float], 
                                   primary_emotion: str) -> Dict[str, float]:
        """면접 관련 감정 지표 계산"""
        # 긴장도 계산 (불안, 두려움, 부정 감정 기반)
        nervousness = (
            emotion_scores.get("fear", 0.0) * 0.4 +
            emotion_scores.get("negative", 0.0) * 0.3 +
            emotion_scores.get("sadness", 0.0) * 0.2 +
            emotion_scores.get("anger", 0.0) * 0.1
        )
        
        # 자신감 레벨 (긍정, 기쁨 감정 기반)
        confidence_level = (
            emotion_scores.get("positive", 0.0) * 0.5 +
            emotion_scores.get("joy", 0.0) * 0.3 +
            emotion_scores.get("neutral", 0.0) * 0.2
        )
        
        # 긍정성 점수
        positivity = (
            emotion_scores.get("positive", 0.0) * 0.6 +
            emotion_scores.get("joy", 0.0) * 0.4
        )
        
        # 스트레스 지표 (분노, 부정, 두려움 기반)
        stress = (
            emotion_scores.get("anger", 0.0) * 0.4 +
            emotion_scores.get("negative", 0.0) * 0.3 +
            emotion_scores.get("fear", 0.0) * 0.3
        )
        
        # 텍스트 길이 기반 보정
        text_length_factor = min(1.0, len(text) / 50.0)  # 50자 기준
        
        return {
            "nervousness_level": nervousness * text_length_factor,
            "confidence_level": confidence_level * text_length_factor,
            "positivity_score": positivity * text_length_factor,
            "stress_indicator": stress * text_length_factor
        }
    
    def _create_empty_result(self, text: str, timestamp: float) -> EmotionResult:
        """빈 결과 생성"""
        return EmotionResult(
            text=text,
            primary_emotion="neutral",
            emotion_scores={"neutral": 1.0},
            confidence=0.0,
            timestamp=timestamp or time.time(),
            processing_time=0.0,
            language="ko"
        )


class MultilingualEmotionAnalyzer:
    """다국어 감정 분석 클래스"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.pipeline = None
        self.device = self._get_device()
        self.model_loaded = False
        self.model_lock = threading.Lock()
    
    def _get_device(self) -> str:
        """사용할 디바이스 결정"""
        if torch.cuda.is_available() and self.config.performance.USE_GPU:
            return "cuda"
        return "cpu"
    
    def load_model(self) -> bool:
        """다국어 감정 분석 모델 로드 (오프라인 우선)"""
        with self.model_lock:
            if self.model_loaded:
                return True
            
            try:
                model_name = self.config.emotion.FALLBACK_MODEL
                logger.info(f"다국어 감정 분석 모델 로딩 중: {model_name}")
                
                # 로컬 캐시에서 모델 로드 시도
                try:
                    self.pipeline = pipeline(
                        "text-classification",
                        model=model_name,
                        device=0 if self.device == "cuda" else -1,
                        return_all_scores=True,
                        local_files_only=True
                    )
                    
                    self.model_loaded = True
                    logger.info("로컬 캐시에서 다국어 감정 분석 모델 로딩 완료")
                    return True
                    
                except Exception as local_error:
                    logger.warning(f"다국어 모델 로컬 캐시 실패: {local_error}")
                    # Mock 모델로 대체
                    return self._load_mock_model()
                
            except Exception as e:
                logger.error(f"다국어 감정 분석 모델 로딩 실패: {e}")
                return self._load_mock_model()
    
    def _load_mock_model(self) -> bool:
        """Mock 다국어 감정 분석 모델 로드"""
        try:
            class MockMultilingualPipeline:
                def __call__(self, text, **kwargs):
                    """Mock 다국어 감정 분석 결과 반환"""
                    # 간단한 키워드 기반 분석
                    text_lower = text.lower()
                    
                    # 영어 키워드
                    if any(word in text_lower for word in ['good', 'great', 'excellent', 'happy', 'love']):
                        return [[
                            {'label': 'joy', 'score': 0.75},
                            {'label': 'neutral', 'score': 0.2},
                            {'label': 'sadness', 'score': 0.05}
                        ]]
                    elif any(word in text_lower for word in ['bad', 'terrible', 'sad', 'angry', 'hate']):
                        return [[
                            {'label': 'sadness', 'score': 0.6},
                            {'label': 'anger', 'score': 0.25},
                            {'label': 'neutral', 'score': 0.15}
                        ]]
                    else:
                        return [[
                            {'label': 'neutral', 'score': 0.7},
                            {'label': 'joy', 'score': 0.2},
                            {'label': 'sadness', 'score': 0.1}
                        ]]
            
            self.pipeline = MockMultilingualPipeline()
            self.model_loaded = True
            logger.info("Mock 다국어 감정 분석 모델 로딩 완료")
            return True
            
        except Exception as e:
            logger.error(f"Mock 다국어 모델 생성 실패: {e}")
            self.model_loaded = False
            return False
    
    def analyze_emotion(self, text: str, language: str = "en", 
                       timestamp: float = None) -> EmotionResult:
        """다국어 텍스트 감정 분석"""
        if not self.model_loaded:
            return self._create_empty_result(text, language, timestamp)
        
        start_time = time.time()
        
        try:
            # 텍스트 전처리
            processed_text = self._preprocess_text(text)
            
            if not processed_text:
                return self._create_empty_result(text, language, timestamp)
            
            # 감정 분석 수행
            with self.model_lock:
                results = self.pipeline(processed_text)
            
            # 결과 처리
            emotion_scores = {}
            for result in results[0]:
                emotion = result['label'].lower()
                score = result['score']
                emotion_scores[emotion] = score
            
            # 주요 감정 결정
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[primary_emotion]
            
            processing_time = time.time() - start_time
            
            return EmotionResult(
                text=text,
                primary_emotion=primary_emotion,
                emotion_scores=emotion_scores,
                confidence=confidence,
                timestamp=timestamp or time.time(),
                processing_time=processing_time,
                language=language
            )
            
        except Exception as e:
            logger.error(f"다국어 감정 분석 오류: {e}")
            return self._create_empty_result(text, language, timestamp)
    
    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        if not text:
            return ""
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) < 2:
            return ""
        
        max_length = self.config.emotion.MAX_LENGTH
        if len(text) > max_length:
            text = text[:max_length]
        
        return text
    
    def _create_empty_result(self, text: str, language: str, timestamp: float) -> EmotionResult:
        """빈 결과 생성"""
        return EmotionResult(
            text=text,
            primary_emotion="neutral",
            emotion_scores={"neutral": 1.0},
            confidence=0.0,
            timestamp=timestamp or time.time(),
            processing_time=0.0,
            language=language
        )


class RealTimeEmotionDetector:
    """실시간 감정 분석 통합 클래스"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        
        # 분석기 인스턴스들
        self.korean_analyzer = KoreanEmotionAnalyzer(config)
        self.multilingual_analyzer = MultilingualEmotionAnalyzer(config)
        
        # 결과 히스토리
        self.emotion_history = deque(maxlen=100)
        self.recent_emotions = deque(maxlen=10)
        
        # 통계
        self.stats = {
            'total_analyzed': 0,
            'emotion_distribution': Counter(),
            'average_confidence': 0.0,
            'average_processing_time': 0.0,
            'language_distribution': Counter()
        }
        
        # 실시간 감정 상태
        self.current_emotional_state = {
            'dominant_emotion': 'neutral',
            'emotion_trend': 'stable',
            'emotional_intensity': 0.0,
            'mood_stability': 1.0
        }
    
    def initialize(self) -> bool:
        """분석기 초기화"""
        korean_success = self.korean_analyzer.load_model()
        multilingual_success = self.multilingual_analyzer.load_model()
        
        if korean_success:
            logger.info("한국어 감정 분석기 초기화 완료")
        
        if multilingual_success:
            logger.info("다국어 감정 분석기 초기화 완료")
        
        return korean_success or multilingual_success
    
    def analyze_text(self, text: str, language: str = "auto", 
                    timestamp: float = None) -> EmotionResult:
        """텍스트 감정 분석"""
        if not text or not text.strip():
            return self._create_empty_result(text, language, timestamp)
        
        # 언어 자동 감지
        if language == "auto":
            language = self._detect_language(text)
        
        # 적절한 분석기 선택
        if language == "ko":
            result = self.korean_analyzer.analyze_emotion(text, timestamp)
        else:
            result = self.multilingual_analyzer.analyze_emotion(text, language, timestamp)
        
        # 결과 후처리
        result = self._post_process_result(result)
        
        # 히스토리 업데이트
        self._update_history(result)
        
        # 통계 업데이트
        self._update_stats(result)
        
        # 실시간 감정 상태 업데이트
        self._update_emotional_state(result)
        
        return result
    
    def _detect_language(self, text: str) -> str:
        """간단한 언어 감지"""
        # 한글 문자 비율로 언어 판단
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'\s', '', text))
        
        if total_chars == 0:
            return "ko"  # 기본값
        
        korean_ratio = korean_chars / total_chars
        
        if korean_ratio > 0.3:  # 30% 이상 한글이면 한국어로 판단
            return "ko"
        else:
            return "en"
    
    def _post_process_result(self, result: EmotionResult) -> EmotionResult:
        """결과 후처리 (스무딩, 보정 등)"""
        # 최근 감정과의 연속성 고려
        if len(self.recent_emotions) > 0:
            recent_emotions = [e.primary_emotion for e in self.recent_emotions]
            emotion_counter = Counter(recent_emotions)
            
            # 급격한 감정 변화 완화
            if len(emotion_counter) > 1:
                most_common = emotion_counter.most_common(1)[0][0]
                if result.primary_emotion != most_common:
                    # 신뢰도가 낮으면 이전 감정 유지
                    if result.confidence < 0.7:
                        result.primary_emotion = most_common
                        result.confidence = 0.6  # 보정된 신뢰도
        
        return result
    
    def _update_history(self, result: EmotionResult):
        """히스토리 업데이트"""
        self.emotion_history.append(result)
        self.recent_emotions.append(result)
    
    def _update_stats(self, result: EmotionResult):
        """통계 업데이트"""
        self.stats['total_analyzed'] += 1
        self.stats['emotion_distribution'][result.primary_emotion] += 1
        self.stats['language_distribution'][result.language] += 1
        
        # 평균 신뢰도 업데이트
        total = self.stats['total_analyzed']
        self.stats['average_confidence'] = (
            (self.stats['average_confidence'] * (total - 1) + result.confidence) / total
        )
        
        # 평균 처리 시간 업데이트
        self.stats['average_processing_time'] = (
            (self.stats['average_processing_time'] * (total - 1) + result.processing_time) / total
        )
    
    def _update_emotional_state(self, result: EmotionResult):
        """실시간 감정 상태 업데이트"""
        # 지배적 감정 업데이트
        self.current_emotional_state['dominant_emotion'] = result.primary_emotion
        
        # 감정 강도 계산
        self.current_emotional_state['emotional_intensity'] = result.confidence
        
        # 감정 트렌드 분석
        if len(self.recent_emotions) >= 3:
            recent_emotions = [e.primary_emotion for e in self.recent_emotions[-3:]]
            
            if len(set(recent_emotions)) == 1:
                trend = 'stable'
            elif recent_emotions[-1] != recent_emotions[0]:
                trend = 'changing'
            else:
                trend = 'fluctuating'
            
            self.current_emotional_state['emotion_trend'] = trend
        
        # 기분 안정성 계산
        if len(self.recent_emotions) >= 5:
            recent_confidences = [e.confidence for e in self.recent_emotions[-5:]]
            stability = 1.0 - np.std(recent_confidences)
            self.current_emotional_state['mood_stability'] = max(0.0, stability)
    
    def get_emotion_summary(self, duration: float = 30.0) -> Dict[str, Any]:
        """최근 감정 분석 요약"""
        if not self.emotion_history:
            return {}
        
        # 최근 duration 초간의 데이터 필터링
        current_time = time.time()
        recent_results = [
            result for result in self.emotion_history
            if current_time - result.timestamp <= duration
        ]
        
        if not recent_results:
            return {}
        
        # 통계 계산
        emotions = [r.primary_emotion for r in recent_results]
        confidences = [r.confidence for r in recent_results]
        
        emotion_distribution = dict(Counter(emotions))
        
        # 면접 관련 지표 평균
        nervousness_levels = [r.nervousness_level for r in recent_results]
        confidence_levels = [r.confidence_level for r in recent_results]
        positivity_scores = [r.positivity_score for r in recent_results]
        stress_indicators = [r.stress_indicator for r in recent_results]
        
        summary = {
            'period': duration,
            'total_samples': len(recent_results),
            'emotion_distribution': emotion_distribution,
            'dominant_emotion': max(emotion_distribution, key=emotion_distribution.get),
            'average_confidence': np.mean(confidences),
            'emotional_state': self.current_emotional_state.copy(),
            'interview_metrics': {
                'average_nervousness': np.mean(nervousness_levels),
                'average_confidence': np.mean(confidence_levels),
                'average_positivity': np.mean(positivity_scores),
                'average_stress': np.mean(stress_indicators)
            }
        }
        
        return summary
    
    def get_interview_feedback(self) -> Dict[str, Any]:
        """면접 피드백 생성"""
        if not self.emotion_history:
            return {"feedback": "분석할 데이터가 없습니다."}
        
        summary = self.get_emotion_summary()
        
        if not summary:
            return {"feedback": "분석할 데이터가 없습니다."}
        
        metrics = summary.get('interview_metrics', {})
        
        feedback = {
            'overall_score': 0,
            'emotional_stability': 0,
            'confidence_assessment': 0,
            'stress_management': 0,
            'recommendations': []
        }
        
        # 전체 점수 계산 (0-100)
        nervousness = metrics.get('average_nervousness', 0)
        confidence = metrics.get('average_confidence', 0)
        positivity = metrics.get('average_positivity', 0)
        stress = metrics.get('average_stress', 0)
        
        overall_score = (
            (1.0 - nervousness) * 25 +  # 긴장도 (낮을수록 좋음)
            confidence * 30 +           # 자신감
            positivity * 25 +           # 긍정성
            (1.0 - stress) * 20         # 스트레스 (낮을수록 좋음)
        )
        
        feedback['overall_score'] = int(max(0, min(100, overall_score)))
        
        # 세부 평가
        feedback['emotional_stability'] = int((1.0 - nervousness - stress) * 100)
        feedback['confidence_assessment'] = int(confidence * 100)
        feedback['stress_management'] = int((1.0 - stress) * 100)
        
        # 추천사항 생성
        recommendations = []
        
        if nervousness > 0.6:
            recommendations.append("긴장을 완화하기 위해 심호흡을 해보세요.")
        
        if confidence < 0.4:
            recommendations.append("더 자신감 있는 어조로 말해보세요.")
        
        if stress > 0.5:
            recommendations.append("스트레스 관리가 필요합니다. 잠시 휴식을 취해보세요.")
        
        if positivity < 0.3:
            recommendations.append("더 긍정적인 표현을 사용해보세요.")
        
        if not recommendations:
            recommendations.append("좋은 감정 상태를 유지하고 있습니다!")
        
        feedback['recommendations'] = recommendations
        
        return feedback
    
    def _create_empty_result(self, text: str, language: str, timestamp: float) -> EmotionResult:
        """빈 결과 생성"""
        return EmotionResult(
            text=text,
            primary_emotion="neutral",
            emotion_scores={"neutral": 1.0},
            confidence=0.0,
            timestamp=timestamp or time.time(),
            processing_time=0.0,
            language=language
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        stats = self.stats.copy()
        stats['current_state'] = self.current_emotional_state.copy()
        stats['history_size'] = len(self.emotion_history)
        return stats


# 편의 함수
def create_emotion_detector(config=None) -> RealTimeEmotionDetector:
    """감정 분석기 인스턴스 생성"""
    return RealTimeEmotionDetector(config)

def test_emotion_models() -> bool:
    """감정 분석 모델 테스트"""
    try:
        detector = create_emotion_detector()
        success = detector.initialize()
        
        if success:
            # 테스트 텍스트
            test_texts = [
                "안녕하세요. 면접에 참여하게 되어 기쁩니다.",
                "Hello, I'm excited to be here for the interview.",
                "조금 긴장되지만 최선을 다하겠습니다."
            ]
            
            for text in test_texts:
                result = detector.analyze_text(text)
                logger.info(f"테스트 결과: {text[:20]}... -> {result.primary_emotion} ({result.confidence:.2f})")
            
            logger.info("감정 분석 모델 테스트 성공")
            return True
        else:
            logger.error("감정 분석 모델 테스트 실패")
            return False
            
    except Exception as e:
        logger.error(f"감정 분석 모델 테스트 오류: {e}")
        return False
