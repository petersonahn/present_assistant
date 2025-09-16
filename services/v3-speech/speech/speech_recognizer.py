"""
음성 인식 모듈
Whisper 모델을 사용한 실시간 음성-텍스트 변환
"""

import whisper
import torch
import numpy as np
import threading
import queue
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from collections import deque
import gc
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import warnings

from config.audio_config import get_config

logger = logging.getLogger(__name__)

# Transformers 경고 억제
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


@dataclass
class TranscriptionResult:
    """음성 인식 결과 데이터 구조"""
    text: str
    language: str
    confidence: float
    timestamp: float
    duration: float
    word_timestamps: List[Dict[str, Any]]
    is_final: bool = True
    processing_time: float = 0.0


class WhisperModelManager:
    """Whisper 모델 관리 클래스"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.model = None
        self.processor = None
        self.device = self._get_device()
        self.model_loaded = False
        self.model_lock = threading.Lock()
        
        # 모델 캐시
        self._model_cache = {}
        
    def _get_device(self) -> str:
        """사용할 디바이스 결정"""
        if self.config.whisper.DEVICE == "auto":
            if torch.cuda.is_available() and self.config.performance.USE_GPU:
                return "cuda"
            else:
                return "cpu"
        return self.config.whisper.DEVICE
    
    def load_model(self, force_reload: bool = False) -> bool:
        """Whisper 모델 로드"""
        with self.model_lock:
            if self.model_loaded and not force_reload:
                return True
            
            try:
                model_name = self.config.whisper.MODEL_NAME
                model_size = self.config.whisper.MODEL_SIZE
                
                logger.info(f"Whisper 모델 로딩 중: {model_size} (디바이스: {self.device})")
                
                # 캐시에서 모델 확인
                cache_key = f"{model_size}_{self.device}"
                if cache_key in self._model_cache and not force_reload:
                    self.model, self.processor = self._model_cache[cache_key]
                    logger.info("캐시된 모델 사용")
                else:
                    # Hugging Face Transformers 사용
                    if model_name.startswith("openai/"):
                        self.processor = WhisperProcessor.from_pretrained(model_name)
                        self.model = WhisperForConditionalGeneration.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                        )
                    else:
                        # OpenAI Whisper 사용 (백업)
                        self.model = whisper.load_model(model_size, device=self.device)
                        self.processor = None
                    
                    # 모델을 디바이스로 이동
                    if hasattr(self.model, 'to'):
                        self.model = self.model.to(self.device)
                    
                    # 캐시에 저장
                    self._model_cache[cache_key] = (self.model, self.processor)
                
                self.model_loaded = True
                logger.info("Whisper 모델 로딩 완료")
                return True
                
            except Exception as e:
                logger.error(f"Whisper 모델 로딩 실패: {e}")
                self.model_loaded = False
                return False
    
    def unload_model(self):
        """모델 언로드 (메모리 절약)"""
        with self.model_lock:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.processor is not None:
                del self.processor
                self.processor = None
            
            self.model_loaded = False
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 가비지 컬렉션
            gc.collect()
            
            logger.info("Whisper 모델 언로드 완료")
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            'model_name': self.config.whisper.MODEL_NAME,
            'model_size': self.config.whisper.MODEL_SIZE,
            'device': self.device,
            'loaded': self.model_loaded,
            'gpu_available': torch.cuda.is_available(),
            'gpu_memory': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        }


class AudioPreprocessor:
    """오디오 전처리 클래스"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.target_sample_rate = 16000  # Whisper 요구사항
    
    def preprocess_audio(self, audio: np.ndarray, original_sr: int) -> np.ndarray:
        """Whisper용 오디오 전처리"""
        try:
            # 샘플레이트 변환
            if original_sr != self.target_sample_rate:
                import librosa
                audio = librosa.resample(
                    audio, 
                    orig_sr=original_sr, 
                    target_sr=self.target_sample_rate
                )
            
            # 정규화
            if self.config.whisper.NORMALIZE:
                audio = self._normalize_audio(audio)
            
            # 패딩/트리밍
            audio = self._pad_or_trim(audio)
            
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"오디오 전처리 오류: {e}")
            return audio.astype(np.float32)
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """오디오 정규화"""
        # RMS 정규화
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            audio = audio / rms * 0.1  # 적절한 레벨로 조정
        
        # 클리핑 방지
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def _pad_or_trim(self, audio: np.ndarray, length: Optional[int] = None) -> np.ndarray:
        """오디오 패딩 또는 트리밍"""
        if length is None:
            # 최소 길이 보장 (0.1초)
            min_length = int(0.1 * self.target_sample_rate)
            if len(audio) < min_length:
                audio = np.pad(audio, (0, min_length - len(audio)))
        else:
            if len(audio) > length:
                audio = audio[:length]
            elif len(audio) < length:
                audio = np.pad(audio, (0, length - len(audio)))
        
        return audio


class LanguageDetector:
    """언어 감지 클래스"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.language_history = deque(maxlen=10)
        self.confidence_threshold = 0.7
    
    def detect_language(self, audio: np.ndarray, model, processor) -> Tuple[str, float]:
        """언어 감지"""
        try:
            if self.config.whisper.LANGUAGE != "auto":
                return self.config.whisper.LANGUAGE, 1.0
            
            # Whisper 모델을 사용한 언어 감지
            if processor is not None:
                # Transformers 방식
                input_features = processor(
                    audio, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).input_features
                
                # 언어 토큰 예측
                with torch.no_grad():
                    predicted_ids = model.generate(
                        input_features.to(model.device),
                        max_length=5,
                        num_beams=1,
                        do_sample=False
                    )
                
                # 언어 디코딩 (간단한 휴리스틱)
                language = "ko" if np.random.random() > 0.5 else "en"  # 실제 구현 필요
                confidence = 0.8
                
            else:
                # OpenAI Whisper 방식
                # 짧은 세그먼트로 언어 감지
                mel = whisper.log_mel_spectrogram(audio).to(model.device)
                _, probs = model.detect_language(mel)
                language = max(probs, key=probs.get)
                confidence = probs[language]
            
            # 히스토리 업데이트
            self.language_history.append((language, confidence))
            
            # 안정화된 언어 반환
            if len(self.language_history) >= 3:
                recent_languages = [lang for lang, conf in self.language_history[-3:]]
                if recent_languages.count(language) >= 2:
                    return language, confidence
            
            return language, confidence
            
        except Exception as e:
            logger.error(f"언어 감지 오류: {e}")
            return "ko", 0.5  # 기본값: 한국어


class RealTimeSpeechRecognizer:
    """실시간 음성 인식 클래스"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        
        # 모델 관리
        self.model_manager = WhisperModelManager(config)
        self.preprocessor = AudioPreprocessor(config)
        self.language_detector = LanguageDetector(config)
        
        # 처리 큐
        self.audio_queue = queue.Queue(maxsize=20)
        self.result_queue = queue.Queue()
        
        # 스레드 관리
        self.processing_thread = None
        self.is_running = False
        self.stop_event = threading.Event()
        
        # 오디오 버퍼 (연속 처리용)
        self.audio_buffer = deque()
        self.buffer_duration = 5.0  # 5초 버퍼
        self.overlap_duration = 1.0  # 1초 오버랩
        
        # 결과 캐시 및 중복 제거
        self.result_cache = deque(maxlen=50)
        self.last_transcription = ""
        self.last_confidence = 0.0
        
        # 통계
        self.stats = {
            'total_processed': 0,
            'successful_transcriptions': 0,
            'average_confidence': 0.0,
            'average_processing_time': 0.0,
            'language_distribution': {}
        }
    
    def start(self) -> bool:
        """음성 인식 시작"""
        if self.is_running:
            logger.warning("이미 음성 인식이 실행 중입니다")
            return False
        
        # 모델 로드
        if not self.model_manager.load_model():
            logger.error("모델 로드 실패")
            return False
        
        try:
            self.is_running = True
            self.stop_event.clear()
            
            # 처리 스레드 시작
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True
            )
            self.processing_thread.start()
            
            logger.info("실시간 음성 인식 시작")
            return True
            
        except Exception as e:
            logger.error(f"음성 인식 시작 실패: {e}")
            self.is_running = False
            return False
    
    def stop(self):
        """음성 인식 중지"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        # 스레드 종료 대기
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # 큐 비우기
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("실시간 음성 인식 중지")
    
    def add_audio_chunk(self, audio: np.ndarray, timestamp: float, sample_rate: int) -> bool:
        """오디오 청크 추가"""
        if not self.is_running:
            return False
        
        try:
            # 큐가 가득 차면 오래된 데이터 제거
            if self.audio_queue.full():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    pass
            
            # 새 데이터 추가
            self.audio_queue.put_nowait({
                'audio': audio,
                'timestamp': timestamp,
                'sample_rate': sample_rate
            })
            
            return True
            
        except Exception as e:
            logger.error(f"오디오 청크 추가 오류: {e}")
            return False
    
    def get_latest_result(self) -> Optional[TranscriptionResult]:
        """최신 인식 결과 반환"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_all_results(self) -> List[TranscriptionResult]:
        """모든 대기 중인 결과 반환"""
        results = []
        while True:
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        return results
    
    def _processing_loop(self):
        """오디오 처리 루프"""
        while not self.stop_event.is_set():
            try:
                # 오디오 데이터 가져오기
                audio_data = self.audio_queue.get(timeout=0.5)
                
                # 전처리
                processed_audio = self.preprocessor.preprocess_audio(
                    audio_data['audio'],
                    audio_data['sample_rate']
                )
                
                # 버퍼에 추가
                self._add_to_buffer(processed_audio, audio_data['timestamp'])
                
                # 충분한 데이터가 쌓이면 처리
                if self._should_process_buffer():
                    self._process_buffer()
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"처리 루프 오류: {e}")
    
    def _add_to_buffer(self, audio: np.ndarray, timestamp: float):
        """버퍼에 오디오 추가"""
        self.audio_buffer.append({
            'audio': audio,
            'timestamp': timestamp
        })
        
        # 버퍼 크기 제한
        buffer_samples = int(self.buffer_duration * 16000)
        total_samples = sum(len(item['audio']) for item in self.audio_buffer)
        
        while total_samples > buffer_samples and self.audio_buffer:
            removed = self.audio_buffer.popleft()
            total_samples -= len(removed['audio'])
    
    def _should_process_buffer(self) -> bool:
        """버퍼 처리 여부 결정"""
        if not self.audio_buffer:
            return False
        
        # 최소 1초 데이터 필요
        total_duration = sum(len(item['audio']) for item in self.audio_buffer) / 16000
        return total_duration >= 1.0
    
    def _process_buffer(self):
        """버퍼 처리 및 음성 인식"""
        if not self.audio_buffer:
            return
        
        try:
            start_time = time.time()
            
            # 버퍼 데이터 결합
            audio_chunks = [item['audio'] for item in self.audio_buffer]
            combined_audio = np.concatenate(audio_chunks)
            
            # 타임스탬프 정보
            first_timestamp = self.audio_buffer[0]['timestamp']
            duration = len(combined_audio) / 16000
            
            # 언어 감지
            language, lang_confidence = self.language_detector.detect_language(
                combined_audio,
                self.model_manager.model,
                self.model_manager.processor
            )
            
            # 음성 인식 수행
            transcription_result = self._transcribe_audio(
                combined_audio,
                language,
                first_timestamp,
                duration
            )
            
            # 처리 시간 기록
            processing_time = time.time() - start_time
            transcription_result.processing_time = processing_time
            
            # 중복 제거 및 품질 검사
            if self._is_valid_result(transcription_result):
                # 결과 큐에 추가
                if not self.result_queue.full():
                    self.result_queue.put(transcription_result)
                
                # 통계 업데이트
                self._update_stats(transcription_result, processing_time)
            
            # 버퍼 일부 제거 (오버랩 유지)
            self._trim_buffer()
            
        except Exception as e:
            logger.error(f"버퍼 처리 오류: {e}")
    
    def _transcribe_audio(self, audio: np.ndarray, language: str, 
                         timestamp: float, duration: float) -> TranscriptionResult:
        """오디오 음성 인식"""
        try:
            with self.model_manager.model_lock:
                if self.model_manager.processor is not None:
                    # Transformers 방식
                    result = self._transcribe_with_transformers(
                        audio, language, timestamp, duration
                    )
                else:
                    # OpenAI Whisper 방식
                    result = self._transcribe_with_openai(
                        audio, language, timestamp, duration
                    )
                
                return result
                
        except Exception as e:
            logger.error(f"음성 인식 오류: {e}")
            return TranscriptionResult(
                text="",
                language=language,
                confidence=0.0,
                timestamp=timestamp,
                duration=duration,
                word_timestamps=[]
            )
    
    def _transcribe_with_transformers(self, audio: np.ndarray, language: str,
                                    timestamp: float, duration: float) -> TranscriptionResult:
        """Transformers Whisper로 음성 인식"""
        processor = self.model_manager.processor
        model = self.model_manager.model
        
        # 입력 특징 추출
        input_features = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(model.device)
        
        # 언어 토큰 설정
        if language == "ko":
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="korean", task="transcribe")
        elif language == "en":
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
        else:
            forced_decoder_ids = None
        
        # 생성
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=448,
                num_beams=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # 디코딩
        transcription = processor.batch_decode(
            predicted_ids.sequences,
            skip_special_tokens=True
        )[0]
        
        # 신뢰도 계산 (간단한 방법)
        if hasattr(predicted_ids, 'scores') and predicted_ids.scores:
            scores = torch.stack(predicted_ids.scores, dim=1)
            confidence = torch.mean(torch.softmax(scores, dim=-1).max(dim=-1)[0]).item()
        else:
            confidence = 0.8  # 기본값
        
        return TranscriptionResult(
            text=transcription.strip(),
            language=language,
            confidence=confidence,
            timestamp=timestamp,
            duration=duration,
            word_timestamps=[]  # Transformers에서는 단어별 타임스탬프 추출이 복잡
        )
    
    def _transcribe_with_openai(self, audio: np.ndarray, language: str,
                              timestamp: float, duration: float) -> TranscriptionResult:
        """OpenAI Whisper로 음성 인식"""
        model = self.model_manager.model
        
        # Whisper 인식 수행
        result = model.transcribe(
            audio,
            language=language if language != "auto" else None,
            word_timestamps=self.config.whisper.RETURN_TIMESTAMPS,
            verbose=False
        )
        
        # 결과 파싱
        text = result.get("text", "").strip()
        detected_language = result.get("language", language)
        
        # 단어별 타임스탬프 추출
        word_timestamps = []
        if "segments" in result:
            for segment in result["segments"]:
                if "words" in segment:
                    for word in segment["words"]:
                        word_timestamps.append({
                            "word": word.get("word", ""),
                            "start": word.get("start", 0.0),
                            "end": word.get("end", 0.0),
                            "probability": word.get("probability", 0.0)
                        })
        
        # 전체 신뢰도 계산
        if word_timestamps:
            confidence = np.mean([w["probability"] for w in word_timestamps])
        else:
            confidence = 0.8  # 기본값
        
        return TranscriptionResult(
            text=text,
            language=detected_language,
            confidence=confidence,
            timestamp=timestamp,
            duration=duration,
            word_timestamps=word_timestamps
        )
    
    def _is_valid_result(self, result: TranscriptionResult) -> bool:
        """결과 유효성 검사"""
        # 빈 텍스트 필터링
        if not result.text or len(result.text.strip()) < 2:
            return False
        
        # 신뢰도 임계값
        if result.confidence < 0.3:
            return False
        
        # 중복 텍스트 제거
        if result.text == self.last_transcription:
            return False
        
        # 너무 유사한 결과 필터링
        similarity = self._calculate_similarity(result.text, self.last_transcription)
        if similarity > 0.9:
            return False
        
        self.last_transcription = result.text
        self.last_confidence = result.confidence
        
        return True
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """텍스트 유사도 계산 (간단한 방법)"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _trim_buffer(self):
        """버퍼 트리밍 (오버랩 유지)"""
        if not self.audio_buffer:
            return
        
        # 오버랩 유지를 위해 마지막 1초 데이터만 유지
        overlap_samples = int(self.overlap_duration * 16000)
        
        # 마지막 청크들의 총 샘플 수 계산
        total_samples = 0
        keep_items = []
        
        for item in reversed(self.audio_buffer):
            total_samples += len(item['audio'])
            keep_items.append(item)
            
            if total_samples >= overlap_samples:
                break
        
        # 버퍼 업데이트
        self.audio_buffer = deque(reversed(keep_items))
    
    def _update_stats(self, result: TranscriptionResult, processing_time: float):
        """통계 업데이트"""
        self.stats['total_processed'] += 1
        if result.text:
            self.stats['successful_transcriptions'] += 1
        
        # 평균 신뢰도 업데이트
        total = self.stats['total_processed']
        self.stats['average_confidence'] = (
            (self.stats['average_confidence'] * (total - 1) + result.confidence) / total
        )
        
        # 평균 처리 시간 업데이트
        self.stats['average_processing_time'] = (
            (self.stats['average_processing_time'] * (total - 1) + processing_time) / total
        )
        
        # 언어 분포 업데이트
        lang = result.language
        if lang in self.stats['language_distribution']:
            self.stats['language_distribution'][lang] += 1
        else:
            self.stats['language_distribution'][lang] = 1
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        stats = self.stats.copy()
        stats.update({
            'is_running': self.is_running,
            'queue_size': self.audio_queue.qsize(),
            'buffer_size': len(self.audio_buffer),
            'model_info': self.model_manager.get_model_info()
        })
        return stats
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# 편의 함수
def create_speech_recognizer(config=None) -> RealTimeSpeechRecognizer:
    """음성 인식기 인스턴스 생성"""
    return RealTimeSpeechRecognizer(config)

def test_whisper_model(model_size: str = "small") -> bool:
    """Whisper 모델 테스트"""
    try:
        # 테스트용 더미 오디오 생성
        test_audio = np.random.randn(16000).astype(np.float32)  # 1초
        
        # 모델 로드 테스트
        config = get_config()
        config.whisper.MODEL_SIZE = model_size
        
        recognizer = create_speech_recognizer(config)
        success = recognizer.start()
        
        if success:
            recognizer.stop()
            logger.info(f"Whisper {model_size} 모델 테스트 성공")
            return True
        else:
            logger.error(f"Whisper {model_size} 모델 테스트 실패")
            return False
            
    except Exception as e:
        logger.error(f"Whisper 모델 테스트 오류: {e}")
        return False
