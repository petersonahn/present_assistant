"""
에러 처리 및 안정성 모듈
서비스 전반의 예외 처리, 복구 로직, 모니터링 기능
"""

import logging
import time
import threading
import traceback
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from collections import deque, defaultdict
from enum import Enum
import functools
import psutil
import gc

logger = logging.getLogger(__name__)


class ErrorLevel(Enum):
    """에러 레벨 정의"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComponentType(Enum):
    """컴포넌트 타입 정의"""
    AUDIO_CAPTURE = "audio_capture"
    SPEECH_ANALYZER = "speech_analyzer"
    SPEECH_RECOGNIZER = "speech_recognizer"
    EMOTION_DETECTOR = "emotion_detector"
    MAIN_SERVICE = "main_service"
    API_SERVER = "api_server"


@dataclass
class ErrorInfo:
    """에러 정보 데이터 구조"""
    timestamp: float
    component: ComponentType
    error_type: str
    error_message: str
    level: ErrorLevel
    traceback_info: str
    context: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False


class CircuitBreaker:
    """서킷 브레이커 패턴 구현"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func, *args, **kwargs):
        """함수 호출 (서킷 브레이커 적용)"""
        with self.lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            
            except Exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """리셋 시도 여부 결정"""
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
    
    def _on_success(self):
        """성공 시 처리"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """실패 시 처리"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
    
    def get_state(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "failure_threshold": self.failure_threshold
        }


class RetryHandler:
    """재시도 처리 클래스"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 exponential_backoff: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.exponential_backoff = exponential_backoff
    
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.retry(func, *args, **kwargs)
        return wrapper
    
    def retry(self, func, *args, **kwargs):
        """재시도 로직"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    logger.error(f"최대 재시도 횟수 초과: {func.__name__}")
                    break
                
                # 재시도 지연
                delay = self._calculate_delay(attempt)
                logger.warning(f"재시도 {attempt + 1}/{self.max_retries} "
                             f"({delay:.1f}초 후): {func.__name__}")
                time.sleep(delay)
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """재시도 지연 시간 계산"""
        if self.exponential_backoff:
            return self.base_delay * (2 ** attempt)
        else:
            return self.base_delay


class HealthMonitor:
    """시스템 헬스 모니터링"""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_check_time = self.start_time
        self.health_history = deque(maxlen=100)
        self.monitoring_active = False
        self.monitor_thread = None
        
        # 임계값 설정
        self.cpu_threshold = 80.0  # CPU 사용률 (%)
        self.memory_threshold = 85.0  # 메모리 사용률 (%)
        self.disk_threshold = 90.0  # 디스크 사용률 (%)
    
    def start_monitoring(self, interval: float = 30.0):
        """모니터링 시작"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("헬스 모니터링 시작")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("헬스 모니터링 중지")
    
    def _monitoring_loop(self, interval: float):
        """모니터링 루프"""
        while self.monitoring_active:
            try:
                health_data = self._collect_health_data()
                self.health_history.append(health_data)
                
                # 경고 확인
                self._check_health_warnings(health_data)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"헬스 모니터링 오류: {e}")
                time.sleep(interval)
    
    def _collect_health_data(self) -> Dict[str, Any]:
        """헬스 데이터 수집"""
        try:
            # CPU 정보
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # 메모리 정보
            memory = psutil.virtual_memory()
            
            # 디스크 정보 (현재 디렉토리)
            disk = psutil.disk_usage('/')
            
            # 프로세스 정보
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return {
                'timestamp': time.time(),
                'uptime': time.time() - self.start_time,
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': (disk.used / disk.total) * 100
                },
                'process': {
                    'memory_rss': process_memory.rss,
                    'memory_vms': process_memory.vms,
                    'cpu_percent': process.cpu_percent()
                }
            }
            
        except Exception as e:
            logger.error(f"헬스 데이터 수집 오류: {e}")
            return {
                'timestamp': time.time(),
                'error': str(e)
            }
    
    def _check_health_warnings(self, health_data: Dict[str, Any]):
        """헬스 경고 확인"""
        warnings = []
        
        # CPU 사용률 확인
        cpu_percent = health_data.get('cpu', {}).get('percent', 0)
        if cpu_percent > self.cpu_threshold:
            warnings.append(f"높은 CPU 사용률: {cpu_percent:.1f}%")
        
        # 메모리 사용률 확인
        memory_percent = health_data.get('memory', {}).get('percent', 0)
        if memory_percent > self.memory_threshold:
            warnings.append(f"높은 메모리 사용률: {memory_percent:.1f}%")
        
        # 디스크 사용률 확인
        disk_percent = health_data.get('disk', {}).get('percent', 0)
        if disk_percent > self.disk_threshold:
            warnings.append(f"높은 디스크 사용률: {disk_percent:.1f}%")
        
        # 경고 로깅
        for warning in warnings:
            logger.warning(f"시스템 헬스 경고: {warning}")
    
    def get_current_health(self) -> Dict[str, Any]:
        """현재 헬스 상태 반환"""
        if not self.health_history:
            return self._collect_health_data()
        
        return self.health_history[-1]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """헬스 요약 정보 반환"""
        if not self.health_history:
            return {}
        
        recent_data = list(self.health_history)[-10:]  # 최근 10개
        
        cpu_values = [d.get('cpu', {}).get('percent', 0) for d in recent_data]
        memory_values = [d.get('memory', {}).get('percent', 0) for d in recent_data]
        
        return {
            'monitoring_active': self.monitoring_active,
            'uptime': time.time() - self.start_time,
            'data_points': len(self.health_history),
            'avg_cpu_percent': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
            'avg_memory_percent': sum(memory_values) / len(memory_values) if memory_values else 0,
            'max_cpu_percent': max(cpu_values) if cpu_values else 0,
            'max_memory_percent': max(memory_values) if memory_values else 0
        }


class ErrorRecoveryManager:
    """에러 복구 관리자"""
    
    def __init__(self):
        self.error_history = deque(maxlen=500)
        self.error_counts = defaultdict(int)
        self.circuit_breakers = {}
        self.recovery_strategies = {}
        self.health_monitor = HealthMonitor()
        
        # 기본 복구 전략 등록
        self._register_default_recovery_strategies()
    
    def register_recovery_strategy(self, component: ComponentType, 
                                 strategy: Callable[[Exception, Dict], bool]):
        """복구 전략 등록"""
        self.recovery_strategies[component] = strategy
        logger.info(f"복구 전략 등록: {component.value}")
    
    def get_circuit_breaker(self, component: ComponentType) -> CircuitBreaker:
        """컴포넌트별 서킷 브레이커 반환"""
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreaker()
        return self.circuit_breakers[component]
    
    def handle_error(self, component: ComponentType, error: Exception, 
                    context: Dict[str, Any] = None) -> bool:
        """에러 처리 및 복구 시도"""
        error_info = ErrorInfo(
            timestamp=time.time(),
            component=component,
            error_type=type(error).__name__,
            error_message=str(error),
            level=self._assess_error_level(error),
            traceback_info=traceback.format_exc(),
            context=context or {}
        )
        
        # 에러 기록
        self.error_history.append(error_info)
        self.error_counts[f"{component.value}_{error_info.error_type}"] += 1
        
        # 로깅
        logger.error(f"에러 발생 [{component.value}]: {error_info.error_message}")
        
        # 복구 시도
        recovery_success = self._attempt_recovery(error_info, error)
        
        # 복구 결과 업데이트
        error_info.recovery_attempted = True
        error_info.recovery_successful = recovery_success
        
        return recovery_success
    
    def _assess_error_level(self, error: Exception) -> ErrorLevel:
        """에러 레벨 평가"""
        error_type = type(error).__name__
        
        # 치명적 에러
        if error_type in ['SystemExit', 'KeyboardInterrupt', 'MemoryError']:
            return ErrorLevel.CRITICAL
        
        # 높은 수준 에러
        elif error_type in ['ConnectionError', 'TimeoutError', 'FileNotFoundError']:
            return ErrorLevel.HIGH
        
        # 중간 수준 에러
        elif error_type in ['ValueError', 'TypeError', 'AttributeError']:
            return ErrorLevel.MEDIUM
        
        # 낮은 수준 에러
        else:
            return ErrorLevel.LOW
    
    def _attempt_recovery(self, error_info: ErrorInfo, error: Exception) -> bool:
        """복구 시도"""
        component = error_info.component
        
        # 복구 전략이 등록된 경우
        if component in self.recovery_strategies:
            try:
                strategy = self.recovery_strategies[component]
                return strategy(error, error_info.context)
            except Exception as recovery_error:
                logger.error(f"복구 전략 실행 실패: {recovery_error}")
        
        # 기본 복구 시도
        return self._default_recovery(error_info)
    
    def _default_recovery(self, error_info: ErrorInfo) -> bool:
        """기본 복구 로직"""
        try:
            # 메모리 정리
            if "memory" in error_info.error_message.lower():
                gc.collect()
                logger.info("메모리 정리 수행")
                return True
            
            # 파일 관련 에러
            if "file" in error_info.error_message.lower():
                logger.info("파일 시스템 복구 시도")
                return False  # 파일 에러는 수동 처리 필요
            
            # 네트워크 관련 에러
            if any(keyword in error_info.error_message.lower() 
                   for keyword in ['connection', 'network', 'timeout']):
                logger.info("네트워크 복구 대기")
                time.sleep(2.0)  # 잠시 대기
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"기본 복구 로직 오류: {e}")
            return False
    
    def _register_default_recovery_strategies(self):
        """기본 복구 전략들 등록"""
        
        def audio_recovery(error: Exception, context: Dict) -> bool:
            """오디오 관련 복구"""
            logger.info("오디오 시스템 복구 시도")
            time.sleep(1.0)
            return True
        
        def model_recovery(error: Exception, context: Dict) -> bool:
            """모델 관련 복구"""
            logger.info("모델 재로딩 시도")
            # 실제로는 모델 재로딩 로직이 필요
            return False
        
        def api_recovery(error: Exception, context: Dict) -> bool:
            """API 관련 복구"""
            logger.info("API 연결 복구 시도")
            time.sleep(0.5)
            return True
        
        # 복구 전략 등록
        self.register_recovery_strategy(ComponentType.AUDIO_CAPTURE, audio_recovery)
        self.register_recovery_strategy(ComponentType.SPEECH_RECOGNIZER, model_recovery)
        self.register_recovery_strategy(ComponentType.EMOTION_DETECTOR, model_recovery)
        self.register_recovery_strategy(ComponentType.API_SERVER, api_recovery)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """에러 통계 반환"""
        if not self.error_history:
            return {}
        
        # 시간대별 에러 분포
        recent_errors = [e for e in self.error_history 
                        if time.time() - e.timestamp < 3600]  # 최근 1시간
        
        # 컴포넌트별 에러 분포
        component_errors = defaultdict(int)
        level_errors = defaultdict(int)
        
        for error in recent_errors:
            component_errors[error.component.value] += 1
            level_errors[error.level.value] += 1
        
        # 복구 성공률
        recovery_attempted = sum(1 for e in recent_errors if e.recovery_attempted)
        recovery_successful = sum(1 for e in recent_errors if e.recovery_successful)
        recovery_rate = (recovery_successful / recovery_attempted * 100 
                        if recovery_attempted > 0 else 0)
        
        return {
            'total_errors': len(self.error_history),
            'recent_errors': len(recent_errors),
            'component_distribution': dict(component_errors),
            'level_distribution': dict(level_errors),
            'recovery_rate': recovery_rate,
            'most_common_errors': dict(self.error_counts)
        }
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """서킷 브레이커 상태 반환"""
        status = {}
        for component, breaker in self.circuit_breakers.items():
            status[component.value] = breaker.get_state()
        return status
    
    def start_monitoring(self):
        """모니터링 시작"""
        self.health_monitor.start_monitoring()
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.health_monitor.stop_monitoring()
    
    def get_health_status(self) -> Dict[str, Any]:
        """전체 헬스 상태 반환"""
        return {
            'health_summary': self.health_monitor.get_health_summary(),
            'current_health': self.health_monitor.get_current_health(),
            'error_statistics': self.get_error_statistics(),
            'circuit_breakers': self.get_circuit_breaker_status()
        }


# 전역 에러 복구 관리자 인스턴스
_error_recovery_manager = None


def get_error_recovery_manager() -> ErrorRecoveryManager:
    """전역 에러 복구 관리자 반환"""
    global _error_recovery_manager
    if _error_recovery_manager is None:
        _error_recovery_manager = ErrorRecoveryManager()
    return _error_recovery_manager


def handle_component_error(component: ComponentType, error: Exception, 
                          context: Dict[str, Any] = None) -> bool:
    """컴포넌트 에러 처리 (편의 함수)"""
    manager = get_error_recovery_manager()
    return manager.handle_error(component, error, context)


def with_error_handling(component: ComponentType, context: Dict[str, Any] = None):
    """에러 처리 데코레이터"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                recovery_success = handle_component_error(component, e, context)
                if not recovery_success:
                    raise e
                # 복구 성공 시 함수 재실행
                return func(*args, **kwargs)
        return wrapper
    return decorator


def with_circuit_breaker(component: ComponentType):
    """서킷 브레이커 데코레이터"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_error_recovery_manager()
            breaker = manager.get_circuit_breaker(component)
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


def with_retry(max_retries: int = 3, base_delay: float = 1.0):
    """재시도 데코레이터"""
    retry_handler = RetryHandler(max_retries, base_delay)
    return retry_handler
