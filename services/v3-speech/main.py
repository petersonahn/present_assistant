"""
실시간 음성 분석 시스템 - FastAPI 서버
v3-speech 서비스 메인 엔드포인트
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import threading
import time
import logging
import json
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
import orjson
from functools import lru_cache
import gc
from collections import deque

from speech.speech_service import RealTimeSpeechService, SpeechAnalysisCallback, create_speech_service
from speech.audio_capture import list_audio_devices, test_audio_device
from config.audio_config import get_config, update_config

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 전역 서비스 인스턴스
speech_service: Optional[RealTimeSpeechService] = None
service_lock = threading.Lock()

# 실시간 결과 스트리밍을 위한 클라이언트 관리
streaming_clients = {}
client_counter = 0


class StreamingCallback(SpeechAnalysisCallback):
    """실시간 스트리밍을 위한 콜백 클래스"""
    
    def __init__(self):
        self.results_buffer = deque(maxlen=100)
        self.connected_clients = set()
    
    def add_client(self, client_id: str):
        """클라이언트 추가"""
        self.connected_clients.add(client_id)
        logger.info(f"스트리밍 클라이언트 추가: {client_id}")
    
    def remove_client(self, client_id: str):
        """클라이언트 제거"""
        self.connected_clients.discard(client_id)
        logger.info(f"스트리밍 클라이언트 제거: {client_id}")
    
    def on_complete_analysis(self, result):
        """완료된 분석 결과를 버퍼에 추가"""
        try:
            # 결과를 JSON 직렬화 가능한 형태로 변환
            result_dict = {
                'timestamp': result.timestamp,
                'overall_score': result.overall_score,
                'audio_info': result.audio_info,
                'processing_time': result.processing_time
            }
            
            # 음성 특징 추가
            if result.speech_features:
                result_dict['speech_features'] = {
                    'speaking_rate': result.speech_features.speaking_rate,
                    'volume_level': result.speech_features.volume_level,
                    'clarity_score': result.speech_features.clarity_score,
                    'interview_score': result.speech_features.interview_score,
                    'confidence_level': result.speech_features.confidence_level,
                    'nervousness_indicator': result.speech_features.nervousness_indicator
                }
            
            # 음성 인식 결과 추가
            if result.transcription:
                result_dict['transcription'] = {
                    'text': result.transcription.text,
                    'language': result.transcription.language,
                    'confidence': result.transcription.confidence
                }
            
            # 감정 분석 결과 추가
            if result.emotion:
                result_dict['emotion'] = {
                    'primary_emotion': result.emotion.primary_emotion,
                    'confidence': result.emotion.confidence,
                    'confidence_level': result.emotion.confidence_level,
                    'nervousness_level': result.emotion.nervousness_level,
                    'positivity_score': result.emotion.positivity_score
                }
            
            # 면접 피드백 추가
            if result.interview_feedback:
                result_dict['interview_feedback'] = result.interview_feedback
            
            # 버퍼에 추가
            self.results_buffer.append(result_dict)
            
        except Exception as e:
            logger.error(f"스트리밍 콜백 처리 오류: {e}")
    
    def get_latest_results(self, count: int = 10) -> List[Dict]:
        """최신 결과들 반환"""
        return list(self.results_buffer)[-count:]


# 전역 스트리밍 콜백
streaming_callback = StreamingCallback()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    global speech_service
    
    # 시작 시 초기화
    logger.info("v3-speech 서비스 초기화 시작...")
    
    try:
        with service_lock:
            speech_service = create_speech_service()
            
            # 스트리밍 콜백 등록
            speech_service.add_callback(streaming_callback)
            
            # 서비스 초기화
            success = await speech_service.initialize()
            
            if success:
                logger.info("v3-speech 서비스 초기화 완료!")
            else:
                logger.warning("일부 컴포넌트 초기화 실패 (제한적 기능 제공)")
        
        yield
        
    except Exception as e:
        logger.error(f"서비스 초기화 실패: {e}")
        yield
    
    # 종료 시 정리
    logger.info("v3-speech 서비스 정리 시작...")
    
    try:
        if speech_service:
            speech_service.cleanup()
        logger.info("v3-speech 서비스 정리 완료")
    except Exception as e:
        logger.error(f"서비스 정리 오류: {e}")


# FastAPI 앱 생성
app = FastAPI(
    title="실시간 음성 분석 시스템",
    description="AI 기반 실시간 음성 분석 및 면접 피드백 API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 최적화된 JSON 응답 함수
def fast_json_response(content: Dict, status_code: int = 200) -> JSONResponse:
    """orjson을 사용한 빠른 JSON 응답"""
    return JSONResponse(
        content=orjson.loads(orjson.dumps(content)),
        status_code=status_code
    )


# 캐시된 설정 정보
@lru_cache(maxsize=1)
def get_cached_config_info():
    """설정 정보 캐싱"""
    config = get_config()
    return config.to_dict()


@app.get("/", response_class=HTMLResponse)
async def root():
    """메인 웹 페이지"""
    return HTMLResponse("""
    <html>
        <head>
            <title>v3-speech API Service</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #2c3e50; }
                .endpoint { margin: 10px 0; padding: 10px; background: #f8f9fa; border-left: 4px solid #007bff; }
                .method { font-weight: bold; color: #007bff; }
            </style>
        </head>
        <body>
            <h1>🎤 v3-speech API Service</h1>
            <p>실시간 음성 분석 API 서비스가 실행 중입니다.</p>
            
            <h2>📋 주요 엔드포인트</h2>
            <div class="endpoint">
                <span class="method">GET</span> <a href="/health">/health</a> - 헬스체크
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <a href="/docs">/docs</a> - API 문서
            </div>
            <div class="endpoint">
                <span class="method">POST</span> /speech/start_realtime - 실시간 분석 시작
            </div>
            <div class="endpoint">
                <span class="method">POST</span> /speech/stop_realtime - 실시간 분석 중지
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /speech/status - 현재 상태 조회
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /speech/stream - 실시간 결과 스트리밍
            </div>
            
            <h2>🔧 관리 기능</h2>
            <div class="endpoint">
                <span class="method">GET</span> <a href="/audio/devices">/audio/devices</a> - 오디오 디바이스 목록
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <a href="/config">/config</a> - 설정 정보
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <a href="/stats">/stats</a> - 서비스 통계
            </div>
            
            <p><strong>프론트엔드:</strong> <a href="http://localhost:3000">http://localhost:3000</a></p>
        </body>
    </html>
    """)


@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    global speech_service
    
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "service_initialized": speech_service is not None,
        "components": {}
    }
    
    if speech_service:
        try:
            # 각 컴포넌트 상태 확인
            health_status["components"]["audio_capture"] = speech_service.audio_capture is not None
            health_status["components"]["speech_analyzer"] = speech_service.speech_analyzer is not None
            health_status["components"]["speech_recognizer"] = (
                speech_service.speech_recognizer is not None and 
                speech_service.speech_recognizer.model_manager.model_loaded
            )
            health_status["components"]["emotion_detector"] = (
                speech_service.emotion_detector is not None and
                (speech_service.emotion_detector.korean_analyzer.model_loaded or
                 speech_service.emotion_detector.multilingual_analyzer.model_loaded)
            )
            
            # 전체 상태 결정
            component_health = list(health_status["components"].values())
            if all(component_health):
                health_status["status"] = "healthy"
            elif any(component_health):
                health_status["status"] = "degraded"
            else:
                health_status["status"] = "unhealthy"
                
        except Exception as e:
            logger.error(f"헬스체크 오류: {e}")
            health_status["status"] = "error"
            health_status["error"] = str(e)
    else:
        health_status["status"] = "uninitialized"
    
    return fast_json_response(health_status)


@app.get("/config")
async def get_config_info():
    """설정 정보 반환"""
    try:
        config_info = get_cached_config_info()
        return fast_json_response({
            "success": True,
            "config": config_info
        })
    except Exception as e:
        logger.error(f"설정 정보 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"설정 정보 조회 실패: {str(e)}")


@app.get("/audio/devices")
async def get_audio_devices():
    """사용 가능한 오디오 디바이스 목록 반환"""
    try:
        devices = list_audio_devices()
        
        # 각 디바이스 테스트 (백그라운드에서)
        for device in devices:
            device['tested'] = False  # 기본값
        
        return fast_json_response({
            "success": True,
            "devices": devices,
            "total_count": len(devices)
        })
        
    except Exception as e:
        logger.error(f"오디오 디바이스 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"디바이스 조회 실패: {str(e)}")


@app.post("/audio/test_device/{device_id}")
async def test_device(device_id: int):
    """특정 오디오 디바이스 테스트"""
    try:
        is_working = test_audio_device(device_id)
        
        return fast_json_response({
            "success": True,
            "device_id": device_id,
            "is_working": is_working,
            "message": "디바이스가 정상 작동합니다" if is_working else "디바이스 테스트 실패"
        })
        
    except Exception as e:
        logger.error(f"디바이스 테스트 오류: {e}")
        raise HTTPException(status_code=500, detail=f"디바이스 테스트 실패: {str(e)}")


@app.post("/speech/start_realtime")
async def start_realtime_analysis(data: Optional[Dict] = None):
    """실시간 음성 분석 시작"""
    global speech_service
    
    if not speech_service:
        raise HTTPException(status_code=503, detail="서비스가 초기화되지 않았습니다")
    
    try:
        # 디바이스 ID 추출 (선택사항)
        device_id = None
        if data and "device_id" in data:
            device_id = data["device_id"]
        
        # 실시간 분석 시작
        success = speech_service.start_realtime_analysis(device_id)
        
        if success:
            return fast_json_response({
                "success": True,
                "message": "실시간 음성 분석이 시작되었습니다",
                "device_id": device_id,
                "timestamp": time.time()
            })
        else:
            raise HTTPException(status_code=500, detail="실시간 분석 시작 실패")
            
    except Exception as e:
        logger.error(f"실시간 분석 시작 오류: {e}")
        raise HTTPException(status_code=500, detail=f"분석 시작 실패: {str(e)}")


@app.post("/speech/stop_realtime")
async def stop_realtime_analysis():
    """실시간 음성 분석 중지"""
    global speech_service
    
    if not speech_service:
        raise HTTPException(status_code=503, detail="서비스가 초기화되지 않았습니다")
    
    try:
        speech_service.stop_realtime_analysis()
        
        return fast_json_response({
            "success": True,
            "message": "실시간 음성 분석이 중지되었습니다",
            "timestamp": time.time()
        })
        
    except Exception as e:
        logger.error(f"실시간 분석 중지 오류: {e}")
        raise HTTPException(status_code=500, detail=f"분석 중지 실패: {str(e)}")


@app.get("/speech/status")
async def get_speech_status():
    """현재 음성 분석 상태 조회"""
    global speech_service
    
    if not speech_service:
        raise HTTPException(status_code=503, detail="서비스가 초기화되지 않았습니다")
    
    try:
        current_state = speech_service.get_current_state()
        
        return fast_json_response({
            "success": True,
            "status": current_state,
            "timestamp": time.time()
        })
        
    except Exception as e:
        logger.error(f"상태 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"상태 조회 실패: {str(e)}")


@app.get("/speech/results/latest")
async def get_latest_results(count: int = 10):
    """최신 분석 결과들 반환"""
    global speech_service
    
    if not speech_service:
        raise HTTPException(status_code=503, detail="서비스가 초기화되지 않았습니다")
    
    try:
        # 대기 중인 모든 결과 가져오기
        results = speech_service.get_all_pending_results()
        
        # 최신 count개만 반환
        latest_results = results[-count:] if len(results) > count else results
        
        # 직렬화 가능한 형태로 변환
        serialized_results = []
        for result in latest_results:
            serialized_result = {
                'timestamp': result.timestamp,
                'overall_score': result.overall_score,
                'audio_info': result.audio_info,
                'processing_time': result.processing_time,
                'is_complete': result.is_complete
            }
            
            # 각 분석 결과 추가 (None 체크)
            if result.speech_features:
                serialized_result['speech_features'] = {
                    'speaking_rate': result.speech_features.speaking_rate,
                    'volume_level': result.speech_features.volume_level,
                    'clarity_score': result.speech_features.clarity_score,
                    'interview_score': result.speech_features.interview_score
                }
            
            if result.transcription:
                serialized_result['transcription'] = {
                    'text': result.transcription.text,
                    'language': result.transcription.language,
                    'confidence': result.transcription.confidence
                }
            
            if result.emotion:
                serialized_result['emotion'] = {
                    'primary_emotion': result.emotion.primary_emotion,
                    'confidence_level': result.emotion.confidence_level,
                    'nervousness_level': result.emotion.nervousness_level
                }
            
            if result.interview_feedback:
                serialized_result['interview_feedback'] = result.interview_feedback
            
            serialized_results.append(serialized_result)
        
        return fast_json_response({
            "success": True,
            "results": serialized_results,
            "count": len(serialized_results),
            "timestamp": time.time()
        })
        
    except Exception as e:
        logger.error(f"결과 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"결과 조회 실패: {str(e)}")


@app.get("/speech/summary")
async def get_session_summary(duration: float = 60.0):
    """세션 요약 정보 반환"""
    global speech_service
    
    if not speech_service:
        raise HTTPException(status_code=503, detail="서비스가 초기화되지 않았습니다")
    
    try:
        summary = speech_service.get_session_summary(duration)
        
        return fast_json_response({
            "success": True,
            "summary": summary,
            "timestamp": time.time()
        })
        
    except Exception as e:
        logger.error(f"요약 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"요약 조회 실패: {str(e)}")


@app.get("/speech/stream")
async def stream_results(request: Request):
    """실시간 분석 결과 스트리밍 (Server-Sent Events)"""
    global client_counter, streaming_callback
    
    # 클라이언트 ID 생성
    client_counter += 1
    client_id = f"client_{client_counter}"
    
    async def event_generator():
        # 클라이언트 등록
        streaming_callback.add_client(client_id)
        
        try:
            while True:
                # 클라이언트 연결 확인
                if await request.is_disconnected():
                    break
                
                # 최신 결과들 가져오기
                latest_results = streaming_callback.get_latest_results(5)
                
                if latest_results:
                    # SSE 형식으로 전송
                    for result in latest_results:
                        data = json.dumps(result, ensure_ascii=False)
                        yield f"data: {data}\n\n"
                
                # 1초 대기
                await asyncio.sleep(1.0)
                
        except Exception as e:
            logger.error(f"스트리밍 오류: {e}")
        finally:
            # 클라이언트 제거
            streaming_callback.remove_client(client_id)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )


@app.get("/stats")
async def get_service_stats():
    """서비스 통계 정보 반환"""
    global speech_service
    
    try:
        base_stats = {
            "service_initialized": speech_service is not None,
            "streaming_clients": len(streaming_callback.connected_clients),
            "timestamp": time.time()
        }
        
        if speech_service:
            comprehensive_stats = speech_service.get_comprehensive_stats()
            base_stats.update(comprehensive_stats)
        
        return fast_json_response({
            "success": True,
            "stats": base_stats
        })
        
    except Exception as e:
        logger.error(f"통계 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")


@app.post("/config/update")
async def update_service_config(config_updates: Dict[str, Any]):
    """서비스 설정 업데이트"""
    try:
        # 설정 업데이트
        update_config(**config_updates)
        
        # 캐시 무효화
        get_cached_config_info.cache_clear()
        
        return fast_json_response({
            "success": True,
            "message": "설정이 업데이트되었습니다",
            "updated_keys": list(config_updates.keys()),
            "timestamp": time.time()
        })
        
    except Exception as e:
        logger.error(f"설정 업데이트 오류: {e}")
        raise HTTPException(status_code=500, detail=f"설정 업데이트 실패: {str(e)}")


@app.post("/system/gc")
async def force_garbage_collection():
    """강제 가비지 컬렉션"""
    try:
        # 가비지 컬렉션 실행
        collected = gc.collect()
        
        return fast_json_response({
            "success": True,
            "message": f"가비지 컬렉션 완료: {collected}개 객체 정리됨",
            "collected_objects": collected,
            "timestamp": time.time()
        })
        
    except Exception as e:
        logger.error(f"가비지 컬렉션 오류: {e}")
        raise HTTPException(status_code=500, detail=f"가비지 컬렉션 실패: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """전역 예외 처리"""
    logger.error(f"전역 예외 발생: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "내부 서버 오류가 발생했습니다",
            "detail": str(exc) if app.debug else "서버 관리자에게 문의하세요",
            "timestamp": time.time()
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # 서버 실행
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=15013,  # v3-speech 전용 포트
        reload=True,
        log_level="info",
        reload_delay=1.0,
        use_colors=True,
        access_log=True
    )
