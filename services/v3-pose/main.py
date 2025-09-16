"""
실시간 면접 피드백 시스템 - FastAPI 서버
포즈 감지 API 엔드포인트
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Dict, List
import logging
import orjson  # 더 빠른 JSON 라이브러리
from functools import lru_cache
import os
from pose_estimator import create_pose_estimator, HumanPoseEstimator
from emotion_analyzer import EmotionAnalyzer

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="실시간 면접 피드백 시스템",
    description="AI 기반 감정+포즈 동시 분석 API",
    version="2.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 템플릿과 정적 파일 설정 (Docker 환경 고려)
import os
if os.path.exists("../../frontend/public"):
    # 로컬 개발 환경
    templates = Jinja2Templates(directory="../../frontend/public")
    app.mount("/static", StaticFiles(directory="../../frontend/static"), name="static")
else:
    # Docker 환경 - 템플릿 비활성화 (API만 제공)
    templates = None

# 전역 포즈 추정기 인스턴스
pose_estimator: HumanPoseEstimator = None
emotion_analyzer: EmotionAnalyzer = None

# 최적화된 JSON 응답 함수
def fast_json_response(content: Dict, status_code: int = 200) -> JSONResponse:
    """orjson을 사용한 빠른 JSON 응답"""
    return JSONResponse(
        content=orjson.loads(orjson.dumps(content)),
        status_code=status_code
    )

# 이미지 처리 결과 캐시 (메모리 절약을 위해 작은 크기)
@lru_cache(maxsize=10)
def get_cached_keypoint_info():
    """키포인트 정보 캐싱"""
    if pose_estimator is None:
        return None
    return {
        "keypoint_names": pose_estimator.KEYPOINT_NAMES,
        "pose_pairs": pose_estimator.POSE_PAIRS,
        "total_keypoints": len(pose_estimator.KEYPOINT_NAMES)
    }

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    global pose_estimator, emotion_analyzer
    try:
        logger.info("포즈 추정 모델 로딩 중...")
        pose_estimator = create_pose_estimator()
        logger.info("포즈 모델 로딩 완료!")
        emotion_analyzer = EmotionAnalyzer()
        logger.info("감정 분석기 로딩 완료!")
    except Exception as e:
        logger.error(f"모델 로딩 실패: {e}")
        raise e

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """메인 웹 페이지"""
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        # Docker 환경에서는 API 정보만 제공
        return HTMLResponse("""
        <html>
            <head><title>v3-pose API Service</title></head>
            <body>
                <h1>🎯 v3-pose API Service</h1>
                <p>포즈 분석 API 서비스가 실행 중입니다.</p>
                <ul>
                    <li><a href="/docs">API 문서</a></li>
                    <li><a href="/health">헬스체크</a></li>
                    <li><a href="/api">API 정보</a></li>
                </ul>
                <p>프론트엔드는 별도 서비스에서 제공됩니다: <a href="http://localhost:3000">http://localhost:3000</a></p>
            </body>
        </html>
        """)

@app.get("/api")
async def api_info():
    """API 정보 엔드포인트"""
    return {
        "message": "실시간 면접 피드백 시스템 API",
        "version": "1.0.0",
        "endpoints": {
            "/pose/analyze": "이미지 포즈 분석",
            "/pose/analyze_base64": "Base64 이미지 포즈 분석",
            "/health": "헬스체크"
        }
    }

@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    return {
        "status": "healthy",
        "model_loaded": pose_estimator is not None
    }

def image_to_cv2(image_data: bytes) -> np.ndarray:
    """바이트 데이터를 OpenCV 이미지로 변환"""
    image = Image.open(io.BytesIO(image_data))
    # PIL Image를 numpy array로 변환
    image_array = np.array(image)
    
    # RGB를 BGR로 변환 (OpenCV 형식)
    if len(image_array.shape) == 3:
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image_array
    
    return image_bgr

def cv2_to_base64(image: np.ndarray) -> str:
    """OpenCV 이미지를 Base64 문자열로 변환 (최적화됨)"""
    # JPEG 압축 품질 조정으로 파일 크기 줄이기
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, 85]  # 85% 품질 (기본 95%보다 빠름)
    _, buffer = cv2.imencode('.jpg', image, encode_param)
    # tobytes()가 tostring()보다 빠름
    image_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
    return image_base64

@app.post("/analyze")
async def analyze_all(data: Dict):
    """
    Base64 이미지에서 감정+포즈 동시 분석
    Args: {"image": "base64_encoded_image"}
    Returns: 감정+포즈 분석 결과
    """
    if pose_estimator is None or emotion_analyzer is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")
    try:
        if "image" not in data:
            raise HTTPException(status_code=400, detail="'image' 필드가 필요합니다")
        image_base64 = data["image"]
        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",")[1]
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_np = np.array(image)
        # 감정 분석
        emotion_result = emotion_analyzer.analyze(image_np)
        # 포즈 분석
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        pose_result = pose_estimator.estimate_pose(image_bgr)
        # (선택) 포즈 시각화 이미지
        include_result_image = data.get("include_result_image", False)
        result_image_base64 = None
        if include_result_image:
            pose_image = pose_estimator.draw_pose(image_bgr, pose_result['keypoints'])
            _, buffer = cv2.imencode('.jpg', pose_image)
            result_image_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer.tobytes()).decode('utf-8')}"
        return fast_json_response({
            "success": True,
            "emotion": emotion_result,
            "pose": pose_result,
            "result_image": result_image_base64
        })
    except Exception as e:
        logger.error(f"감정+포즈 분석 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"감정+포즈 분석 실패: {str(e)}")

@app.post("/pose/analyze")
async def analyze_pose(file: UploadFile = File(...)):
    """
    업로드된 이미지에서 포즈 분석
    
    Args:
        file: 업로드된 이미지 파일
        
    Returns:
        포즈 분석 결과
    """
    if pose_estimator is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")
    
    try:
        # 파일 읽기
        image_data = await file.read()
        
        # 이미지 형식 확인
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")
        
        # OpenCV 이미지로 변환
        image = image_to_cv2(image_data)
        
        # 포즈 추정 수행
        result = pose_estimator.estimate_pose(image)
        
        # 포즈가 그려진 이미지 생성
        pose_image = pose_estimator.draw_pose(image, result['keypoints'])
        
        # 결과 이미지를 Base64로 인코딩
        result_image_base64 = cv2_to_base64(pose_image)
        
        return fast_json_response({
            "success": True,
            "data": {
                "keypoints": result['keypoints'],
                "analysis": result['analysis'],
                "image_shape": result['image_shape'],
                "result_image": f"data:image/jpeg;base64,{result_image_base64}",
                "keypoint_count": len(result['keypoints'])
            }
        })
        
    except Exception as e:
        logger.error(f"포즈 분석 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"포즈 분석 실패: {str(e)}")

@app.post("/pose/analyze_base64")
async def analyze_pose_base64(data: Dict):
    """
    Base64 이미지에서 포즈 분석
    
    Args:
        data: {"image": "base64_encoded_image"}
        
    Returns:
        포즈 분석 결과
    """
    if pose_estimator is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")
    
    try:
        if "image" not in data:
            raise HTTPException(status_code=400, detail="'image' 필드가 필요합니다")
        
        # Base64 디코딩
        image_base64 = data["image"]
        if image_base64.startswith("data:image"):
            # data:image/jpeg;base64, 부분 제거
            image_base64 = image_base64.split(",")[1]
        
        image_data = base64.b64decode(image_base64)
        
        # OpenCV 이미지로 변환
        image = image_to_cv2(image_data)
        
        # 포즈 추정 수행
        result = pose_estimator.estimate_pose(image)
        
        # 포즈가 그려진 이미지 생성 (선택사항)
        include_result_image = data.get("include_result_image", False)
        response_data = {
            "keypoints": result['keypoints'],
            "analysis": result['analysis'],
            "image_shape": result['image_shape'],
            "keypoint_count": len(result['keypoints'])
        }
        
        if include_result_image:
            pose_image = pose_estimator.draw_pose(image, result['keypoints'])
            result_image_base64 = cv2_to_base64(pose_image)
            response_data["result_image"] = f"data:image/jpeg;base64,{result_image_base64}"
        
        return fast_json_response({
            "success": True,
            "data": response_data
        })
        
    except Exception as e:
        logger.error(f"포즈 분석 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"포즈 분석 실패: {str(e)}")

@app.get("/pose/keypoints")
async def get_keypoint_info():
    """키포인트 정보 반환"""
    if pose_estimator is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")
    
    cached_info = get_cached_keypoint_info()
    if cached_info is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")
    
    return fast_json_response(cached_info)

@app.post("/pose/feedback")
async def get_pose_feedback(keypoints: List[Dict]):
    """
    키포인트 데이터로부터 포즈 피드백 생성
    
    Args:
        keypoints: 키포인트 리스트
        
    Returns:
        포즈 분석 및 피드백
    """
    if pose_estimator is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")
    
    try:
        analysis = pose_estimator.analyze_pose(keypoints)
        
        return fast_json_response({
            "success": True,
            "analysis": analysis
        })
        
    except Exception as e:
        logger.error(f"피드백 생성 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"피드백 생성 실패: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # 서버 실행
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=15012,
        reload=True,
        log_level="info",
        reload_delay=1.0,  # 재시작 지연시간 추가
        use_colors=True
    )
