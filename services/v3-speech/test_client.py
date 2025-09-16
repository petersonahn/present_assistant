#!/usr/bin/env python3
"""
v3-speech 서비스 테스트 클라이언트
실시간 음성 분석 API 테스트 및 데모
"""

import requests
import json
import time
import asyncio
import aiohttp
import threading
from typing import Dict, Any
import argparse
import sys
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:15013"


class SpeechServiceClient:
    """v3-speech 서비스 클라이언트"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.is_streaming = False
        self.streaming_thread = None
    
    def health_check(self) -> Dict[str, Any]:
        """헬스체크"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"헬스체크 실패: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_audio_devices(self) -> Dict[str, Any]:
        """오디오 디바이스 목록 조회"""
        try:
            response = self.session.get(f"{self.base_url}/audio/devices")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"디바이스 조회 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def test_device(self, device_id: int) -> Dict[str, Any]:
        """특정 디바이스 테스트"""
        try:
            response = self.session.post(f"{self.base_url}/audio/test_device/{device_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"디바이스 테스트 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def start_realtime_analysis(self, device_id: int = None) -> Dict[str, Any]:
        """실시간 분석 시작"""
        try:
            data = {}
            if device_id is not None:
                data["device_id"] = device_id
            
            response = self.session.post(
                f"{self.base_url}/speech/start_realtime",
                json=data if data else None
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"실시간 분석 시작 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def stop_realtime_analysis(self) -> Dict[str, Any]:
        """실시간 분석 중지"""
        try:
            response = self.session.post(f"{self.base_url}/speech/stop_realtime")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"실시간 분석 중지 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """현재 상태 조회"""
        try:
            response = self.session.get(f"{self.base_url}/speech/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"상태 조회 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def get_latest_results(self, count: int = 5) -> Dict[str, Any]:
        """최신 결과 조회"""
        try:
            response = self.session.get(
                f"{self.base_url}/speech/results/latest?count={count}"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"결과 조회 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def get_summary(self, duration: float = 30.0) -> Dict[str, Any]:
        """세션 요약 조회"""
        try:
            response = self.session.get(
                f"{self.base_url}/speech/summary?duration={duration}"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"요약 조회 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """서비스 통계 조회"""
        try:
            response = self.session.get(f"{self.base_url}/stats")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def start_streaming(self, callback=None):
        """실시간 스트리밍 시작"""
        if self.is_streaming:
            logger.warning("이미 스트리밍 중입니다")
            return
        
        self.is_streaming = True
        self.streaming_thread = threading.Thread(
            target=self._streaming_worker,
            args=(callback,),
            daemon=True
        )
        self.streaming_thread.start()
        logger.info("실시간 스트리밍 시작")
    
    def stop_streaming(self):
        """실시간 스트리밍 중지"""
        self.is_streaming = False
        if self.streaming_thread:
            self.streaming_thread.join(timeout=2.0)
        logger.info("실시간 스트리밍 중지")
    
    def _streaming_worker(self, callback):
        """스트리밍 워커 스레드"""
        try:
            response = self.session.get(
                f"{self.base_url}/speech/stream",
                stream=True,
                headers={"Accept": "text/event-stream"}
            )
            
            for line in response.iter_lines():
                if not self.is_streaming:
                    break
                
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])  # 'data: ' 제거
                            if callback:
                                callback(data)
                            else:
                                self._default_stream_callback(data)
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON 파싱 오류: {e}")
                        except Exception as e:
                            logger.error(f"스트리밍 콜백 오류: {e}")
        
        except Exception as e:
            logger.error(f"스트리밍 오류: {e}")
    
    def _default_stream_callback(self, data):
        """기본 스트리밍 콜백"""
        timestamp = data.get('timestamp', 0)
        score = data.get('overall_score', 0)
        
        print(f"[{time.strftime('%H:%M:%S', time.localtime(timestamp))}] "
              f"점수: {score:.1f}")
        
        # 음성 특징 출력
        if 'speech_features' in data:
            features = data['speech_features']
            print(f"  📊 말하기속도: {features.get('speaking_rate', 0):.1f} WPM, "
                  f"명료도: {features.get('clarity_score', 0):.2f}")
        
        # 감정 분석 출력
        if 'emotion' in data:
            emotion = data['emotion']
            print(f"  😊 감정: {emotion.get('primary_emotion', 'unknown')}, "
                  f"자신감: {emotion.get('confidence_level', 0):.2f}")
        
        # 음성 인식 출력
        if 'transcription' in data:
            transcription = data['transcription']
            text = transcription.get('text', '').strip()
            if text:
                print(f"  🗣️  \"{text}\"")
        
        print()


def test_basic_functionality():
    """기본 기능 테스트"""
    print("🧪 v3-speech 서비스 기본 기능 테스트")
    print("=" * 50)
    
    client = SpeechServiceClient()
    
    # 1. 헬스체크
    print("1. 헬스체크...")
    health = client.health_check()
    print(f"   상태: {health.get('status', 'unknown')}")
    
    if health.get('status') != 'healthy':
        print("   ⚠️  서비스가 정상 상태가 아닙니다")
        if 'components' in health:
            for component, status in health['components'].items():
                print(f"   - {component}: {'✅' if status else '❌'}")
        return False
    
    # 2. 오디오 디바이스 조회
    print("\n2. 오디오 디바이스 조회...")
    devices = client.get_audio_devices()
    if devices.get('success'):
        device_list = devices.get('devices', [])
        print(f"   감지된 디바이스: {len(device_list)}개")
        for device in device_list[:3]:  # 상위 3개만 출력
            print(f"   - [{device['id']}] {device['name']}")
    else:
        print("   ❌ 디바이스 조회 실패")
    
    # 3. 설정 정보 조회
    print("\n3. 서비스 통계...")
    stats = client.get_stats()
    if stats.get('success'):
        service_stats = stats.get('stats', {})
        print(f"   초기화됨: {'✅' if service_stats.get('service_initialized') else '❌'}")
        print(f"   스트리밍 클라이언트: {service_stats.get('streaming_clients', 0)}개")
    
    print("\n✅ 기본 기능 테스트 완료")
    return True


def test_realtime_analysis(duration: float = 10.0, device_id: int = None):
    """실시간 분석 테스트"""
    print(f"\n🎤 실시간 분석 테스트 ({duration}초)")
    print("=" * 50)
    
    client = SpeechServiceClient()
    
    try:
        # 실시간 분석 시작
        print("실시간 분석 시작...")
        result = client.start_realtime_analysis(device_id)
        if not result.get('success'):
            print(f"❌ 분석 시작 실패: {result.get('error')}")
            return False
        
        print("✅ 분석 시작됨")
        print("🗣️  마이크에 대고 말해보세요...")
        print()
        
        # 스트리밍 시작
        client.start_streaming()
        
        # 지정된 시간만큼 대기
        time.sleep(duration)
        
        # 스트리밍 중지
        client.stop_streaming()
        
        # 실시간 분석 중지
        print("\n실시간 분석 중지...")
        result = client.stop_realtime_analysis()
        if result.get('success'):
            print("✅ 분석 중지됨")
        
        # 최종 요약 조회
        print("\n📊 세션 요약:")
        summary = client.get_summary(duration)
        if summary.get('success'):
            summary_data = summary.get('summary', {})
            if summary_data:
                print(f"   총 샘플: {summary_data.get('total_samples', 0)}개")
                print(f"   평균 점수: {summary_data.get('average_overall_score', 0):.1f}")
                print(f"   평균 말하기 속도: {summary_data.get('average_speech_rate', 0):.1f} WPM")
                
                emotions = summary_data.get('dominant_emotions', {})
                if emotions:
                    dominant = max(emotions, key=emotions.get)
                    print(f"   주요 감정: {dominant} ({emotions[dominant]}회)")
            else:
                print("   분석할 데이터가 없습니다")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⏹️  사용자에 의해 중단됨")
        client.stop_streaming()
        client.stop_realtime_analysis()
        return True
    
    except Exception as e:
        print(f"\n❌ 테스트 오류: {e}")
        client.stop_streaming()
        client.stop_realtime_analysis()
        return False


def interactive_mode():
    """대화형 모드"""
    print("\n🎮 대화형 모드")
    print("=" * 50)
    print("명령어:")
    print("  start [device_id] - 실시간 분석 시작")
    print("  stop              - 실시간 분석 중지")
    print("  status            - 현재 상태 조회")
    print("  results [count]   - 최신 결과 조회")
    print("  summary [duration]- 세션 요약")
    print("  devices           - 오디오 디바이스 목록")
    print("  stream            - 스트리밍 시작/중지")
    print("  help              - 도움말")
    print("  quit              - 종료")
    print()
    
    client = SpeechServiceClient()
    streaming = False
    
    while True:
        try:
            command = input("v3-speech> ").strip().split()
            if not command:
                continue
            
            cmd = command[0].lower()
            
            if cmd == 'quit' or cmd == 'exit':
                if streaming:
                    client.stop_streaming()
                break
            
            elif cmd == 'help':
                print("사용 가능한 명령어: start, stop, status, results, summary, devices, stream, help, quit")
            
            elif cmd == 'start':
                device_id = int(command[1]) if len(command) > 1 else None
                result = client.start_realtime_analysis(device_id)
                print("✅ 시작됨" if result.get('success') else f"❌ 실패: {result.get('error')}")
            
            elif cmd == 'stop':
                result = client.stop_realtime_analysis()
                print("✅ 중지됨" if result.get('success') else f"❌ 실패: {result.get('error')}")
            
            elif cmd == 'status':
                result = client.get_status()
                if result.get('success'):
                    status = result.get('status', {})
                    print(f"실행 중: {'✅' if status.get('is_running') else '❌'}")
                    print(f"음성 감지: {'🗣️' if status.get('is_speaking') else '🤫'}")
                    print(f"세션 시간: {status.get('session_duration', 0):.1f}초")
                else:
                    print(f"❌ 조회 실패: {result.get('error')}")
            
            elif cmd == 'results':
                count = int(command[1]) if len(command) > 1 else 5
                result = client.get_latest_results(count)
                if result.get('success'):
                    results = result.get('results', [])
                    print(f"최신 {len(results)}개 결과:")
                    for i, res in enumerate(results[-3:]):  # 최신 3개만 출력
                        score = res.get('overall_score', 0)
                        timestamp = res.get('timestamp', 0)
                        time_str = time.strftime('%H:%M:%S', time.localtime(timestamp))
                        print(f"  {i+1}. [{time_str}] 점수: {score:.1f}")
                else:
                    print(f"❌ 조회 실패: {result.get('error')}")
            
            elif cmd == 'summary':
                duration = float(command[1]) if len(command) > 1 else 30.0
                result = client.get_summary(duration)
                if result.get('success'):
                    summary = result.get('summary', {})
                    if summary:
                        print(f"📊 최근 {duration}초 요약:")
                        print(f"  평균 점수: {summary.get('average_overall_score', 0):.1f}")
                        print(f"  총 샘플: {summary.get('total_samples', 0)}개")
                    else:
                        print("요약할 데이터가 없습니다")
                else:
                    print(f"❌ 조회 실패: {result.get('error')}")
            
            elif cmd == 'devices':
                result = client.get_audio_devices()
                if result.get('success'):
                    devices = result.get('devices', [])
                    print(f"🎤 오디오 디바이스 ({len(devices)}개):")
                    for device in devices:
                        print(f"  [{device['id']}] {device['name']}")
                else:
                    print(f"❌ 조회 실패: {result.get('error')}")
            
            elif cmd == 'stream':
                if streaming:
                    client.stop_streaming()
                    streaming = False
                    print("✅ 스트리밍 중지")
                else:
                    client.start_streaming()
                    streaming = True
                    print("✅ 스트리밍 시작 (실시간 결과를 출력합니다)")
            
            else:
                print(f"알 수 없는 명령어: {cmd}")
                print("'help'를 입력하여 사용 가능한 명령어를 확인하세요")
        
        except KeyboardInterrupt:
            print("\n종료합니다...")
            if streaming:
                client.stop_streaming()
            break
        
        except Exception as e:
            print(f"오류: {e}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="v3-speech 서비스 테스트 클라이언트")
    parser.add_argument("--url", default=BASE_URL, help="서비스 URL")
    parser.add_argument("--test", choices=["basic", "realtime", "interactive"], 
                       default="basic", help="테스트 모드")
    parser.add_argument("--duration", type=float, default=10.0, 
                       help="실시간 테스트 지속시간 (초)")
    parser.add_argument("--device", type=int, help="사용할 오디오 디바이스 ID")
    
    args = parser.parse_args()
    
    global BASE_URL
    BASE_URL = args.url
    
    print(f"🎤 v3-speech 테스트 클라이언트")
    print(f"서비스 URL: {args.url}")
    print()
    
    try:
        if args.test == "basic":
            success = test_basic_functionality()
            sys.exit(0 if success else 1)
        
        elif args.test == "realtime":
            if not test_basic_functionality():
                print("❌ 기본 기능 테스트 실패 - 실시간 테스트를 건너뜁니다")
                sys.exit(1)
            
            success = test_realtime_analysis(args.duration, args.device)
            sys.exit(0 if success else 1)
        
        elif args.test == "interactive":
            interactive_mode()
            sys.exit(0)
    
    except KeyboardInterrupt:
        print("\n\n👋 테스트 클라이언트 종료")
        sys.exit(0)


if __name__ == "__main__":
    main()
