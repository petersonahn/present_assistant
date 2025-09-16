#!/usr/bin/env python3
"""
v3-speech 서비스 간단 테스트
의존성 없이 기본 Python만 사용
"""

import json
import urllib.request
import urllib.error

BASE_URL = "http://localhost:15013"

def test_api(endpoint, description):
    """API 엔드포인트 테스트"""
    try:
        url = f"{BASE_URL}{endpoint}"
        print(f"🔍 {description}")
        print(f"   URL: {url}")
        
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            print(f"   ✅ 성공: {response.status}")
            
            # 중요한 정보만 출력
            if endpoint == "/health":
                print(f"   📊 상태: {data.get('status')}")
                components = data.get('components', {})
                success_count = sum(1 for v in components.values() if v)
                print(f"   🔧 컴포넌트: {success_count}/{len(components)} 성공")
            
            elif endpoint == "/audio/devices":
                device_count = data.get('total_count', 0)
                print(f"   🎤 오디오 디바이스: {device_count}개")
            
            elif endpoint == "/speech/status":
                status = data.get('status', {})
                print(f"   🗣️  실행 중: {status.get('is_running', False)}")
                print(f"   📈 세션 시간: {status.get('session_duration', 0):.1f}초")
            
            return True
            
    except urllib.error.URLError as e:
        print(f"   ❌ 연결 실패: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON 파싱 실패: {e}")
        return False
    except Exception as e:
        print(f"   ❌ 오류: {e}")
        return False

def main():
    print("🎤 v3-speech 서비스 간단 테스트")
    print("=" * 50)
    
    tests = [
        ("/health", "헬스체크"),
        ("/audio/devices", "오디오 디바이스 목록"),
        ("/speech/status", "음성 분석 상태"),
        ("/stats", "서비스 통계"),
        ("/config", "설정 정보")
    ]
    
    success_count = 0
    
    for endpoint, description in tests:
        if test_api(endpoint, description):
            success_count += 1
        print()
    
    print("=" * 50)
    print(f"🎯 테스트 결과: {success_count}/{len(tests)} 성공")
    
    if success_count == len(tests):
        print("✅ 모든 테스트 통과! v3-speech 서비스가 정상 작동합니다.")
        print("\n🚀 사용 가능한 API:")
        print(f"   - API 문서: {BASE_URL}/docs")
        print(f"   - 실시간 스트리밍: {BASE_URL}/speech/stream")
        print(f"   - 웹 인터페이스: {BASE_URL}/")
    else:
        print("⚠️  일부 테스트 실패. 서비스 상태를 확인하세요.")

if __name__ == "__main__":
    main()
