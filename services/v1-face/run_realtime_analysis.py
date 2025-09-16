#!/usr/bin/env python3
"""
실시간 면접 표정 분석 실행 스크립트
웹캠을 사용하여 실시간으로 감정을 분석하고 결과를 저장합니다.
"""

import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from realtime_emotion_analyzer import RealtimeEmotionAnalyzer

def main():
    print("🎭 실시간 면접 표정 분석 시스템")
    print("="*50)
    print("📝 사용법:")
    print("   1. 웹캠이 연결되어 있는지 확인하세요")
    print("   2. 분석을 시작하면 실시간으로 감정이 분석됩니다")
    print("   3. 종료하려면 'q' 키를 누르거나 Ctrl+C를 누르세요")
    print("   4. 종료 후 results 폴더에서 상세한 분석 결과를 확인할 수 있습니다")
    print("="*50)
    
    try:
        # 분석기 초기화 및 실행
        analyzer = RealtimeEmotionAnalyzer()
        analyzer.run_realtime_analysis()
        
    except KeyboardInterrupt:
        print("\n⏹️ 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")
        print("💡 해결 방법:")
        print("   1. 웹캠이 제대로 연결되어 있는지 확인하세요")
        print("   2. 다른 프로그램에서 웹캠을 사용하고 있지 않은지 확인하세요")
        print("   3. 필요한 패키지가 설치되어 있는지 확인하세요")

if __name__ == "__main__":
    main()
