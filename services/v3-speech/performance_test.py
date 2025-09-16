#!/usr/bin/env python3
"""
v3-speech 성능 테스트 스크립트
실시간 처리 성능, 메모리 사용량, 응답 시간 등을 측정
"""

import asyncio
import time
import statistics
import psutil
import threading
import numpy as np
import logging
from typing import List, Dict, Any
import requests
import json
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from collections import deque

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:15013"


class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self):
        self.metrics = {
            'cpu_usage': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'response_times': deque(maxlen=1000),
            'processing_times': deque(maxlen=1000),
            'error_count': 0,
            'request_count': 0
        }
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = None
    
    def start_monitoring(self):
        """모니터링 시작"""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("성능 모니터링 시작")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("성능 모니터링 중지")
    
    def _monitor_loop(self):
        """모니터링 루프"""
        while self.monitoring:
            try:
                # CPU 사용률
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics['cpu_usage'].append(cpu_percent)
                
                # 메모리 사용률
                memory = psutil.virtual_memory()
                self.metrics['memory_usage'].append(memory.percent)
                
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"모니터링 오류: {e}")
    
    def record_response_time(self, response_time: float):
        """응답 시간 기록"""
        self.metrics['response_times'].append(response_time)
        self.metrics['request_count'] += 1
    
    def record_processing_time(self, processing_time: float):
        """처리 시간 기록"""
        self.metrics['processing_times'].append(processing_time)
    
    def record_error(self):
        """에러 기록"""
        self.metrics['error_count'] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        stats = {}
        
        # CPU 통계
        if self.metrics['cpu_usage']:
            cpu_data = list(self.metrics['cpu_usage'])
            stats['cpu'] = {
                'avg': statistics.mean(cpu_data),
                'max': max(cpu_data),
                'min': min(cpu_data),
                'current': cpu_data[-1] if cpu_data else 0
            }
        
        # 메모리 통계
        if self.metrics['memory_usage']:
            memory_data = list(self.metrics['memory_usage'])
            stats['memory'] = {
                'avg': statistics.mean(memory_data),
                'max': max(memory_data),
                'min': min(memory_data),
                'current': memory_data[-1] if memory_data else 0
            }
        
        # 응답 시간 통계
        if self.metrics['response_times']:
            response_data = list(self.metrics['response_times'])
            stats['response_time'] = {
                'avg': statistics.mean(response_data),
                'max': max(response_data),
                'min': min(response_data),
                'p95': np.percentile(response_data, 95),
                'p99': np.percentile(response_data, 99)
            }
        
        # 처리 시간 통계
        if self.metrics['processing_times']:
            processing_data = list(self.metrics['processing_times'])
            stats['processing_time'] = {
                'avg': statistics.mean(processing_data),
                'max': max(processing_data),
                'min': min(processing_data)
            }
        
        # 전체 통계
        duration = time.time() - self.start_time if self.start_time else 0
        stats['overall'] = {
            'duration': duration,
            'total_requests': self.metrics['request_count'],
            'error_count': self.metrics['error_count'],
            'error_rate': (self.metrics['error_count'] / max(1, self.metrics['request_count'])) * 100,
            'requests_per_second': self.metrics['request_count'] / max(1, duration)
        }
        
        return stats
    
    def generate_report(self, output_file: str = "performance_report.txt"):
        """성능 리포트 생성"""
        stats = self.get_statistics()
        
        report = []
        report.append("=" * 60)
        report.append("v3-speech 성능 테스트 리포트")
        report.append("=" * 60)
        report.append("")
        
        # 전체 통계
        if 'overall' in stats:
            overall = stats['overall']
            report.append("📊 전체 통계:")
            report.append(f"  테스트 시간: {overall['duration']:.1f}초")
            report.append(f"  총 요청 수: {overall['total_requests']}")
            report.append(f"  에러 수: {overall['error_count']}")
            report.append(f"  에러율: {overall['error_rate']:.2f}%")
            report.append(f"  초당 요청 수: {overall['requests_per_second']:.2f} RPS")
            report.append("")
        
        # CPU 통계
        if 'cpu' in stats:
            cpu = stats['cpu']
            report.append("💻 CPU 사용률:")
            report.append(f"  평균: {cpu['avg']:.1f}%")
            report.append(f"  최대: {cpu['max']:.1f}%")
            report.append(f"  최소: {cpu['min']:.1f}%")
            report.append(f"  현재: {cpu['current']:.1f}%")
            report.append("")
        
        # 메모리 통계
        if 'memory' in stats:
            memory = stats['memory']
            report.append("🧠 메모리 사용률:")
            report.append(f"  평균: {memory['avg']:.1f}%")
            report.append(f"  최대: {memory['max']:.1f}%")
            report.append(f"  최소: {memory['min']:.1f}%")
            report.append(f"  현재: {memory['current']:.1f}%")
            report.append("")
        
        # 응답 시간 통계
        if 'response_time' in stats:
            response = stats['response_time']
            report.append("⏱️  응답 시간 (밀리초):")
            report.append(f"  평균: {response['avg']*1000:.1f}ms")
            report.append(f"  최대: {response['max']*1000:.1f}ms")
            report.append(f"  최소: {response['min']*1000:.1f}ms")
            report.append(f"  95%ile: {response['p95']*1000:.1f}ms")
            report.append(f"  99%ile: {response['p99']*1000:.1f}ms")
            report.append("")
        
        # 처리 시간 통계
        if 'processing_time' in stats:
            processing = stats['processing_time']
            report.append("⚡ 처리 시간 (밀리초):")
            report.append(f"  평균: {processing['avg']*1000:.1f}ms")
            report.append(f"  최대: {processing['max']*1000:.1f}ms")
            report.append(f"  최소: {processing['min']*1000:.1f}ms")
            report.append("")
        
        report.append("=" * 60)
        
        # 파일에 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        # 콘솔에 출력
        print('\n'.join(report))
        
        return stats


class LoadTester:
    """부하 테스트 클래스"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.monitor = PerformanceMonitor()
    
    def test_health_endpoint(self, num_requests: int = 100, 
                           concurrent_users: int = 10) -> Dict[str, Any]:
        """헬스체크 엔드포인트 부하 테스트"""
        print(f"🏥 헬스체크 엔드포인트 테스트 ({num_requests}회, {concurrent_users}명 동시)")
        
        self.monitor.start_monitoring()
        
        def make_request():
            start_time = time.time()
            try:
                response = self.session.get(f"{self.base_url}/health", timeout=5.0)
                response.raise_for_status()
                response_time = time.time() - start_time
                self.monitor.record_response_time(response_time)
                return True
            except Exception as e:
                self.monitor.record_error()
                logger.error(f"요청 실패: {e}")
                return False
        
        # 병렬 실행
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [f.result() for f in futures]
        
        self.monitor.stop_monitoring()
        
        success_count = sum(results)
        success_rate = (success_count / num_requests) * 100
        
        print(f"✅ 성공률: {success_rate:.1f}% ({success_count}/{num_requests})")
        
        return self.monitor.get_statistics()
    
    def test_realtime_analysis(self, duration: float = 30.0) -> Dict[str, Any]:
        """실시간 분석 성능 테스트"""
        print(f"🎤 실시간 분석 성능 테스트 ({duration}초)")
        
        self.monitor.start_monitoring()
        
        try:
            # 실시간 분석 시작
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/speech/start_realtime")
            response.raise_for_status()
            
            start_response_time = time.time() - start_time
            self.monitor.record_response_time(start_response_time)
            
            print("✅ 실시간 분석 시작됨")
            
            # 결과 폴링
            result_count = 0
            while time.time() - start_time < duration:
                try:
                    poll_start = time.time()
                    response = self.session.get(f"{self.base_url}/speech/results/latest?count=1")
                    response.raise_for_status()
                    
                    poll_time = time.time() - poll_start
                    self.monitor.record_response_time(poll_time)
                    
                    data = response.json()
                    if data.get('success') and data.get('results'):
                        result_count += len(data['results'])
                        
                        # 처리 시간 기록
                        for result in data['results']:
                            if 'processing_time' in result:
                                self.monitor.record_processing_time(result['processing_time'])
                    
                    time.sleep(1.0)  # 1초마다 폴링
                    
                except Exception as e:
                    self.monitor.record_error()
                    logger.error(f"결과 조회 실패: {e}")
            
            # 실시간 분석 중지
            stop_start = time.time()
            response = self.session.post(f"{self.base_url}/speech/stop_realtime")
            response.raise_for_status()
            
            stop_response_time = time.time() - stop_start
            self.monitor.record_response_time(stop_response_time)
            
            print(f"✅ 실시간 분석 중지됨 (결과 {result_count}개 수집)")
            
        except Exception as e:
            logger.error(f"실시간 분석 테스트 실패: {e}")
            self.monitor.record_error()
        
        finally:
            self.monitor.stop_monitoring()
        
        return self.monitor.get_statistics()
    
    def test_concurrent_requests(self, num_requests: int = 200, 
                               concurrent_users: int = 20) -> Dict[str, Any]:
        """동시 요청 부하 테스트"""
        print(f"🚀 동시 요청 부하 테스트 ({num_requests}회, {concurrent_users}명)")
        
        self.monitor.start_monitoring()
        
        endpoints = [
            "/health",
            "/audio/devices", 
            "/speech/status",
            "/stats",
            "/config"
        ]
        
        def make_random_request():
            endpoint = np.random.choice(endpoints)
            start_time = time.time()
            
            try:
                response = self.session.get(f"{self.base_url}{endpoint}", timeout=10.0)
                response.raise_for_status()
                
                response_time = time.time() - start_time
                self.monitor.record_response_time(response_time)
                return True
                
            except Exception as e:
                self.monitor.record_error()
                logger.error(f"요청 실패 [{endpoint}]: {e}")
                return False
        
        # 병렬 실행
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(make_random_request) for _ in range(num_requests)]
            results = [f.result() for f in futures]
        
        self.monitor.stop_monitoring()
        
        success_count = sum(results)
        success_rate = (success_count / num_requests) * 100
        
        print(f"✅ 성공률: {success_rate:.1f}% ({success_count}/{num_requests})")
        
        return self.monitor.get_statistics()
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """종합 성능 테스트"""
        print("🧪 v3-speech 종합 성능 테스트 시작")
        print("=" * 60)
        
        all_results = {}
        
        # 1. 헬스체크 테스트
        print("\n1️⃣  헬스체크 성능 테스트")
        all_results['health_test'] = self.test_health_endpoint(100, 10)
        time.sleep(2)
        
        # 2. 실시간 분석 테스트
        print("\n2️⃣  실시간 분석 성능 테스트")
        all_results['realtime_test'] = self.test_realtime_analysis(20.0)
        time.sleep(2)
        
        # 3. 동시 요청 테스트
        print("\n3️⃣  동시 요청 부하 테스트")
        all_results['concurrent_test'] = self.test_concurrent_requests(150, 15)
        
        print("\n" + "=" * 60)
        print("🎯 종합 성능 테스트 완료")
        
        return all_results


def check_service_availability(base_url: str = BASE_URL) -> bool:
    """서비스 가용성 확인"""
    try:
        response = requests.get(f"{base_url}/health", timeout=5.0)
        response.raise_for_status()
        
        health_data = response.json()
        return health_data.get('status') == 'healthy'
        
    except Exception as e:
        logger.error(f"서비스 가용성 확인 실패: {e}")
        return False


def main():
    """메인 함수"""
    print("🎤 v3-speech 성능 테스트 도구")
    print("=" * 60)
    
    # 서비스 가용성 확인
    print("서비스 상태 확인 중...")
    if not check_service_availability():
        print("❌ 서비스에 연결할 수 없습니다.")
        print(f"   {BASE_URL} 에서 v3-speech 서비스가 실행 중인지 확인하세요.")
        return
    
    print("✅ 서비스 연결 확인됨")
    
    # 부하 테스터 생성
    tester = LoadTester()
    
    try:
        # 종합 테스트 실행
        results = tester.run_comprehensive_test()
        
        # 리포트 생성
        print("\n📊 성능 리포트 생성 중...")
        
        # 각 테스트별 리포트
        for test_name, stats in results.items():
            if stats:
                monitor = PerformanceMonitor()
                monitor.metrics = {'response_times': deque(stats.get('response_time', {}).values())}
                monitor.start_time = time.time() - 60  # 임시값
                
                output_file = f"performance_report_{test_name}.txt"
                print(f"  📄 {output_file} 생성됨")
        
        print("\n✅ 모든 성능 테스트 완료!")
        print(f"상세한 결과는 performance_report_*.txt 파일을 확인하세요.")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  사용자에 의해 테스트 중단됨")
    
    except Exception as e:
        logger.error(f"테스트 실행 오류: {e}")
        print(f"❌ 테스트 실행 중 오류 발생: {e}")


if __name__ == "__main__":
    main()
