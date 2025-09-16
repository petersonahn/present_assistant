// 실시간 면접 피드백 시스템 JavaScript

class InterviewFeedbackSystem {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.isRunning = false;
        this.stream = null;
        this.analysisInterval = null;
        this.startTime = null;
        this.timerInterval = null;
        
        // 설정
        this.settings = {
            analysisInterval: 2000,
            showKeypoints: true,
            voiceFeedback: false,
            sensitivity: 5
        };
        
        this.initializeElements();
        this.bindEvents();
        this.initializeEmotionBars();
        this.checkServerConnection();
    }
    
    initializeElements() {
        this.elements = {
            startBtn: document.getElementById('start-btn'),
            stopBtn: document.getElementById('stop-btn'),
            captureBtn: document.getElementById('capture-btn'),
            settingsBtn: document.getElementById('settings-btn'),
            timerDisplay: document.getElementById('timer-display'),
            connectionStatus: document.getElementById('connection-status'),
            feedbackMessages: document.getElementById('feedback-messages'),
            
            // 분석 결과 요소들
            confidenceBar: document.getElementById('confidence-bar'),
            confidenceValue: document.getElementById('confidence-value'),
            focusBar: document.getElementById('focus-bar'),
            focusValue: document.getElementById('focus-value'),
            
            shoulderStatus: document.getElementById('shoulder-status'),
            headStatus: document.getElementById('head-status'),
            armStatus: document.getElementById('arm-status'),
            keypointCount: document.getElementById('keypoint-count'),
            
            scoreCircle: document.getElementById('score-circle'),
            scoreText: document.getElementById('score-text'),
            scoreFeedback: document.getElementById('score-feedback'),
            
            notification: document.getElementById('notification'),
            notificationText: document.getElementById('notification-text'),
            
            // 통계 요소들
            totalFrames: document.getElementById('total-frames'),
            goodPostureRate: document.getElementById('good-posture-rate'),
            averageScore: document.getElementById('average-score'),
            sessionTime: document.getElementById('session-time'),
            
            // 모달 요소들
            settingsModal: document.getElementById('settings-modal'),
            loading: document.getElementById('loading')
        };
    }
    
    bindEvents() {
        this.elements.startBtn.addEventListener('click', () => this.startAnalysis());
        this.elements.stopBtn.addEventListener('click', () => this.stopAnalysis());
        this.elements.captureBtn.addEventListener('click', () => this.captureImage());
        this.elements.settingsBtn.addEventListener('click', () => this.openSettings());
        
        // 모달 이벤트
        document.querySelector('.close-btn').addEventListener('click', () => this.closeModal());
        window.addEventListener('click', (e) => {
            if (e.target === this.elements.settingsModal) {
                this.closeModal();
            }
        });
        
        // 설정 변경 이벤트
        document.getElementById('sensitivity').addEventListener('input', (e) => {
            document.getElementById('sensitivity-value').textContent = e.target.value;
        });
    }
    
    initializeEmotionBars() {
        // 감정 분석 바를 기본값으로 초기화 (자세분석과 독립적)
        this.elements.confidenceBar.style.width = '50%';
        this.elements.confidenceValue.textContent = '50%';
        
        this.elements.focusBar.style.width = '50%';
        this.elements.focusValue.textContent = '50%';
        
        // 감정 분석 미구현 상태임을 표시
        console.log('감정 분석 기능은 별도 모듈에서 구현 예정입니다');
    }
    
    async checkServerConnection() {
        try {
            const response = await fetch('/api/v3-pose/health');
            const data = await response.json();
            
            if (data.status === 'healthy' && data.model_loaded) {
                this.updateConnectionStatus(true);
            } else {
                this.updateConnectionStatus(false, '모델이 로드되지 않았습니다');
            }
        } catch (error) {
            this.updateConnectionStatus(false, '서버에 연결할 수 없습니다');
            console.error('Connection check failed:', error);
        }
    }
    
    updateConnectionStatus(isOnline, message = '') {
        const status = this.elements.connectionStatus;
        if (isOnline) {
            status.className = 'status online';
            status.innerHTML = '<i class="fas fa-circle"></i> 연결됨';
        } else {
            status.className = 'status offline';
            status.innerHTML = `<i class="fas fa-circle"></i> ${message || '연결 끊김'}`;
        }
    }
    
    async startAnalysis() {
        try {
            this.showLoading(true);
            
            // 카메라 스트림 시작
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { 
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                },
                audio: false
            });
            
            this.video.srcObject = this.stream;
            await new Promise(resolve => {
                this.video.onloadedmetadata = resolve;
            });
            
            // 캔버스 크기 설정
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            
            this.isRunning = true;
            this.startTime = Date.now();
            
            // UI 업데이트
            this.elements.startBtn.disabled = true;
            this.elements.stopBtn.disabled = false;
            this.elements.captureBtn.disabled = false;
            
            // 타이머 시작
            this.startTimer();
            
            // 분석 시작
            this.startPoseAnalysis();
            
            this.showLoading(false);
            this.addFeedbackMessage('분석이 시작되었습니다!', 'success');
            
        } catch (error) {
            console.error('Failed to start analysis:', error);
            this.showLoading(false);
            this.addFeedbackMessage('카메라 접근에 실패했습니다', 'error');
            alert('카메라 접근 권한을 허용해주세요.');
        }
    }
    
    stopAnalysis() {
        this.isRunning = false;
        
        // 스트림 정지
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        // 인터벌 정리
        if (this.analysisInterval) {
            clearInterval(this.analysisInterval);
            this.analysisInterval = null;
        }
        
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
        
        // UI 리셋
        this.elements.startBtn.disabled = false;
        this.elements.stopBtn.disabled = true;
        this.elements.captureBtn.disabled = true;
        
        // 캔버스 클리어
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.addFeedbackMessage('분석이 중지되었습니다', 'warning');
    }
    
    startTimer() {
        this.timerInterval = setInterval(() => {
            if (this.startTime) {
                const elapsed = Date.now() - this.startTime;
                const minutes = Math.floor(elapsed / 60000);
                const seconds = Math.floor((elapsed % 60000) / 1000);
                this.elements.timerDisplay.textContent = 
                    `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            }
        }, 1000);
    }
    
    startPoseAnalysis() {
        this.analysisInterval = setInterval(async () => {
            if (!this.isRunning) return;
            
            try {
                await this.analyzePose();
            } catch (error) {
                console.error('Pose analysis failed:', error);
            }
        }, this.settings.analysisInterval);
    }
    
    async analyzePose() {
        // 비디오에서 프레임 캡처
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // 캔버스를 Base64로 변환
        const imageData = this.canvas.toDataURL('image/jpeg', 0.8);
        
        try {
            const response = await fetch('/api/v3-pose/pose/analyze_base64', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: imageData,
                    include_result_image: false
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.updateAnalysisResults(result.data);
                this.drawKeypoints(result.data.keypoints);
            } else {
                console.error('Analysis failed:', result);
            }
            
        } catch (error) {
            console.error('API request failed:', error);
            this.updateConnectionStatus(false, 'API 요청 실패');
        }
    }
    
    updateAnalysisResults(data) {
        const { keypoints, analysis } = data;
        
        // 키포인트 개수 업데이트
        this.elements.keypointCount.textContent = keypoints.length;
        
        // 자세 분석 업데이트
        this.updatePostureIndicators(analysis);
        
        // 점수 업데이트 (자세분석만)
        this.updatePostureScore(analysis.posture_score);
        
        // 피드백 메시지 업데이트
        this.updateFeedback(analysis.feedback);
        
        // 실시간 통계 업데이트
        this.updateRealTimeStats(analysis);
        
        // 알림 표시 (중요한 피드백만)
        if (analysis.feedback.length > 0 && analysis.posture_score < 50) {
            this.showNotification(analysis.feedback[0]);
        }
        
        // 키포인트 시각화
        this.visualizeKeypoints(keypoints);
        
        // 감정 분석은 별도로 처리 (향후 구현)
        // this.updateEmotionAnalysis();
    }
    
    updatePostureIndicators(analysis) {
        // 어깨 균형
        this.elements.shoulderStatus.textContent = this.getStatusText(analysis.shoulder_balance);
        this.elements.shoulderStatus.className = `indicator-value ${analysis.shoulder_balance}`;
        
        // 머리 위치
        this.elements.headStatus.textContent = this.getStatusText(analysis.head_position);
        this.elements.headStatus.className = `indicator-value ${analysis.head_position}`;
        
        // 팔 자세
        this.elements.armStatus.textContent = this.getStatusText(analysis.arm_position);
        this.elements.armStatus.className = `indicator-value ${analysis.arm_position}`;
    }
    
    getStatusText(status) {
        const statusMap = {
            'balanced': '균형잡힘 ✓',
            'unbalanced': '불균형 ⚠',
            'partial': '부분감지 ◐',
            'estimated': '추정됨 ⚡',
            'detected': '감지됨 🔍',
            'straight': '바른자세 ✓',
            'tilted': '기울어짐 ⚠',
            'natural': '자연스러움 ✓',
            'raised': '부자연스러움 ⚠',
            'unknown': '감지중...'
        };
        return statusMap[status] || '감지중...';
    }
    
    updatePostureScore(score) {
        // 자세 점수만 업데이트 (원형 차트)
        this.elements.scoreCircle.style.strokeDasharray = `${score}, 100`;
        this.elements.scoreText.textContent = score;
        
        // 점수에 따른 색상 변경
        if (score >= 80) {
            this.elements.scoreCircle.style.stroke = 'var(--success-color)';
            this.elements.scoreFeedback.textContent = '훌륭한 자세입니다! 👍';
        } else if (score >= 60) {
            this.elements.scoreCircle.style.stroke = 'var(--warning-color)';
            this.elements.scoreFeedback.textContent = '좋은 자세입니다 👌';
        } else {
            this.elements.scoreCircle.style.stroke = 'var(--danger-color)';
            this.elements.scoreFeedback.textContent = '자세 개선이 필요해요 📐';
        }
    }
    
    updateEmotionAnalysis() {
        // 감정 분석 API 연동 시 구현 예정
        // 현재는 자세분석과 독립적으로 작동
        
        // 예시: 실제 감정 분석 API 호출
        // const emotionData = await this.analyzeEmotion();
        // this.elements.confidenceBar.style.width = `${emotionData.confidence}%`;
        // this.elements.focusBar.style.width = `${emotionData.focus}%`;
        
        console.log('감정 분석은 별도 모듈에서 처리됩니다');
    }
    
    updateFeedback(feedbackList) {
        // 최근 3개 피드백만 표시
        const recentFeedback = feedbackList.slice(0, 3);
        
        recentFeedback.forEach((feedback, index) => {
            setTimeout(() => {
                this.addFeedbackMessage(feedback, this.getFeedbackType(feedback));
            }, index * 500);
        });
    }
    
    getFeedbackType(feedback) {
        if (feedback.includes('✓') || feedback.includes('좋') || feedback.includes('훌륭')) {
            return 'success';
        } else if (feedback.includes('⚠') || feedback.includes('조금') || feedback.includes('개선')) {
            return 'warning';
        } else {
            return 'error';
        }
    }
    
    addFeedbackMessage(message, type = 'info') {
        const messageElement = document.createElement('p');
        messageElement.className = `feedback-message ${type}`;
        messageElement.textContent = message;
        
        // 새 메시지를 맨 위에 추가
        this.elements.feedbackMessages.insertBefore(messageElement, this.elements.feedbackMessages.firstChild);
        
        // 오래된 메시지 제거 (최대 5개까지 유지)
        const messages = this.elements.feedbackMessages.children;
        if (messages.length > 5) {
            for (let i = 5; i < messages.length; i++) {
                messages[i].remove();
            }
        }
    }
    
    drawKeypoints(keypoints) {
        if (!this.settings.showKeypoints) return;
        
        // 캔버스 클리어
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // 키포인트 그리기
        keypoints.forEach(kp => {
            // 키포인트 점 그리기
            this.ctx.beginPath();
            this.ctx.arc(kp.x, kp.y, 5, 0, 2 * Math.PI);
            this.ctx.fillStyle = this.getKeypointColor(kp.name);
            this.ctx.fill();
            
            // 신뢰도가 높은 경우 라벨 표시
            if (kp.confidence > 0.5) {
                this.ctx.fillStyle = 'white';
                this.ctx.font = '12px Arial';
                this.ctx.fillText(kp.name, kp.x + 8, kp.y - 8);
            }
        });
        
        // 스켈레톤 연결선 그리기
        this.drawSkeleton(keypoints);
    }
    
    getKeypointColor(name) {
        const colorMap = {
            'nose': '#fbbf24',
            'neck': '#10b981',
            'l_eye': '#fbbf24',
            'r_eye': '#fbbf24',
            'l_ear': '#a78bfa',
            'r_ear': '#a78bfa',
            'l_shoulder': '#3b82f6',
            'r_shoulder': '#3b82f6',
            'l_elbow': '#ef4444',
            'r_elbow': '#ef4444',
            'l_wrist': '#f59e0b',
            'r_wrist': '#f59e0b'
        };
        return colorMap[name] || '#6b7280';
    }
    
    drawSkeleton(keypoints) {
        const connections = [
            ['l_shoulder', 'r_shoulder'],
            ['l_shoulder', 'l_elbow'],
            ['l_elbow', 'l_wrist'],
            ['r_shoulder', 'r_elbow'],
            ['r_elbow', 'r_wrist'],
            ['neck', 'l_shoulder'],
            ['neck', 'r_shoulder'],
            ['neck', 'nose']
        ];
        
        const keypointMap = {};
        keypoints.forEach(kp => {
            keypointMap[kp.name] = kp;
        });
        
        this.ctx.strokeStyle = '#10b981';
        this.ctx.lineWidth = 2;
        
        connections.forEach(([start, end]) => {
            if (keypointMap[start] && keypointMap[end]) {
                this.ctx.beginPath();
                this.ctx.moveTo(keypointMap[start].x, keypointMap[start].y);
                this.ctx.lineTo(keypointMap[end].x, keypointMap[end].y);
                this.ctx.stroke();
            }
        });
    }
    
    showNotification(message) {
        this.elements.notificationText.textContent = message;
        this.elements.notification.classList.remove('hidden');
        
        // 3초 후 자동 숨김
        setTimeout(() => {
            this.elements.notification.classList.add('hidden');
        }, 3000);
    }
    
    async captureImage() {
        if (!this.isRunning) return;
        
        // 현재 프레임 캡처
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // 다운로드 링크 생성
        const link = document.createElement('a');
        link.download = `interview-capture-${new Date().getTime()}.jpg`;
        link.href = this.canvas.toDataURL('image/jpeg', 0.9);
        link.click();
        
        this.addFeedbackMessage('이미지가 캡처되었습니다 📸', 'success');
    }
    
    openSettings() {
        this.elements.settingsModal.style.display = 'block';
        
        // 현재 설정 값 로드
        document.getElementById('analysis-interval').value = this.settings.analysisInterval;
        document.getElementById('show-keypoints').checked = this.settings.showKeypoints;
        document.getElementById('voice-feedback').checked = this.settings.voiceFeedback;
        document.getElementById('sensitivity').value = this.settings.sensitivity;
        document.getElementById('sensitivity-value').textContent = this.settings.sensitivity;
    }
    
    closeModal() {
        this.elements.settingsModal.style.display = 'none';
    }
    
    showLoading(show) {
        if (show) {
            this.elements.loading.classList.remove('hidden');
        } else {
            this.elements.loading.classList.add('hidden');
        }
    }
    
    updateRealTimeStats(analysis) {
        // 실시간 통계 누적
        if (!this.stats) {
            this.stats = {
                totalFrames: 0,
                goodPosture: 0,
                averageScore: 0,
                scoreHistory: []
            };
        }
        
        this.stats.totalFrames++;
        this.stats.scoreHistory.push(analysis.posture_score);
        
        if (analysis.posture_score >= 70) {
            this.stats.goodPosture++;
        }
        
        // 최근 10개 프레임의 평균 계산
        const recentScores = this.stats.scoreHistory.slice(-10);
        this.stats.averageScore = Math.round(
            recentScores.reduce((sum, score) => sum + score, 0) / recentScores.length
        );
        
        // UI 업데이트
        this.elements.totalFrames.textContent = this.stats.totalFrames;
        this.elements.goodPostureRate.textContent = `${Math.round(this.stats.goodPosture/this.stats.totalFrames*100)}%`;
        this.elements.averageScore.textContent = this.stats.averageScore;
        
        // 세션 시간 업데이트
        if (this.startTime) {
            const elapsed = Date.now() - this.startTime;
            const minutes = Math.floor(elapsed / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            this.elements.sessionTime.textContent = 
                `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
    }
    
    visualizeKeypoints(keypoints) {
        // 캔버스에 키포인트 시각화 - 기존 drawKeypoints와 통합
        this.drawKeypoints(keypoints);
        
        // 추가적인 시각적 효과
        if (keypoints.length > 0) {
            this.addVisualEffects(keypoints);
        }
    }
    
    addVisualEffects(keypoints) {
        // 포즈 품질에 따른 시각적 피드백
        const nosePoint = keypoints.find(kp => kp.name === 'nose');
        const neckPoint = keypoints.find(kp => kp.name === 'neck');
        
        if (nosePoint && neckPoint) {
            // 머리 위치 표시
            const headTilt = Math.abs(nosePoint.x - neckPoint.x);
            const color = headTilt < 30 ? '#10b981' : '#f59e0b';
            
            this.ctx.beginPath();
            this.ctx.arc(nosePoint.x, nosePoint.y, 15, 0, 2 * Math.PI);
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 3;
            this.ctx.stroke();
        }
    }
}

// 설정 저장 함수 (전역)
function saveSettings() {
    const system = window.feedbackSystem;
    
    system.settings.analysisInterval = parseInt(document.getElementById('analysis-interval').value);
    system.settings.showKeypoints = document.getElementById('show-keypoints').checked;
    system.settings.voiceFeedback = document.getElementById('voice-feedback').checked;
    system.settings.sensitivity = parseInt(document.getElementById('sensitivity').value);
    
    // 분석 간격이 변경된 경우 재시작
    if (system.isRunning && system.analysisInterval) {
        clearInterval(system.analysisInterval);
        system.startPoseAnalysis();
    }
    
    system.closeModal();
    system.addFeedbackMessage('설정이 저장되었습니다', 'success');
}

// 모달 닫기 함수 (전역)
function closeModal() {
    window.feedbackSystem.closeModal();
}

// DOM 로드 후 시스템 초기화
document.addEventListener('DOMContentLoaded', () => {
    window.feedbackSystem = new InterviewFeedbackSystem();
});
