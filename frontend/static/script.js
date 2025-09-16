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
        
        // 음성 관련 상태
        this.speechRunning = false;
        this.audioStream = null;
        this.audioContext = null;
        this.analyzer = null;
        this.speechInterval = null;

        // WS 관련 상태
        this.sessionId = null;
        this.ws = null;
        this._pollTimer = null;
        
        // 설정
        this.settings = {
            analysisInterval: 2000,
            showKeypoints: false,  // 면접 환경에서 키포인트 비표시
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
            tremorStatus: document.getElementById('tremor-status'),
            keypointCount: document.getElementById('keypoint-count'),
            qualityFill: document.getElementById('quality-fill'),
            qualityText: document.getElementById('quality-text'),
            
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
            loading: document.getElementById('loading'),
            
            // 음성 분석 요소들
            speechStartBtn: document.getElementById('speech-start-btn'),
            speechStopBtn: document.getElementById('speech-stop-btn'),
            volumeFill: document.getElementById('volume-fill'),
            volumeValue: document.getElementById('volume-value'),
            speechRate: document.getElementById('speech-rate'),
            emotionStatus: document.getElementById('emotion-status'),
            speechConfidence: document.getElementById('speech-confidence'),
            transcriptionText: document.getElementById('transcription-text')
        };
    }
    
    bindEvents() {
        this.elements.startBtn.addEventListener('click', () => this.startAnalysis());
        this.elements.stopBtn.addEventListener('click', () => this.stopAnalysis());
        this.elements.settingsBtn.addEventListener('click', () => this.openSettings());
        
        // 음성 분석 이벤트
        if (this.elements.speechStartBtn) this.elements.speechStartBtn.addEventListener('click', () => this.startSpeechAnalysis());
        if (this.elements.speechStopBtn)  this.elements.speechStopBtn.addEventListener('click', () => this.stopSpeechAnalysis());
        
        // 모달 이벤트
        const closeBtn = document.querySelector('.close-btn');
        if (closeBtn) closeBtn.addEventListener('click', () => this.closeModal());
        window.addEventListener('click', (e) => {
            if (e.target === this.elements.settingsModal) {
                this.closeModal();
            }
        });
        
        // 설정 변경 이벤트
        const sens = document.getElementById('sensitivity');
        if (sens) sens.addEventListener('input', (e) => {
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
            
            // 카메라 스트림 시작 (오디오 포함)
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { 
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                },
                audio: true  // 오디오 활성화
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
            
            // 타이머 시작
            this.startTimer();
            
            // 분석 시작
            this.startPoseAnalysis();

            // fb-aggregator WebSocket 연결 (안전 URL 방식)
            this.connectFeedbackWS();
            // 필요시 폴링 백업을 쓰고 싶다면 주석 해제
            // this.pollFeedbackLatest();

            this.showLoading(false);
            this.addFeedbackMessage('분석이 시작되었습니다!', 'success');
            
        } catch (error) {
            console.error('Failed to start analysis:', error);
            this.showLoading(false);
            this.addFeedbackMessage('카메라 접근에 실패했습니다', 'error');
            // alert('카메라 접근 권한을 허용해주세요.');
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
        // 캔버스 완전 클리어 (이전 그림 요소 모두 제거)
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
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
        
        // 키포인트 품질 업데이트
        this.updateKeypointQuality(keypoints.length);
        
        // 자세 분석 업데이트
        this.updatePostureIndicators(analysis);
        
        // 점수 업데이트 (자세분석만)
        this.updatePostureScore(analysis.posture_score);
        
        // 피드백 메시지 업데이트
        this.updateFeedback(analysis.feedback);
        
        // 실시간 통계 업데이트
        this.updateRealTimeStats(analysis);
        
        // 알림 표시 (중요한 피드백만)
        if (analysis.tremor_detected) {
            this.showNotification('긴장을 풀고 자연스럽게 앉아보세요 🧘‍♀️');
        } else if (analysis.feedback.length > 0 && analysis.posture_score < 50) {
            this.showNotification(analysis.feedback[0]);
        }
    }
    
    updatePostureIndicators(analysis) {
        // 어깨 균형
        this.elements.shoulderStatus.textContent = this.getStatusText(analysis.shoulder_balance);
        this.elements.shoulderStatus.className = `indicator-value ${analysis.shoulder_balance}`;
        
        // 고개 기울임
        this.elements.headStatus.textContent = this.getStatusText(analysis.head_tilt);
        this.elements.headStatus.className = `indicator-value ${analysis.head_tilt}`;
        
        // 팔 자세
        this.elements.armStatus.textContent = this.getStatusText(analysis.arm_position);
        this.elements.armStatus.className = `indicator-value ${analysis.arm_position}`;
        
        // 떨림 감지
        const tremorStatus = analysis.tremor_detected ? 'tremor_detected' : 'no_tremor';
        this.elements.tremorStatus.textContent = this.getStatusText(tremorStatus);
        this.elements.tremorStatus.className = `indicator-value ${tremorStatus}`;
    }
    
    getStatusText(status) {
        const statusMap = {
            'balanced': '균형잡힘 ✓',
            'fair': '양호함 👌',
            'unbalanced': '불균형 ⚠',
            'partial': '부분감지 ◐',
            'estimated': '추정됨 ⚡',
            'detected': '감지됨 🔍',
            'straight': '바른자세 ✓',
            'slightly_tilted': '약간기울어짐 ⚠',
            'tilted': '기울어짐 ⚠',
            'neck_only': '목만감지 ◐',
            'natural': '자연스러움 ✓',
            'raised': '부자연스러움 ⚠',
            'tremor_detected': '떨림감지 🔴',
            'no_tremor': '안정됨 ✓',
            'unknown': '분석중...'
        };
        return statusMap[status] || '분석중...';
    }
    
    updateKeypointQuality(keypointCount) {
        // 18개 키포인트 중 감지된 개수로 품질 계산
        const maxKeypoints = 18;
        const quality = Math.round((keypointCount / maxKeypoints) * 100);
        
        // 품질 바 업데이트
        this.elements.qualityFill.style.width = `${quality}%`;
        this.elements.qualityText.textContent = `${quality}%`;
        
        // 품질에 따른 색상 변경
        if (quality >= 80) {
            this.elements.qualityFill.style.background = 'linear-gradient(90deg, var(--success-color), #22c55e)';
        } else if (quality >= 50) {
            this.elements.qualityFill.style.background = 'linear-gradient(90deg, var(--warning-color), #fbbf24)';
        } else {
            this.elements.qualityFill.style.background = 'linear-gradient(90deg, var(--danger-color), #f87171)';
        }
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
    
    // 면접 모드: 시각화 완전 비활성화 - 캔버스만 클리어
    drawKeypoints() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    showNotification(message) {
        this.elements.notificationText.textContent = message;
        this.elements.notification.classList.remove('hidden');
        
        // 3초 후 자동 숨김
        setTimeout(() => {
            this.elements.notification.classList.add('hidden');
        }, 3000);
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

    // ===== fb-aggregator WebSocket & 렌더링 =====

    ensureSessionId() {
        if (this.sessionId) return this.sessionId;
        const url = new URL(window.location.href);
        const fromQuery = url.searchParams.get('sid');
        if (fromQuery) this.sessionId = fromQuery;
        else this.sessionId = 'sid-' + Math.random().toString(36).slice(2, 10);
        return this.sessionId;
    }

    connectFeedbackWS() {
        try {
            const sid = this.ensureSessionId();

            // 안전한 URL 생성: /ws + wss/ws 프로토콜 자동 보정 + 쿼리 파라미터 구성
            const u = new URL('/ws', window.location.origin);
            u.protocol = (window.location.protocol === 'https:') ? 'wss:' : 'ws:';
            u.searchParams.set('session_id', sid);

            // 중복 연결 방지
            if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) return;

            this.ws = new WebSocket(u.href);
            this.ws.onopen = () => console.log('[fb-aggregator] WS connected', u.href);
            this.ws.onmessage = (e) => {
                try {
                    const data = JSON.parse(e.data);
                    this.renderAggregatorFeedback(data);
                } catch (_) { /* 텍스트일 경우 무시 */ }
            };
            this.ws.onclose = () => {
                console.warn('[fb-aggregator] WS closed, retrying…');
                setTimeout(() => this.connectFeedbackWS(), 1200);
            };
            this.ws.onerror = (error) => {
                console.warn('[fb-aggregator] WS error:', error);
                try { this.ws.close(); } catch(_) {}
                // WebSocket 실패해도 다른 기능은 계속 작동
            };
        } catch (e) {
            console.warn('WS 연결 실패:', e);
        }
    }

    // 필요 시 폴링(백업) 사용하려면 호출
    async pollFeedbackLatest() {
        if (this._pollTimer) return;
        const sid = this.ensureSessionId();
        const poll = async () => {
            try {
                const res = await fetch(`/api/feedback/latest?session_id=${encodeURIComponent(sid)}`);
                if (res.ok) {
                    const data = await res.json();
                    if (data && Object.keys(data).length) {
                        this.renderAggregatorFeedback(data);
                    }
                }
            } catch (_) {}
        };
        await poll();
        this._pollTimer = setInterval(poll, 2000);
    }

    renderAggregatorFeedback(fb) {
        // 종합 점수
        if (typeof fb.overall === 'number') {
            const circle = document.getElementById('score-circle');
            const text = document.getElementById('score-text');
            const label = document.getElementById('score-feedback');
            const v = Math.max(0, Math.min(100, Math.round(fb.overall)));
            circle.setAttribute('stroke-dasharray', `${v}, 100`);
            text.textContent = v;
            if (v >= 80)      { label.textContent = '훌륭한 자세입니다! 👍'; }
            else if (v >= 60) { label.textContent = '좋은 자세입니다 👌'; }
            else              { label.textContent = '자세 개선이 필요해요 📐'; }
        }

        // 보조 지표(있으면 표시)
        if (typeof fb.voice === 'number') {
            const bar = document.getElementById('confidence-bar');
            const val = document.getElementById('confidence-value');
            bar.style.width = `${fb.voice}%`;
            val.textContent = `${Math.round(fb.voice)}%`;
        }
        if (typeof fb.face === 'number') {
            const bar = document.getElementById('focus-bar');
            const val = document.getElementById('focus-value');
            bar.style.width = `${fb.face}%`;
            val.textContent = `${Math.round(fb.face)}%`;
        }

        // 메시지
        if (Array.isArray(fb.tips)) {
            const wrap = document.getElementById('feedback-messages');
            fb.tips.slice(0,3).forEach(msg=>{
                const p = document.createElement('p');
                p.className = 'feedback-message';
                p.textContent = msg;
                wrap.insertBefore(p, wrap.firstChild);
            });
            // 오래된 메시지 정리(최대 5개 유지)
            const nodes = wrap.children;
            if (nodes.length > 5) {
                for (let i = 5; i < nodes.length; i++) nodes[i].remove();
            }
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

// ====== 음성 분석 메서드 (prototype 부착) ======

InterviewFeedbackSystem.prototype.startSpeechAnalysis = async function() {
    if (this.speechRunning) return;
    try {
        // v3-speech API에 분석 시작 요청
        const response = await fetch('/api/v3-speech/speech/start_realtime', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        this.speechRunning = true;
        if (this.elements.speechStartBtn) this.elements.speechStartBtn.disabled = true;
        if (this.elements.speechStopBtn)  this.elements.speechStopBtn.disabled = false;
        
        // 음성 활성화 상태 UI 업데이트(선택)
        const firstCard = document.querySelector('.analysis-card');
        if (firstCard) firstCard.classList.add('speech-active');
        
        // 실시간 음성 데이터 스트리밍 시작
        this.startSpeechStreaming();
        this.addFeedbackMessage('음성 분석이 시작되었습니다', 'success');
    } catch (error) {
        console.error('음성 분석 시작 실패:', error);
        this.addFeedbackMessage('음성 분석 시작에 실패했습니다', 'error');
    }
};

InterviewFeedbackSystem.prototype.stopSpeechAnalysis = async function() {
    if (!this.speechRunning) return;
    try {
        const response = await fetch('/api/v3-speech/speech/stop_realtime', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        // 응답 상태와 무관하게 로컬 상태 정리
        this.speechRunning = false;
        if (this.elements.speechStartBtn) this.elements.speechStartBtn.disabled = false;
        if (this.elements.speechStopBtn)  this.elements.speechStopBtn.disabled = true;
        
        const firstCard = document.querySelector('.analysis-card');
        if (firstCard) firstCard.classList.remove('speech-active');
        
        if (this.speechInterval) { clearInterval(this.speechInterval); this.speechInterval = null; }
        if (this.audioContext)  { try { this.audioContext.close(); } catch(_){} this.audioContext = null; }
        
        this.addFeedbackMessage('음성 분석이 중지되었습니다', 'info');
    } catch (error) {
        console.error('음성 분석 중지 실패:', error);
        this.addFeedbackMessage('음성 분석 중지에 실패했습니다', 'error');
    }
};

InterviewFeedbackSystem.prototype.startSpeechStreaming = function() {
    this.speechInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/v3-speech/speech/status');
            const data = await response.json();
            if (data.success && data.status) {
                this.updateSpeechUI(data.status);
            }

            // 최신 분석 결과 조회
            const resultsResponse = await fetch('/api/v3-speech/speech/results/latest');
            if (resultsResponse.ok) {
                const results = await resultsResponse.json();
                if (results.success && results.data) {
                    this.updateTranscription(results.data);
                }
            }
        } catch (error) {
            console.error('음성 데이터 스트리밍 오류:', error);
        }
    }, 1000); // 1초마다 업데이트
};

InterviewFeedbackSystem.prototype.updateSpeechUI = function(status) {
    // 음량 업데이트
    const volume = Math.round((status.current_volume || 0) * 100);
    if (this.elements.volumeFill)  this.elements.volumeFill.style.width = `${volume}%`;
    if (this.elements.volumeValue) this.elements.volumeValue.textContent = `${volume}%`;
    
    // 말하기 속도 업데이트
    if (this.elements.speechRate) this.elements.speechRate.textContent = `${Math.round(status.speech_rate || 0)} WPM`;
    
    // 감정 상태 업데이트
    const emotion = status.dominant_emotion || 'neutral';
    if (this.elements.emotionStatus) {
        this.elements.emotionStatus.textContent = this.getEmotionLabel(emotion);
        this.elements.emotionStatus.className = `indicator-value ${emotion}`;
    }
    
    // 자신감 업데이트
    const confidence = Math.round((status.confidence_level || 0) * 100);
    if (this.elements.speechConfidence) this.elements.speechConfidence.textContent = `${confidence}%`;
};

InterviewFeedbackSystem.prototype.updateTranscription = function(data) {
    if (data.transcription && this.elements.transcriptionText) {
        const transcriptionDiv = this.elements.transcriptionText;
        transcriptionDiv.textContent = data.transcription;
        transcriptionDiv.scrollTop = transcriptionDiv.scrollHeight;
    }
};

InterviewFeedbackSystem.prototype.getEmotionLabel = function(emotion) {
    const labels = {
        'neutral': '중립',
        'positive': '긍정적',
        'negative': '부정적',
        'excited':  '흥미로운',
        'confident':'자신감',
        'nervous':  '긴장'
    };
    return labels[emotion] || '중립';
};

// DOM 로드 후 시스템 초기화
document.addEventListener('DOMContentLoaded', () => {
    window.feedbackSystem = new InterviewFeedbackSystem();
});
