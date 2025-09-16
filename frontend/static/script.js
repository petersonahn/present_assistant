// 실시간 면접 피드백 시스템 JavaScript (개선판)
// - 세션 ID 단일화: URL(session_id|sid) → window.SESSION_ID → localStorage → 생성
// - WS/HTTP가 동일 세션을 사용하도록 통일
// - SVG 원형 게이지 stroke-dasharray 속성 설정 방식 수정
// - WS 재연결 및 폴백 폴링 보강
// - 불필요한 오디오 권한 팝업 방지(카메라 시작은 영상만)

class InterviewFeedbackSystem {
    constructor() {
      // ─────────────────────────────────────
      // 공용 세션 ID 확보 (모든 스크립트와 공유)
      // ─────────────────────────────────────
      this.sessionId = this.ensureSessionId();
  
      // DOM 엘리먼트
      this.video = document.getElementById('video');
      this.canvas = document.getElementById('canvas');
      this.ctx = this.canvas.getContext('2d');
  
      // 런타임 상태
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
  
    // ─────────────────────────────────────
    // 세션 ID: URL → window.SESSION_ID → localStorage → 생성
    // 결정된 값을 전역/스토리지에 저장해 다른 스크립트와 공유
    // ─────────────────────────────────────
    ensureSessionId() {
      try {
        // 1) 이미 정의된 전역값 우선
        if (window.SESSION_ID) return window.SESSION_ID;
  
        const u = new URL(window.location.href);
        const fromQuery = u.searchParams.get('session_id') || u.searchParams.get('sid');
        const fromStorage = localStorage.getItem('session_id');
  
        const sid = fromQuery || fromStorage || (crypto.randomUUID?.() || ('sid-' + Math.random().toString(36).slice(2)));
  
        // 전역 및 로컬 저장
        window.SESSION_ID = sid;
        localStorage.setItem('session_id', sid);
  
        // 다른 탭 변경 동기화
        window.addEventListener('storage', (e) => {
          if (e.key === 'session_id' && e.newValue) {
            window.SESSION_ID = e.newValue;
            this.sessionId = e.newValue;
            try { this.ws && this.ws.close(); } catch(_) {}
          }
        });
  
        return sid;
      } catch (e) {
        const fallback = 'sid-' + Math.random().toString(36).slice(2);
        window.SESSION_ID = fallback;
        return fallback;
      }
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
      this.elements.startBtn?.addEventListener('click', () => this.startAnalysis());
      this.elements.stopBtn?.addEventListener('click', () => this.stopAnalysis());
      this.elements.settingsBtn?.addEventListener('click', () => this.openSettings());
  
      // 음성 분석 이벤트
      this.elements.speechStartBtn?.addEventListener('click', () => this.startSpeechAnalysis());
      this.elements.speechStopBtn?.addEventListener('click', () => this.stopSpeechAnalysis());
  
      // 모달 이벤트
      const closeBtn = document.querySelector('.close-btn');
      closeBtn?.addEventListener('click', () => this.closeModal());
      window.addEventListener('click', (e) => {
        if (e.target === this.elements.settingsModal) this.closeModal();
      });
  
      // 설정 변경 이벤트
      const sens = document.getElementById('sensitivity');
      sens?.addEventListener('input', (e) => {
        const v = (e.target?.value ?? this.settings.sensitivity);
        const holder = document.getElementById('sensitivity-value');
        if (holder) holder.textContent = v;
      });
    }
  
    initializeEmotionBars() {
      // 감정 분석 바를 기본값으로 초기화 (자세분석과 독립적)
      if (this.elements.confidenceBar) this.elements.confidenceBar.style.width = '50%';
      if (this.elements.confidenceValue) this.elements.confidenceValue.textContent = '50%';
  
      if (this.elements.focusBar) this.elements.focusBar.style.width = '50%';
      if (this.elements.focusValue) this.elements.focusValue.textContent = '50%';
  
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
      if (!status) return;
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
  
        // 카메라 스트림 시작 (영상만; 음성은 별도 모듈 시작 시 요청)
        this.stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'user'
          },
          audio: false
        });
  
        this.video.srcObject = this.stream;
        await new Promise(resolve => { this.video.onloadedmetadata = resolve; });
  
        // 캔버스 크기 설정
        this.canvas.width = this.video.videoWidth || 1280;
        this.canvas.height = this.video.videoHeight || 720;
  
        this.isRunning = true;
        this.startTime = Date.now();
  
        // UI 업데이트
        if (this.elements.startBtn) this.elements.startBtn.disabled = true;
        if (this.elements.stopBtn) this.elements.stopBtn.disabled = false;
  
        // 타이머 시작
        this.startTimer();
  
        // 분석 시작
        this.startPoseAnalysis();
  
        // fb-aggregator WebSocket 연결 (안전 URL 방식)
        this.connectFeedbackWS();
        // 필요 시 폴백 폴링: this.pollFeedbackLatest();
  
        this.showLoading(false);
        this.addFeedbackMessage('분석이 시작되었습니다!', 'success');
  
      } catch (error) {
        console.error('Failed to start analysis:', error);
        this.showLoading(false);
        this.addFeedbackMessage('카메라 접근에 실패했습니다', 'error');
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
      if (this.analysisInterval) { clearInterval(this.analysisInterval); this.analysisInterval = null; }
      if (this.timerInterval) { clearInterval(this.timerInterval); this.timerInterval = null; }
  
      // UI 리셋
      if (this.elements.startBtn) this.elements.startBtn.disabled = false;
      if (this.elements.stopBtn) this.elements.stopBtn.disabled = true;
  
      // 캔버스 클리어
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  
      this.addFeedbackMessage('분석이 중지되었습니다', 'warning');
    }
  
    startTimer() {
      this.timerInterval = setInterval(() => {
        if (!this.startTime) return;
        const elapsed = Date.now() - this.startTime;
        const minutes = Math.floor(elapsed / 60000);
        const seconds = Math.floor((elapsed % 60000) / 1000);
        if (this.elements.timerDisplay)
          this.elements.timerDisplay.textContent = `${minutes.toString().padStart(2,'0')}:${seconds.toString().padStart(2,'0')}`;
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
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: imageData, include_result_image: false })
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
      if (!data) return;
      const { keypoints = [], analysis = {} } = data;
  
      // 키포인트 개수 업데이트
      if (this.elements.keypointCount) this.elements.keypointCount.textContent = keypoints.length;
  
      // 키포인트 품질 업데이트
      if (typeof keypoints.length === 'number') this.updateKeypointQuality(keypoints.length);
  
      // 자세 분석 업데이트
      this.updatePostureIndicators(analysis);
  
      // 점수 업데이트 (자세분석만)
      if (typeof analysis.posture_score === 'number') this.updatePostureScore(analysis.posture_score);
  
      // 피드백 메시지 업데이트
      if (Array.isArray(analysis.feedback)) this.updateFeedback(analysis.feedback);
  
      // 실시간 통계 업데이트
      this.updateRealTimeStats(analysis);
  
      // 알림 표시 (중요한 피드백만)
      if (analysis.tremor_detected) {
        this.showNotification('긴장을 풀고 자연스럽게 앉아보세요 🧘‍♀️');
      } else if (Array.isArray(analysis.feedback) && analysis.feedback.length > 0 && analysis.posture_score < 50) {
        this.showNotification(analysis.feedback[0]);
      }
    }
  
    updatePostureIndicators(analysis = {}) {
      // 어깨 균형
      if (this.elements.shoulderStatus) {
        this.elements.shoulderStatus.textContent = this.getStatusText(analysis.shoulder_balance);
        this.elements.shoulderStatus.className = `indicator-value ${analysis.shoulder_balance ?? 'unknown'}`;
      }
  
      // 고개 기울임
      if (this.elements.headStatus) {
        this.elements.headStatus.textContent = this.getStatusText(analysis.head_tilt);
        this.elements.headStatus.className = `indicator-value ${analysis.head_tilt ?? 'unknown'}`;
      }
  
      // 팔 자세
      if (this.elements.armStatus) {
        this.elements.armStatus.textContent = this.getStatusText(analysis.arm_position);
        this.elements.armStatus.className = `indicator-value ${analysis.arm_position ?? 'unknown'}`;
      }
  
      // 떨림 감지
      if (this.elements.tremorStatus) {
        const tremorStatus = analysis.tremor_detected ? 'tremor_detected' : 'no_tremor';
        this.elements.tremorStatus.textContent = this.getStatusText(tremorStatus);
        this.elements.tremorStatus.className = `indicator-value ${tremorStatus}`;
      }
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
      const maxKeypoints = 18;
      const quality = Math.max(0, Math.min(100, Math.round((keypointCount / maxKeypoints) * 100)));
  
      if (this.elements.qualityFill) this.elements.qualityFill.style.width = `${quality}%`;
      if (this.elements.qualityText) this.elements.qualityText.textContent = `${quality}%`;
  
      if (this.elements.qualityFill) {
        if (quality >= 80) {
          this.elements.qualityFill.style.background = 'linear-gradient(90deg, var(--success-color), #22c55e)';
        } else if (quality >= 50) {
          this.elements.qualityFill.style.background = 'linear-gradient(90deg, var(--warning-color), #fbbf24)';
        } else {
          this.elements.qualityFill.style.background = 'linear-gradient(90deg, var(--danger-color), #f87171)';
        }
      }
    }
  
    updatePostureScore(score) {
      const v = Math.max(0, Math.min(100, Math.round(score)));
      if (this.elements.scoreCircle) this.elements.scoreCircle.setAttribute('stroke-dasharray', `${v}, 100`);
      if (this.elements.scoreText) this.elements.scoreText.textContent = v;
  
      if (this.elements.scoreCircle && this.elements.scoreFeedback) {
        if (v >= 80) {
          this.elements.scoreCircle.style.stroke = 'var(--success-color)';
          this.elements.scoreFeedback.textContent = '훌륭한 자세입니다! 👍';
        } else if (v >= 60) {
          this.elements.scoreCircle.style.stroke = 'var(--warning-color)';
          this.elements.scoreFeedback.textContent = '좋은 자세입니다 👌';
        } else {
          this.elements.scoreCircle.style.stroke = 'var(--danger-color)';
          this.elements.scoreFeedback.textContent = '자세 개선이 필요해요 📐';
        }
      }
    }
  
    updateEmotionAnalysis() {
      console.log('감정 분석은 별도 모듈에서 처리됩니다');
    }
  
    updateFeedback(feedbackList = []) {
      const recent = feedbackList.slice(0, 3);
      recent.forEach((feedback, idx) => {
        setTimeout(() => this.addFeedbackMessage(feedback, this.getFeedbackType(feedback)), idx * 500);
      });
    }
  
    getFeedbackType(feedback = '') {
      if (feedback.includes('✓') || feedback.includes('좋') || feedback.includes('훌륭')) return 'success';
      if (feedback.includes('⚠') || feedback.includes('조금') || feedback.includes('개선')) return 'warning';
      return 'error';
    }
  
    addFeedbackMessage(message, type = 'info') {
      if (!this.elements.feedbackMessages) return;
      const el = document.createElement('p');
      el.className = `feedback-message ${type}`;
      el.textContent = message;
  
      // 새 메시지를 맨 위에 추가
      this.elements.feedbackMessages.insertBefore(el, this.elements.feedbackMessages.firstChild);
  
      // 오래된 메시지 제거 (최대 5개 유지)
      const nodes = this.elements.feedbackMessages.children;
      if (nodes.length > 5) {
        for (let i = 5; i < nodes.length; i++) nodes[i].remove();
      }
    }
  
    // 면접 모드: 시각화 비활성화 - 캔버스만 클리어
    drawKeypoints() { this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height); }
  
    showNotification(message) {
      if (!this.elements.notification || !this.elements.notificationText) return;
      this.elements.notificationText.textContent = message;
      this.elements.notification.classList.remove('hidden');
      setTimeout(() => this.elements.notification.classList.add('hidden'), 3000);
    }
  
    openSettings() {
      if (!this.elements.settingsModal) return;
      this.elements.settingsModal.style.display = 'block';
  
      // 현재 설정 값 로드
      const setVal = (id, v) => { const e = document.getElementById(id); if (e) e.value = v; };
      const setChk = (id, v) => { const e = document.getElementById(id); if (e) e.checked = v; };
  
      setVal('analysis-interval', this.settings.analysisInterval);
      setChk('show-keypoints', this.settings.showKeypoints);
      setChk('voice-feedback', this.settings.voiceFeedback);
      setVal('sensitivity', this.settings.sensitivity);
      const sensVal = document.getElementById('sensitivity-value');
      if (sensVal) sensVal.textContent = this.settings.sensitivity;
    }
  
    closeModal() { this.elements.settingsModal && (this.elements.settingsModal.style.display = 'none'); }
  
    showLoading(show) {
      if (!this.elements.loading) return;
      this.elements.loading.classList.toggle('hidden', !show);
    }
  
    updateRealTimeStats(analysis = {}) {
      if (!this.stats) { this.stats = { totalFrames: 0, goodPosture: 0, averageScore: 0, scoreHistory: [] }; }
  
      this.stats.totalFrames++;
      if (typeof analysis.posture_score === 'number') this.stats.scoreHistory.push(analysis.posture_score);
  
      if (typeof analysis.posture_score === 'number' && analysis.posture_score >= 70) this.stats.goodPosture++;
  
      // 최근 10개 평균
      const recent = this.stats.scoreHistory.slice(-10);
      this.stats.averageScore = recent.length ? Math.round(recent.reduce((s, v) => s + v, 0) / recent.length) : 0;
  
      if (this.elements.totalFrames) this.elements.totalFrames.textContent = this.stats.totalFrames;
      if (this.elements.goodPostureRate) this.elements.goodPostureRate.textContent = `${Math.round((this.stats.goodPosture / this.stats.totalFrames) * 100)}%`;
      if (this.elements.averageScore) this.elements.averageScore.textContent = this.stats.averageScore;
  
      // 세션 시간
      if (this.startTime && this.elements.sessionTime) {
        const elapsed = Date.now() - this.startTime;
        const minutes = Math.floor(elapsed / 60000);
        const seconds = Math.floor((elapsed % 60000) / 1000);
        this.elements.sessionTime.textContent = `${minutes.toString().padStart(2,'0')}:${seconds.toString().padStart(2,'0')}`;
      }
    }
  
    // ===== fb-aggregator WebSocket & 렌더링 =====
  
    connectFeedbackWS() {
      try {
        const sid = this.sessionId || this.ensureSessionId();
  
        // 안전한 URL 생성: /ws + wss/ws 프로토콜 자동 보정 + 쿼리 파라미터 구성
        const u = new URL('/ws', window.location.origin);
        u.protocol = (window.location.protocol === 'https:') ? 'wss:' : 'ws:';
        u.searchParams.set('session_id', sid);
  
        // 중복 연결 방지
        if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) return;
  
        this.ws = new WebSocket(u.href);
        this.ws.onopen = () => {
          console.log('[fb-aggregator] WS connected', u.href);
          // 연결 즉시 최신값 한번 조회 (정상 라우팅 전제)
          this.fetchLatestOnce().catch(()=>{});
        };
        this.ws.onmessage = (e) => {
          try { const data = JSON.parse(e.data); this.renderAggregatorFeedback(data); }
          catch (_) { /* 텍스트일 경우 무시 */ }
        };
        this.ws.onclose = () => {
          console.warn('[fb-aggregator] WS closed, retrying…');
          // 폴백 폴링 (옵션) — 필요 시 주석 해제
          // this.pollFeedbackLatest();
          setTimeout(() => this.connectFeedbackWS(), 1200);
        };
        this.ws.onerror = (error) => {
          console.warn('[fb-aggregator] WS error:', error);
          try { this.ws.close(); } catch(_) {}
        };
      } catch (e) {
        console.warn('WS 연결 실패:', e);
      }
    }
  
    // 필요 시 폴링(백업) 사용
    async pollFeedbackLatest() {
      if (this._pollTimer) return;
      const sid = this.sessionId || this.ensureSessionId();
      const poll = async () => {
        try {
          const res = await fetch(`/api/feedback/latest?session_id=${encodeURIComponent(sid)}`);
          if (res.ok) {
            const data = await res.json();
            if (data && Object.keys(data).length) this.renderAggregatorFeedback(data);
          }
        } catch (_) {}
      };
      await poll();
      this._pollTimer = setInterval(poll, 2000);
    }
  
    async fetchLatestOnce() {
      const sid = this.sessionId || this.ensureSessionId();
      try {
        const res = await fetch(`/api/feedback/latest?session_id=${encodeURIComponent(sid)}`, { headers: { 'Accept': 'application/json' } });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (data && Object.keys(data).length) this.renderAggregatorFeedback(data);
        return data;
      } catch (e) {
        console.warn('[latest] fetch error:', e);
      }
    }
  
    renderAggregatorFeedback(fb = {}) {
      // 종합 점수
      if (typeof fb.overall === 'number') {
        const circle = document.getElementById('score-circle');
        const text = document.getElementById('score-text');
        const label = document.getElementById('score-feedback');
        const v = Math.max(0, Math.min(100, Math.round(fb.overall)));
        circle?.setAttribute('stroke-dasharray', `${v}, 100`);
        if (text) text.textContent = v;
        if (circle && label) {
          if (v >= 80)      { circle.style.stroke = 'var(--success-color)'; label.textContent = '훌륭한 자세입니다! 👍'; }
          else if (v >= 60) { circle.style.stroke = 'var(--warning-color)'; label.textContent = '좋은 자세입니다 👌'; }
          else              { circle.style.stroke = 'var(--danger-color)';  label.textContent = '자세 개선이 필요해요 📐'; }
        }
      }
  
      // 보조 지표
      if (typeof fb.voice === 'number') {
        const bar = document.getElementById('confidence-bar');
        const val = document.getElementById('confidence-value');
        if (bar) bar.style.width = `${fb.voice}%`;
        if (val) val.textContent = `${Math.round(fb.voice)}%`;
      }
      if (typeof fb.face === 'number') {
        const bar = document.getElementById('focus-bar');
        const val = document.getElementById('focus-value');
        if (bar) bar.style.width = `${fb.face}%`;
        if (val) val.textContent = `${Math.round(fb.face)}%`;
      }
  
      // 메시지
      if (Array.isArray(fb.tips)) {
        const wrap = document.getElementById('feedback-messages');
        if (wrap) {
          fb.tips.slice(0, 3).forEach(msg => {
            const p = document.createElement('p');
            p.className = 'feedback-message';
            p.textContent = msg;
            wrap.insertBefore(p, wrap.firstChild);
          });
          // 오래된 메시지 정리(최대 5개 유지)
          const nodes = wrap.children;
          if (nodes.length > 5) for (let i = 5; i < nodes.length; i++) nodes[i].remove();
        }
      }
    }
  }
  
  // 설정 저장 함수 (전역)
  function saveSettings() {
    const system = window.feedbackSystem;
    if (!system) return;
  
    const getVal = (id, fallback) => { const e = document.getElementById(id); return e ? e.value : fallback; };
    const getChk = (id, fallback) => { const e = document.getElementById(id); return e ? !!e.checked : fallback; };
  
    system.settings.analysisInterval = parseInt(getVal('analysis-interval', system.settings.analysisInterval));
    system.settings.showKeypoints = getChk('show-keypoints', system.settings.showKeypoints);
    system.settings.voiceFeedback = getChk('voice-feedback', system.settings.voiceFeedback);
    system.settings.sensitivity = parseInt(getVal('sensitivity', system.settings.sensitivity));
  
    // 분석 간격이 변경된 경우 재시작
    if (system.isRunning && system.analysisInterval) {
      clearInterval(system.analysisInterval);
      system.startPoseAnalysis();
    }
  
    system.closeModal();
    system.addFeedbackMessage('설정이 저장되었습니다', 'success');
  }
  
  // 모달 닫기 함수 (전역)
  function closeModal() { window.feedbackSystem?.closeModal(); }
  
  // ====== 음성 분석 메서드 (prototype 부착) ======
  
  InterviewFeedbackSystem.prototype.startSpeechAnalysis = async function () {
    if (this.speechRunning) return;
    try {
      // v3-speech API에 분석 시작 요청 (현재는 서버 폴링 기반)
      const response = await fetch('/api/v3-speech/speech/start_realtime', { method: 'POST', headers: { 'Content-Type': 'application/json' } });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
  
      this.speechRunning = true;
      this.elements.speechStartBtn && (this.elements.speechStartBtn.disabled = true);
      this.elements.speechStopBtn  && (this.elements.speechStopBtn.disabled  = false);
  
      const firstCard = document.querySelector('.analysis-card');
      firstCard?.classList.add('speech-active');
  
      this.startSpeechStreaming();
      this.addFeedbackMessage('음성 분석이 시작되었습니다', 'success');
    } catch (error) {
      console.error('음성 분석 시작 실패:', error);
      this.addFeedbackMessage('음성 분석 시작에 실패했습니다', 'error');
    }
  };
  
  InterviewFeedbackSystem.prototype.stopSpeechAnalysis = async function () {
    if (!this.speechRunning) return;
    try {
      await fetch('/api/v3-speech/speech/stop_realtime', { method: 'POST', headers: { 'Content-Type': 'application/json' } });
    } catch (_) { /* ignore */ }
  
    this.speechRunning = false;
    this.elements.speechStartBtn && (this.elements.speechStartBtn.disabled = false);
    this.elements.speechStopBtn  && (this.elements.speechStopBtn.disabled  = true);
  
    const firstCard = document.querySelector('.analysis-card');
    firstCard?.classList.remove('speech-active');
  
    if (this.speechInterval) { clearInterval(this.speechInterval); this.speechInterval = null; }
    if (this.audioContext)  { try { this.audioContext.close(); } catch(_){} this.audioContext = null; }
  
    this.addFeedbackMessage('음성 분석이 중지되었습니다', 'info');
  };
  
  InterviewFeedbackSystem.prototype.startSpeechStreaming = function () {
    this.speechInterval = setInterval(async () => {
      try {
        const response = await fetch('/api/v3-speech/speech/status');
        const data = await response.json();
        if (data.success && data.status) this.updateSpeechUI(data.status);
  
        const resultsResponse = await fetch('/api/v3-speech/speech/results/latest');
        if (resultsResponse.ok) {
          const results = await resultsResponse.json();
          if (results.success && results.data) this.updateTranscription(results.data);
        }
      } catch (error) {
        console.error('음성 데이터 스트리밍 오류:', error);
      }
    }, 1000);
  };
  
  InterviewFeedbackSystem.prototype.updateSpeechUI = function (status = {}) {
    // 음량
    const volume = Math.round((status.current_volume || 0) * 100);
    this.elements.volumeFill  && (this.elements.volumeFill.style.width = `${volume}%`);
    this.elements.volumeValue && (this.elements.volumeValue.textContent = `${volume}%`);
  
    // 말하기 속도
    const speechRate = Math.round(status.speech_rate || 0);
    let speedAssessment = '';
    if (speechRate > 0) {
      if (speechRate < 120) speedAssessment = ' (느림)';
      else if (speechRate < 160) speedAssessment = ' (적당함)';
      else if (speechRate < 200) speedAssessment = ' (빠름)';
      else speedAssessment = ' (너무 빠름)';
    }
    this.elements.speechRate && (this.elements.speechRate.textContent = `${speechRate} WPM${speedAssessment}`);
  
    // 감정 상태(서버 상태 값 반영; 실제 감정분석 모듈 연동 시 교체)
    const emotion = status.dominant_emotion || 'neutral';
    if (this.elements.emotionStatus) {
      this.elements.emotionStatus.textContent = this.getEmotionLabel(emotion);
      this.elements.emotionStatus.className = `indicator-value ${emotion}`;
    }
  
    // 자신감
    const confidence = Math.round((status.confidence_level || 0) * 100);
    this.elements.speechConfidence && (this.elements.speechConfidence.textContent = `${confidence}%`);
  
    // 전사(모의)
    if (status.is_speaking && Math.random() < 0.3 && this.elements.transcriptionText) {
      const mockTranscriptions = [
        '안녕하세요. 면접에 참여하게 되어 기쁩니다.',
        '저는 이 회사에서 일하고 싶습니다.',
        '제 경험을 말씀드리겠습니다.',
        '질문에 답변하겠습니다.',
        '감사합니다.'
      ];
      const randomText = mockTranscriptions[Math.floor(Math.random() * mockTranscriptions.length)];
      this.elements.transcriptionText.textContent = randomText;
    }
  };
  
  InterviewFeedbackSystem.prototype.updateTranscription = function (data = {}) {
    if (data.transcription && this.elements.transcriptionText) {
      const t = this.elements.transcriptionText;
      t.textContent = data.transcription;
      t.scrollTop = t.scrollHeight;
    }
  };
  
  InterviewFeedbackSystem.prototype.getEmotionLabel = function (emotion) {
    const labels = { 'neutral':'중립', 'positive':'긍정적', 'negative':'부정적', 'excited':'흥미로운', 'confident':'자신감', 'nervous':'긴장' };
    return labels[emotion] || '중립';
  };
  
  // DOM 로드 후 시스템 초기화
  document.addEventListener('DOMContentLoaded', () => {
    window.feedbackSystem = new InterviewFeedbackSystem();
  });
  