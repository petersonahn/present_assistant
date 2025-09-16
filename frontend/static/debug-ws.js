// frontend/static/debug-ws.js — 통합/개선판
// - 세션 단일화: URL(session_id|sid) → localStorage → cookie → 생성 → window.SESSION_ID 고정
// - 동적 URL 생성(getWSUrl/getLatestUrl): 세션 변경 시 즉시 반영
// - setSession() 시 WS 재연결 + latest 재조회
// - 하트비트(ping), 지수 백오프 재연결, 간단 로거
// - __fb 헬퍼: fetchLatest/reconnect/push/setSession/seed

(() => {
  "use strict";

  // ─────────────────────────────────────────────
  // 0) 세션ID: URL > localStorage > cookie > 생성
  //    (모든 스크립트가 같은 세션을 쓰도록 window.SESSION_ID에 고정)
  // ─────────────────────────────────────────────
  function getCookie(name) {
    return document.cookie.split("; ").find(v => v.startsWith(name + "="))?.split("=")[1];
  }
  function setCookie(name, value, days = 1) {
    const exp = new Date(Date.now() + days * 864e5).toUTCString();
    document.cookie = `${name}=${value}; Path=/; SameSite=Lax; Secure; Expires=${exp}`;
  }
  function resolveSessionId() {
    const u = new URL(location.href);
    const q = u.searchParams.get("session_id") || u.searchParams.get("sid");
    const saved = localStorage.getItem("session_id") || getCookie("session_id");
    let sid = q || saved || (crypto.randomUUID?.() || "sid-" + Math.random().toString(36).slice(2));
    if (q && q !== saved) {
      localStorage.setItem("session_id", q);
      setCookie("session_id", q, 1);
      sid = q;
    } else {
      localStorage.setItem("session_id", sid);
      setCookie("session_id", sid, 1);
    }
    window.SESSION_ID = sid; // 전역 통일 포인트
    return sid;
  }

  let sessionId = window.SESSION_ID || resolveSessionId();

  // 동적 URL 빌더 (세션 변경 시 즉시 반영)
  function getWSUrl() {
    const base = new URL("/ws", window.location.origin);
    base.protocol = (window.location.protocol === "https:") ? "wss:" : "ws:";
    base.searchParams.set("session_id", sessionId);
    return base.href;
  }
  function getLatestUrl() {
    return `/api/feedback/latest?session_id=${encodeURIComponent(sessionId)}`;
  }
  const AGG_URL = "/api/aggregate"; // 수동 주입용

  // ─────────────────────────────────────────────
  // 1) 로그 유틸
  // ─────────────────────────────────────────────
  function logGroup(title, obj) {
    try {
      console.groupCollapsed(`%c${title}`,'color:#22a;font-weight:bold;');
      console.log(obj);
      if (obj && typeof obj === 'object') {
        if ('overall' in obj) console.log('overall:', obj.overall);
        if ('voice'   in obj) console.log('voice  :', obj.voice);
        if ('pose'    in obj) console.log('pose   :', obj.pose);
        if ('face'    in obj) console.log('face   :', obj.face);
        if ('emotion' in obj) console.log('emotion:', obj.emotion);
        if ('tips'    in obj) console.log('tips   :', obj.tips);
      }
    } finally { console.groupEnd(); }
  }

  // ─────────────────────────────────────────────
  // 2) 최신 스냅샷 1회 (필요 시 폴링 주석 해제)
  // ─────────────────────────────────────────────
  async function fetchLatestOnce() {
    try {
      const res = await fetch(getLatestUrl(), { headers: { 'Accept': 'application/json' } });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      logGroup('HTTP latest snapshot', data);
      return data;
    } catch (e) {
      console.warn('[latest] fetch error:', e);
    }
  }

  // ─────────────────────────────────────────────
  // 3) WS 연결 + 자동 재연결 + 하트비트
  // ─────────────────────────────────────────────
  let ws, aliveTimer, retry = 0;
  function connectWS() {
    const WS_URL = getWSUrl();
    console.log('[WS] connecting:', WS_URL);
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      console.log('%c[WS] opened', 'color:green');
      retry = 0;
      // 연결 직후 한 번 최신값 조회
      fetchLatestOnce();
      // keepalive
      aliveTimer = setInterval(() => {
        try { ws.readyState === 1 && ws.send(JSON.stringify({ type: 'ping', t: Date.now() })); } catch {}
      }, 25_000);
    };

    ws.onmessage = (ev) => {
      console.log('[WS raw]', ev.data);
      try { const msg = JSON.parse(ev.data); logGroup('WS message (parsed)', msg); }
      catch { /* not JSON */ }
    };

    ws.onclose = (ev) => {
      console.warn('[WS] closed', ev.code, ev.reason);
      clearInterval(aliveTimer);
      const jitter = Math.floor(Math.random() * 200);
      const delay = Math.min(30_000, 1000 * Math.pow(2, retry++)) + jitter;
      setTimeout(connectWS, delay);
    };

    ws.onerror = (e) => {
      console.warn('[WS] error', e);
    };
  }

  // ─────────────────────────────────────────────
  // 4) 페이지 로드시 실행
  // ─────────────────────────────────────────────
  window.addEventListener('DOMContentLoaded', async () => {
    console.log('%c[DEBUG] session_id = ' + sessionId, 'color:#aa22ff;font-weight:bold;');
    connectWS();
    await fetchLatestOnce();
    // 필요 시 폴링:
    // setInterval(fetchLatestOnce, 5000);
  });

  // ─────────────────────────────────────────────
  // 5) 콘솔 도우미 (세션 교체/재연결/수동 주입)
  // ─────────────────────────────────────────────
  async function pushAggregate(partial) {
    const payload = Object.assign({ session_id: sessionId }, partial || {});
    const res = await fetch(AGG_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json().catch(() => ({}));
    console.log('[aggregate] POST →', data);
    return data;
  }

  async function seedDemo() {
    return pushAggregate({
      pose: { posture: 76 },
      voice: { clarity: 80 },
      face: 62,
      tips: ['시선 유지 좋아요', '어깨 각도 안정적입니다']
    });
  }

  function setSession(newSid) {
    if (!newSid || newSid === sessionId) return;
    localStorage.setItem('session_id', newSid);
    setCookie('session_id', newSid, 1);
    window.SESSION_ID = newSid;
    sessionId = newSid;
    console.log('%c[SESSION switched] ' + newSid, 'color:#aa22ff;');
    try { ws && ws.close(); } catch {}
    // onclose 핸들러가 재연결을 트리거하지만, 즉시 최신값도 당겨오기
    fetchLatestOnce();
  }

  window.__fb = {
    get sessionId() { return sessionId; },
    fetchLatest: fetchLatestOnce,
    reconnect: () => { try { ws && ws.close(); } catch {} },
    push: pushAggregate,                // 예: __fb.push({ voice:{clarity:72} })
    seed: seedDemo,                     // 예: __fb.seed()
    setSession                          // 예: __fb.setSession('my-sid')
  };

  // 다른 탭에서 바뀌면 동기화
  window.addEventListener('storage', (e) => {
    if (e.key === 'session_id' && e.newValue) {
      setSession(e.newValue);
    }
  });
})();
