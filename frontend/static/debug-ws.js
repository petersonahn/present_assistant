// frontend/static/js/debug-ws.js
(() => {
    // 1) 세션ID: URL ?session_id 파라미터 우선, 없으면 랜덤
    const url = new URL(location.href);
    const sessionId = url.searchParams.get("session_id") || (crypto.randomUUID?.() || String(Date.now()));
    const WS_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws?session_id=${encodeURIComponent(sessionId)}`;
    const LATEST_URL = `/api/feedback/latest?session_id=${encodeURIComponent(sessionId)}`;
  
    // 2) 예쁜 로그 유틸
    function logGroup(title, obj) {
      try {
        console.groupCollapsed(`%c${title}`, "color:#22a; font-weight:bold;");
        console.log(obj);
        if (obj && typeof obj === "object") {
          if ("overall" in obj) console.log("overall:", obj.overall);
          if ("voice" in obj)   console.log("voice  :", obj.voice);
          if ("pose" in obj)    console.log("pose   :", obj.pose);
          if ("face" in obj)    console.log("face   :", obj.face);
          if ("emotion" in obj) console.log("emotion:", obj.emotion);
          if ("tips" in obj)    console.log("tips   :", obj.tips);
        }
      } finally {
        console.groupEnd();
      }
    }
  
    // 3) 최신 스냅샷 1회 + 주기적 폴링(선택)
    async function fetchLatestOnce() {
      try {
        const res = await fetch(LATEST_URL, { headers: { "Accept": "application/json" } });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        logGroup("HTTP latest snapshot", data);
        return data;
      } catch (e) {
        console.warn("[latest] fetch error:", e);
      }
    }
  
    // 4) WS 연결 + 자동 재연결 + 하트비트
    let ws, aliveTimer, retry = 0;
    function connectWS() {
      console.log("[WS] connecting:", WS_URL);
      ws = new WebSocket(WS_URL);
  
      ws.onopen = () => {
        console.log("%c[WS] opened", "color:green");
        retry = 0;
        // keepalive
        aliveTimer = setInterval(() => {
          try { ws.readyState === 1 && ws.send(JSON.stringify({ type: "ping", t: Date.now() })); } catch {}
        }, 25000);
      };
  
      ws.onmessage = (ev) => {
        // 원문 로그
        console.log("[WS raw]", ev.data);
        // JSON이면 구조 로그
        try {
          const msg = JSON.parse(ev.data);
          logGroup("WS message (parsed)", msg);
        } catch {
          // JSON이 아니면 패스
        }
      };
  
      ws.onclose = (ev) => {
        console.warn("[WS] closed", ev.code, ev.reason);
        clearInterval(aliveTimer);
        // 지수 백오프 재연결
        const delay = Math.min(30000, 1000 * Math.pow(2, retry++));
        setTimeout(connectWS, delay);
      };
  
      ws.onerror = (e) => {
        console.warn("[WS] error", e);
      };
    }
  
    // 5) 페이지 로드 시 바로 실행
    window.addEventListener("DOMContentLoaded", async () => {
      console.log("%c[DEBUG] session_id = " + sessionId, "color:#aa22ff; font-weight:bold;");
      connectWS();
      await fetchLatestOnce();
      // 필요하면 주기 폴링(예: 5초)
      // setInterval(fetchLatestOnce, 5000);
    });
  
    // 6) 콘솔에서 수동 호출 편의
    window.__fb = {
      sessionId,
      fetchLatest: fetchLatestOnce,
      reconnect: () => { try { ws && ws.close(); } catch {}; },
    };
  })();
  