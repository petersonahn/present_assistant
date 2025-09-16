(function () {
    function getQuerySID() {
      const u = new URL(location.href);
      return u.searchParams.get('session_id') || u.searchParams.get('sid');
    }
    function getCookie(name) {
      return document.cookie.split('; ').find(v => v.startsWith(name + '='))?.split('=')[1];
    }
    function setCookie(name, value, days=1) {
      const d = new Date(Date.now() + days*864e5).toUTCString();
      document.cookie = `${name}=${value}; Path=/; SameSite=Lax; Secure; Expires=${d}`;
    }
    function uuid() {
      try { return crypto.randomUUID(); }
      catch { return 'sid-' + Math.random().toString(36).slice(2); }
    }
  
    let sid = getQuerySID()
           || localStorage.getItem('session_id')
           || getCookie('session_id')
           || uuid();
  
    localStorage.setItem('session_id', sid);
    setCookie('session_id', sid, 1);
    window.SESSION_ID = sid;
    console.log('[SESSION]', sid);
  
    // 다른 탭에서 세션 바꾸면 동기화
    window.addEventListener('storage', (e) => {
      if (e.key === 'session_id' && e.newValue) {
        window.SESSION_ID = e.newValue;
        console.log('[SESSION] updated from storage', e.newValue);
      }
    });
  })();
  