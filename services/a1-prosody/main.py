from fastapi import FastAPI
app = FastAPI(title="Prosody")
@app.get("/"), @app.get("/api/health")
def root(): return {"status":"ok","service":"a1-prosody"}
