from fastapi import FastAPI
app = FastAPI(title="Emotion")
@app.get("/"), @app.get("/api/health")
def root(): return {"status":"ok","service":"v2-emotion"}
