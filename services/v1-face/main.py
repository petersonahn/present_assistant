from fastapi import FastAPI
app = FastAPI(title="Face")
@app.get("/"), @app.get("/api/health")
def root(): return {"status":"ok","service":"v1-face"}
