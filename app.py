from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"ok": True, "service": "ssb-backend"}

@app.get("/healthz")
def healthz():
    return {"ok": True}
