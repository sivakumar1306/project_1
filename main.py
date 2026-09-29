import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from routers import chat, auth, profile, health, history, fall_risk, demo, fall_detection

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="MedXAI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(profile.router, prefix="/api/v1", tags=["profile"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(history.router, prefix="/api/v1", tags=["history"])
app.include_router(fall_risk.router, prefix="/api/v1", tags=["fall-risk"])
app.include_router(demo.router, prefix="/api/v1", tags=["demo"])
app.include_router(fall_detection.router, prefix="/api/v1", tags=["fall-detection"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/demo", include_in_schema=False)
def demo_page():
    return FileResponse(os.path.join(STATIC_DIR, "demo.html"), media_type="text/html")

@app.get("/")
def root():
    return {"status": "MedXAI backend is live "}