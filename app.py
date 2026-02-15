"""
AI Clip Creator - УНИВЕРСАЛЬНАЯ ВЕРСИЯ
FastAPI приложение с поддержкой уровней производительности
"""

import os
import sys
import uuid
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Импорт наших модулей
from youtube import download_youtube_video, get_youtube_metadata, YouTubeAuth, YouTubeUploader
from video_processor import VideoProcessor, ProcessingConfig, VideoMode, TrackingMode
from config import PerformanceLevel, get_config, detect_environment, print_environment_info

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Базовые пути
BASE_DIR = Path(__file__).resolve().parent
WORK_DIR = BASE_DIR / "work"
OUT_DIR = BASE_DIR / "out"
PREVIEW_DIR = BASE_DIR / "preview"
STATIC_DIR = BASE_DIR / "static"
ASSETS_DIR = BASE_DIR / "assets"

# Создание необходимых директорий
for directory in [WORK_DIR, OUT_DIR, PREVIEW_DIR, STATIC_DIR, ASSETS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# OAuth директории
OAUTH_DIR = WORK_DIR / "oauth"
CLIENTS_DIR = OAUTH_DIR / "clients"
TOKENS_DIR = OAUTH_DIR / "tokens"
for d in [OAUTH_DIR, CLIENTS_DIR, TOKENS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Инициализация YouTube менеджеров
youtube_auth = YouTubeAuth(str(TOKENS_DIR), str(CLIENTS_DIR))
youtube_uploader = YouTubeUploader(youtube_auth)

# Инициализация FastAPI
app = FastAPI(
    title="AI Clip Creator",
    description="Универсальное создание вертикальных видео",
    version="3.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Монтирование статических файлов
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/out", StaticFiles(directory=OUT_DIR), name="out")
app.mount("/preview", StaticFiles(directory=PREVIEW_DIR), name="preview")
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

# Глобальное состояние
app_state = {
    "ws_clients": set(),
    "current_jobs": {},
    "settings": {},
    "performance_level": None,  # Будет установлен автоматически или вручную
}


# ==================== WebSocket управление ====================
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting: {e}")
                dead_connections.append(connection)
        
        for conn in dead_connections:
            self.disconnect(conn)

    async def send_log(self, message: str, level: str = "info"):
        await self.broadcast({
            "type": "log",
            "level": level,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })

    async def send_progress(self, job_id: str, current: int, total: int, message: str = ""):
        await self.broadcast({
            "type": "progress",
            "job_id": job_id,
            "current": current,
            "total": total,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_json({
            "type": "init",
            "state": {
                "jobs": list(app_state["current_jobs"].values()),
                "settings": app_state["settings"],
                "performance_level": app_state["performance_level"]
            }
        })
        
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# ==================== Основные маршруты ====================

@app.on_event("startup")
async def startup_event():
    """При запуске показываем информацию об окружении"""
    print_environment_info()
    
    # Устанавливаем рекомендуемый уровень производительности
    from config import get_recommended_level
    app_state["performance_level"] = get_recommended_level()
    logger.info(f"Default performance level set to: {app_state['performance_level']}")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Главная страница"""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        # Создаем базовый index.html если его нет
        return HTMLResponse(content=f"""
        <html>
        <head><title>AI Clip Creator</title></head>
        <body>
        <h1>AI Clip Creator v3.0</h1>
        <p>Status: Running</p>
        <p>Access API docs: <a href="/docs">/docs</a></p>
        </body>
        </html>
        """)
    
    with open(index_file, 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())


@app.get("/api/health")
async def health_check():
    """Проверка здоровья сервиса"""
    env = detect_environment()
    return {
        "status": "healthy",
        "version": "3.0.0",
        "environment": env,
        "performance_level": str(app_state["performance_level"]),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/performance/levels")
async def get_performance_levels():
    """Получить доступные уровни производительности"""
    from config import PERFORMANCE_PRESETS, get_recommended_level
    
    levels = []
    for level, config in PERFORMANCE_PRESETS.items():
        levels.append({
            "value": int(level),
            "name": level.name,
            "whisper_model": config.whisper_model,
            "yolo_model": config.yolo_model,
            "use_gpu": config.use_gpu,
            "enable_ai": config.enable_ai_analysis,
            "max_segments": config.max_segments,
        })
    
    return {
        "success": True,
        "levels": levels,
        "current": int(app_state["performance_level"]),
        "recommended": int(get_recommended_level())
    }


@app.post("/api/performance/set")
async def set_performance_level(level: int):
    """Установить уровень производительности"""
    try:
        perf_level = PerformanceLevel(level)
        app_state["performance_level"] = perf_level
        
        await manager.broadcast({
            "type": "performance_changed",
            "level": int(perf_level)
        })
        
        return {
            "success": True,
            "level": int(perf_level),
            "name": perf_level.name
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid performance level")


@app.post("/api/process/url")
async def process_url(url: str = Form(...), settings: str = Form("{}")):
    """Обработка видео с YouTube URL"""
    try:
        settings_dict = json.loads(settings)
        job_id = str(uuid.uuid4())[:8]
        
        job = {
            "id": job_id,
            "status": "downloading",
            "source_type": "url",
            "source": url,
            "created_at": datetime.now().isoformat(),
            "progress": 0
        }
        
        app_state["current_jobs"][job_id] = job
        await manager.broadcast({"type": "job_created", "job": job})
        
        asyncio.create_task(_process_video_job(job_id, url, "url", settings_dict))
        
        return {"success": True, "job_id": job_id}
        
    except Exception as e:
        logger.error(f"Process URL error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process/file")
async def process_file(file: UploadFile = File(...), settings: str = Form("{}")):
    """Обработка загруженного файла"""
    try:
        settings_dict = json.loads(settings)
        job_id = str(uuid.uuid4())[:8]
        
        upload_dir = WORK_DIR / "uploads"
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / f"{job_id}_{file.filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        job = {
            "id": job_id,
            "status": "processing",
            "source_type": "file",
            "source": str(file_path),
            "created_at": datetime.now().isoformat(),
            "progress": 0
        }
        
        app_state["current_jobs"][job_id] = job
        await manager.broadcast({"type": "job_created", "job": job})
        
        asyncio.create_task(_process_video_job(job_id, str(file_path), "file", settings_dict))
        
        return {"success": True, "job_id": job_id}
        
    except Exception as e:
        logger.error(f"Process file error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _process_video_job(
    job_id: str,
    source: str,
    source_type: str,
    settings: Dict[str, Any]
):
    """Фоновая обработка видео"""
    job = app_state["current_jobs"][job_id]
    
    try:
        # 1. Загрузка видео (если URL)
        if source_type == "url":
            await manager.send_log("Загрузка видео с YouTube...", "info")
            job["status"] = "downloading"
            await manager.broadcast({"type": "job_updated", "job": job})
            
            download_dir = WORK_DIR / "download"
            download_dir.mkdir(exist_ok=True)
            
            video_path = await asyncio.get_event_loop().run_in_executor(
                None,
                download_youtube_video,
                source,
                str(download_dir)
            )
            
            await manager.send_log(f"Видео скачано: {Path(video_path).name}", "success")
        else:
            video_path = source
        
        # 2. Подготовка конфигурации
        await manager.send_log("Подготовка конфигурации...", "info")
        
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            await manager.send_log("ВНИМАНИЕ: OpenAI API ключ не найден", "warning")
        
        # Получаем performance level из settings или используем глобальный
        perf_level = settings.get("performance_level")
        if perf_level is not None:
            perf_level = PerformanceLevel(int(perf_level))
        else:
            perf_level = app_state["performance_level"]
        
        config = ProcessingConfig(
            input_path=video_path,
            output_dir=str(OUT_DIR),
            performance_level=perf_level,
            
            mode=VideoMode[settings.get("content_mode", "DYNAMIC").upper()],
            tracking_mode=TrackingMode[settings.get("tracking_mode", "PERSON").upper()],
            
            output_width=int(settings.get("output_width", 1080)),
            output_height=int(settings.get("output_height", 1920)),
            
            whisper_language=settings.get("whisper_language", "ru"),
            
            subtitle_font_size=int(settings.get("font_size", 56)),
            subtitle_position=float(settings.get("subtitle_position", 0.82)),
            
            openai_api_key=openai_key,
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.proxyapi.ru/openai/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5-nano")
        )
        
        # 3. Обработка
        job["status"] = "processing"
        await manager.broadcast({"type": "job_updated", "job": job})
        await manager.send_log(f"Начинаем обработку (Level {perf_level})...", "info")
        
        processor = VideoProcessor(config)
        
        async def progress_cb(message: str, current: int, total: int):
            job["progress"] = int((current / total) * 100) if total > 0 else 0
            await manager.send_progress(job_id, current, total, message)
            await manager.broadcast({"type": "job_updated", "job": job})
        
        output_files = await processor.process(progress_callback=progress_cb)
        
        # 4. Завершение
        job["status"] = "completed"
        job["output_files"] = output_files
        job["completed_at"] = datetime.now().isoformat()
        
        await manager.send_log(f"✅ Обработка завершена! Создано {len(output_files)} клипов", "success")
        await manager.broadcast({"type": "job_completed", "job": job})
        
    except Exception as e:
        logger.error(f"Job {job_id} error: {e}", exc_info=True)
        job["status"] = "error"
        job["error"] = str(e)
        await manager.send_log(f"❌ Ошибка: {e}", "error")
        await manager.broadcast({"type": "job_error", "job": job})


# YouTube API endpoints (сокращенная версия)
@app.get("/api/youtube/metadata")
async def get_youtube_meta(url: str):
    try:
        metadata = await asyncio.get_event_loop().run_in_executor(
            None,
            get_youtube_metadata,
            url
        )
        return {"success": True, "metadata": metadata}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting AI Clip Creator v3.0 (Universal)")
    
    # Определяем порт (для Colab используем переменную окружения)
    port = int(os.getenv("PORT", 9008))
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )