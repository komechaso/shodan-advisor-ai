from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcriber import Transcriber
from web.analyzer_web import analyze_transcript

app = FastAPI(title="商談アドバイスくん")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2GB

# Job store: {job_id: {"events": [...], "done": bool}}
_jobs: dict[str, dict] = {}


def _push(job_id: str, event: dict) -> None:
    if job_id in _jobs:
        _jobs[job_id]["events"].append(event)


def _process(job_id: str, file_path: str, file_name: str) -> None:
    """バックグラウンドスレッドで録画を処理する"""
    try:
        ext = Path(file_name).suffix.lower()
        is_audio = ext in {".mp3", ".m4a", ".wav", ".aac", ".ogg", ".flac"}

        if is_audio:
            _push(job_id, {"type": "progress", "step": 2, "pct": 20, "message": "音声ファイルを確認中..."})
        else:
            _push(job_id, {"type": "progress", "step": 2, "pct": 10, "message": "音声を抽出中..."})

        transcriber = Transcriber()
        file_id = f"web_{job_id}"

        _push(job_id, {"type": "progress", "step": 3, "pct": 30, "message": "文字起こし中（数分かかります）..."})
        transcript = transcriber.transcribe(file_id, file_name, Path(file_path))

        _push(job_id, {"type": "progress", "step": 4, "pct": 70, "message": "AIが分析中..."})
        result = analyze_transcript(transcript["text"], file_name)

        _push(job_id, {"type": "done", "pct": 100, "result": result})

    except Exception as exc:
        _push(job_id, {"type": "error", "message": str(exc)})
    finally:
        _jobs[job_id]["done"] = True
        shutil.rmtree(Path(file_path).parent, ignore_errors=True)


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/api/analyze")
async def start_analysis(file: UploadFile = File(...)):
    allowed_types = ["video/", "audio/", "application/octet-stream"]
    if file.content_type and not any(file.content_type.startswith(t) for t in allowed_types):
        raise HTTPException(400, "対応していないファイル形式です（動画・音声ファイルをアップロードしてください）")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"events": [], "done": False}

    tmpdir = tempfile.mkdtemp()
    safe_name = Path(file.filename).name if file.filename else "recording.mp4"
    file_path = Path(tmpdir) / safe_name

    written = 0
    with open(file_path, "wb") as f:
        while True:
            chunk = await file.read(4 * 1024 * 1024)  # 4MB chunks
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                shutil.rmtree(tmpdir, ignore_errors=True)
                raise HTTPException(413, "ファイルサイズが2GBを超えています")
            f.write(chunk)

    _push(job_id, {"type": "progress", "step": 1, "pct": 5, "message": "アップロード完了"})

    thread = threading.Thread(
        target=_process,
        args=(job_id, str(file_path), safe_name),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@app.get("/api/stream/{job_id}")
async def stream_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "ジョブが見つかりません")

    async def generator():
        sent = 0
        import asyncio
        while True:
            job = _jobs.get(job_id, {})
            events = job.get("events", [])

            for ev in events[sent:]:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                sent += 1

            if job.get("done") and sent >= len(events):
                break

            await asyncio.sleep(0.4)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("web.main:app", host="0.0.0.0", port=port, reload=False)
