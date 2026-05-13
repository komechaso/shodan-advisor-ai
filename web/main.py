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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcriber import Transcriber
from web.analyzer_web import analyze_transcript

app = FastAPI(title="商談アドバイスくん")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2GB (single-part upload)

# Job store: {job_id: {"events": [...], "done": bool}}
_jobs: dict[str, dict] = {}

# Chunk store for large file uploads: {job_id: {"total": N, "received": N, "file_path": str, "name": str, "dir": str}}
_chunks: dict[str, dict] = {}


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


import re as _re
import urllib.request as _urllib_request
import http.cookiejar as _cookiejar


def _extract_drive_file_id(url: str) -> str:
    for pattern in [r'/file/d/([a-zA-Z0-9_-]+)', r'[?&]id=([a-zA-Z0-9_-]+)']:
        m = _re.search(pattern, url)
        if m:
            return m.group(1)
    raise RuntimeError(
        "URLからファイルIDを取得できませんでした。\n"
        "Google Driveの「共有」→「リンクをコピー」で取得したURLを貼り付けてください。"
    )


def _download_drive_file(file_id: str, out_dir: str) -> str:
    """Download a public Google Drive file using urllib (no extra deps)."""
    jar = _cookiejar.CookieJar()
    opener = _urllib_request.build_opener(_urllib_request.HTTPCookieProcessor(jar))
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]

    base_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = opener.open(base_url)

    # Large files show a virus-scan warning — extract the confirm token
    confirm = next((c.value for c in jar if c.name.startswith("download_warning")), None)
    if confirm:
        resp = opener.open(f"{base_url}&confirm={confirm}")

    # Determine filename from Content-Disposition header
    cd = resp.headers.get("Content-Disposition", "")
    fname_match = _re.findall(r'filename[^;=\n]*=([^;\n]*)', cd)
    fname = fname_match[0].strip().strip('"') if fname_match else f"recording_{file_id}.mp4"
    # Sanitize filename
    fname = Path(fname).name or f"recording_{file_id}.mp4"

    out_path = Path(out_dir) / fname
    with open(out_path, "wb") as f:
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            f.write(chunk)

    if out_path.stat().st_size == 0:
        raise RuntimeError(
            "ダウンロードしたファイルが空です。\n"
            "ファイルが「リンクを知っている全員が閲覧可」に設定されているか確認してください。"
        )
    return str(out_path)


def _download_and_process(job_id: str, drive_url: str) -> None:
    """Download from Google Drive then run the normal analysis pipeline."""
    tmpdir = tempfile.mkdtemp()
    try:
        _push(job_id, {"type": "progress", "step": 1, "pct": 5,
                       "message": "Google Driveからダウンロード中（大容量ファイルは数分〜数十分かかります）..."})

        file_id = _extract_drive_file_id(drive_url)
        file_path = _download_drive_file(file_id, tmpdir)
        file_name = Path(file_path).name

        _push(job_id, {"type": "progress", "step": 1, "pct": 28,
                       "message": f"ダウンロード完了（{file_name}）"})
        _process(job_id, file_path, file_name)

    except Exception as exc:
        _push(job_id, {"type": "error", "message": str(exc)})
        _jobs[job_id]["done"] = True
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.post("/api/analyze-drive")
async def analyze_drive(url: str = Form(...)):
    if not url.strip():
        raise HTTPException(400, "URLが指定されていません")
    if "drive.google.com" not in url and "docs.google.com" not in url:
        raise HTTPException(400, "Google DriveのURLを指定してください")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"events": [], "done": False}

    thread = threading.Thread(
        target=_download_and_process,
        args=(job_id, url.strip()),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@app.post("/api/upload-chunk")
async def upload_chunk(
    job_id: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    file_name: str = Form(...),
    chunk: UploadFile = File(...),
):
    """Receive one chunk of a large file. Chunks must arrive in order 0,1,2,..."""
    if job_id not in _chunks:
        tmpdir = tempfile.mkdtemp()
        safe_name = Path(file_name).name if file_name else "recording.mp4"
        _chunks[job_id] = {
            "total": total_chunks,
            "received": 0,
            "dir": tmpdir,
            "name": safe_name,
            "file_path": str(Path(tmpdir) / safe_name),
        }
        _jobs[job_id] = {"events": [], "done": False}

    store = _chunks[job_id]

    with open(store["file_path"], "ab") as f:
        while True:
            data = await chunk.read(4 * 1024 * 1024)
            if not data:
                break
            f.write(data)

    store["received"] += 1
    return {"received": chunk_index, "total": total_chunks}


@app.post("/api/start-analysis/{job_id}")
async def start_chunked_analysis(job_id: str):
    """Start analysis after all chunks have been uploaded."""
    if job_id not in _chunks:
        raise HTTPException(404, "ジョブが見つかりません")

    store = _chunks.pop(job_id)
    file_path = store["file_path"]
    safe_name = store["name"]

    _push(job_id, {"type": "progress", "step": 1, "pct": 5, "message": "アップロード完了"})

    thread = threading.Thread(
        target=_process,
        args=(job_id, file_path, safe_name),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("web.main:app", host="0.0.0.0", port=port, reload=False)
