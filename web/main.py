from __future__ import annotations

import json
import os
import re as _re
import shutil
import subprocess
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
        import asyncio
        sent = 0
        ticks = 0
        while True:
            job = _jobs.get(job_id, {})
            events = job.get("events", [])

            for ev in events[sent:]:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                sent += 1
                ticks = 0

            if job.get("done") and sent >= len(events):
                # Linger 2s so client finishes parsing events before stream closes
                await asyncio.sleep(2)
                break

            ticks += 1
            # Keepalive data event every ~15s to prevent Render proxy timeout
            # Use data event (not SSE comment) so Render's proxy resets its idle timer
            if ticks % 37 == 0:
                yield 'data: {"type":"ping"}\n\n'

            await asyncio.sleep(0.4)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _extract_drive_file_id(url: str) -> str:
    for pattern in [r'/file/d/([a-zA-Z0-9_-]+)', r'[?&]id=([a-zA-Z0-9_-]+)']:
        m = _re.search(pattern, url)
        if m:
            return m.group(1)
    raise RuntimeError(
        "URLからファイルIDを取得できませんでした。\n"
        "Google Driveの「共有」→「リンクをコピー」で取得したURLを貼り付けてください。"
    )


def _download_drive_file(drive_url: str, dest_path: Path) -> None:
    """Download a Google Drive file. Tries multiple URL formats in order."""
    import requests

    file_id = _extract_drive_file_id(drive_url)

    # Try these download URL patterns in order
    candidate_urls = [
        f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
        f"https://drive.google.com/uc?id={file_id}&export=download&confirm=t",
        f"https://docs.google.com/uc?id={file_id}&export=download&confirm=t",
    ]

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    last_error = ""
    for url in candidate_urls:
        try:
            resp = session.get(url, stream=True, timeout=60, allow_redirects=True)
            resp.raise_for_status()
        except requests.RequestException as e:
            last_error = str(e)
            continue

        content_type = resp.headers.get("Content-Type", "")
        if "text/html" in content_type:
            # Google returned an auth/confirm page — try next URL
            last_error = f"HTMLページが返されました（{url}）"
            resp.close()
            continue

        # Got a non-HTML response — stream to disk
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    f.write(chunk)

        if not dest_path.exists() or dest_path.stat().st_size == 0:
            dest_path.unlink(missing_ok=True)
            last_error = "ダウンロードファイルが空でした"
            continue

        # Final guard: confirm the file is not secretly an HTML page
        with open(dest_path, "rb") as f:
            header = f.read(512).lower()
        if b"<!doctype" in header or b"<html" in header:
            dest_path.unlink(missing_ok=True)
            last_error = "ファイルの代わりにHTMLページが保存されました"
            continue

        return  # success

    raise RuntimeError(
        "Google Driveからのダウンロードに失敗しました。\n"
        "ファイルの共有設定を「リンクを知っている全員が閲覧可」にしてください。\n"
        f"（試みたURL全てで失敗: {last_error}）"
    )


def _download_and_process(job_id: str, drive_url: str) -> None:
    """Download Google Drive video then extract audio with ffmpeg."""
    tmpdir = tempfile.mkdtemp()
    try:
        _push(job_id, {"type": "progress", "step": 2, "pct": 5,
                       "message": "Google Driveからダウンロード中（大容量ファイルは数十分かかる場合があります）..."})

        video_path = Path(tmpdir) / "video.mp4"

        # Monitor download progress and push events every 15s so the SSE
        # connection stays alive during long downloads (Render proxy timeout)
        stop_monitor = threading.Event()
        def _monitor_download():
            while not stop_monitor.wait(15):
                if video_path.exists():
                    size_mb = video_path.stat().st_size / (1024 * 1024)
                    _push(job_id, {"type": "progress", "step": 2, "pct": 5,
                                   "message": f"Google Driveからダウンロード中... {size_mb:.0f}MB 取得済み"})
        monitor = threading.Thread(target=_monitor_download, daemon=True)
        monitor.start()
        try:
            _download_drive_file(drive_url, video_path)
        finally:
            stop_monitor.set()
            monitor.join(timeout=2)

        size_mb = video_path.stat().st_size / (1024 * 1024)
        _push(job_id, {"type": "progress", "step": 2, "pct": 20,
                       "message": f"ダウンロード完了（{size_mb:.0f}MB）。音声を抽出中..."})

        audio_path = Path(tmpdir) / "audio.mp3"
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vn", "-ar", "16000", "-ac", "1", "-b:a", "32k",
                str(audio_path),
            ],
            capture_output=True, text=True, timeout=7200,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"音声の抽出に失敗しました。動画ファイルが壊れているか、非対応の形式の可能性があります。\n"
                f"詳細: {result.stderr[-300:]}"
            )

        # 動画ファイルはもう不要なので削除してディスク容量を確保
        video_path.unlink(missing_ok=True)

        if not audio_path.exists() or audio_path.stat().st_size == 0:
            raise RuntimeError("音声ファイルの作成に失敗しました。")

        _push(job_id, {"type": "progress", "step": 3, "pct": 30,
                       "message": "文字起こし中（数分かかります）..."})
        _process(job_id, str(audio_path), "audio.mp3")

    except Exception as exc:
        _push(job_id, {"type": "error", "message": str(exc)[:800]})
        _jobs[job_id]["done"] = True
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """SSE接続が切れた場合のフォールバック用ステータス取得エンドポイント"""
    job = _jobs.get(job_id)
    if not job:
        return {"exists": False}
    events = job.get("events", [])
    error_events = [e for e in events if e.get("type") == "error"]
    done_events  = [e for e in events if e.get("type") == "done"]
    return {
        "exists": True,
        "done":   job["done"],
        "error":  error_events[-1]["message"] if error_events else None,
        "result": done_events[-1]["result"]   if done_events  else None,
    }


@app.get("/api/test-drive")
async def test_drive_url(url: str):
    """Diagnostic: test if a Google Drive URL is downloadable."""
    import requests
    try:
        file_id = _extract_drive_file_id(url)
        resp = requests.head(
            f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
            allow_redirects=True,
        )
        return {
            "ok": resp.status_code == 200,
            "file_id": file_id,
            "status_code": resp.status_code,
            "content_type": resp.headers.get("Content-Type", ""),
            "content_disposition": resp.headers.get("Content-Disposition", ""),
            "content_length": resp.headers.get("Content-Length", "unknown"),
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "type": type(e).__name__}


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
