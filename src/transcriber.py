from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from openai import OpenAI

CACHE_DIR = Path("cache/transcripts")
WHISPER_SIZE_LIMIT = 24 * 1024 * 1024  # 24MB (Whisper limit is 25MB)


FFMPEG_TIMEOUT = 600  # 10分でタイムアウト


def _ffmpeg(*args: str) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", *args],
            capture_output=True, text=True,
            timeout=FFMPEG_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpegがタイムアウトしました（{FFMPEG_TIMEOUT}秒）")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg失敗:\n{result.stderr[-800:]}")
    return result


def _get_duration(path: Path) -> float:
    """ffmpegで音声の長さ（秒）を取得する"""
    result = subprocess.run(
        [
            "ffmpeg", "-i", str(path),
            "-f", "null", "-",
        ],
        capture_output=True, text=True,
        timeout=60,
    )
    # stderr から "Duration: HH:MM:SS.mm" を抽出
    for line in result.stderr.splitlines():
        if "Duration:" in line:
            duration_str = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = duration_str.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    raise RuntimeError(f"duration取得失敗: {path}")


class Transcriber:
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, file_id: str) -> Path:
        return CACHE_DIR / f"{file_id}.json"

    def load_cached(self, file_id: str) -> dict | None:
        path = self._cache_path(file_id)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _extract_audio(self, video_path: Path, output_path: Path):
        """movから音声を抽出してmp3（16kHz/mono/64kbps）に変換する"""
        _ffmpeg(
            "-i", str(video_path),
            "-vn", "-ar", "16000", "-ac", "1", "-b:a", "64k",
            str(output_path),
        )

    def _split_audio(self, audio_path: Path) -> list[Path]:
        """25MB超のmp3を時間で均等分割する（再エンコードで正確に切る）"""
        if audio_path.stat().st_size <= WHISPER_SIZE_LIMIT:
            return [audio_path]

        total_duration = _get_duration(audio_path)
        # 余裕を持って分割数を決める（ビットレート64kbps → 1秒≒8KB）
        n_chunks = int(audio_path.stat().st_size / WHISPER_SIZE_LIMIT) + 1
        chunk_duration = total_duration / n_chunks

        chunks = []
        for i in range(n_chunks):
            start = i * chunk_duration
            chunk_path = audio_path.parent / f"{audio_path.stem}_part{i:02d}.mp3"
            _ffmpeg(
                "-i", str(audio_path),
                "-ss", f"{start:.3f}",
                "-t", f"{chunk_duration:.3f}",
                "-ar", "16000", "-ac", "1", "-b:a", "64k",
                str(chunk_path),
            )
            chunks.append(chunk_path)

        return chunks

    def _transcribe_chunks(self, chunks: list[Path], original: Path) -> str:
        texts = []
        for chunk in chunks:
            with open(chunk, "rb") as f:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="ja",
                )
            texts.append(response.text)
            if chunk != original:
                chunk.unlink(missing_ok=True)
        return "\n".join(texts)

    def transcribe(self, file_id: str, file_name: str, video_path: Path) -> dict:
        cached = self.load_cached(file_id)
        if cached:
            print(f"  キャッシュ使用: {file_name}")
            return cached

        print(f"  音声抽出中: {file_name}")
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "audio.mp3"
            self._extract_audio(video_path, audio_path)
            size_mb = audio_path.stat().st_size / (1024 * 1024)
            print(f"  文字起こし中: {size_mb:.1f}MB → Whisper API")
            chunks = self._split_audio(audio_path)
            if len(chunks) > 1:
                print(f"  ({len(chunks)}分割して処理)")
            text = self._transcribe_chunks(chunks, audio_path)

        result = {
            "file_id": file_id,
            "file_name": file_name,
            "transcribed_at": datetime.now().isoformat(),
            "text": text,
        }

        self._cache_path(file_id).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result
