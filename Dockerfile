FROM python:3.11-slim

# ffmpegをインストール
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements_web.txt .
RUN pip install --no-cache-dir -r requirements_web.txt

COPY . .

EXPOSE 8000

CMD sh -c "python -m uvicorn web.main:app --host 0.0.0.0 --port ${PORT:-8000}"
