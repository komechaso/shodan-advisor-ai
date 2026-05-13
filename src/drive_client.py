import os
import io
import sys
from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SUPPORTED_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}


class DriveClient:
    def __init__(self, credentials_file: str = "credentials.json", token_file: str = "token.json"):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = self._authenticate()

    def _authenticate(self):
        creds = None
        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_file):
                    print(f"エラー: {self.credentials_file} が見つかりません。")
                    print("Google Cloud Console からOAuth 2.0 クライアントIDをダウンロードして")
                    print(f"{self.credentials_file} として保存してください。")
                    print("詳細: https://developers.google.com/drive/api/quickstart/python")
                    sys.exit(1)
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_file, SCOPES)
                creds = flow.run_local_server(port=0)

            with open(self.token_file, "w") as f:
                f.write(creds.to_json())

        return build("drive", "v3", credentials=creds)

    def find_folder(self, folder_name: str) -> list[dict]:
        """フォルダ名でGoogle Driveを検索する"""
        results = self.service.files().list(
            q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
            fields="files(id, name, parents)",
            pageSize=10,
        ).execute()
        return results.get("files", [])

    def list_folders(self, parent_id: str = None) -> list[dict]:
        """フォルダ一覧を取得する"""
        query = "mimeType='application/vnd.google-apps.folder' and trashed=false"
        if parent_id:
            query += f" and '{parent_id}' in parents"
        results = self.service.files().list(
            q=query,
            fields="files(id, name, parents)",
            pageSize=50,
            orderBy="name",
        ).execute()
        return results.get("files", [])

    def list_recordings(self, folder_id: str) -> list[dict]:
        """指定フォルダ内の動画ファイルを全て取得する（再帰的）"""
        recordings = []
        self._collect_recordings(folder_id, recordings)
        return recordings

    def _collect_recordings(self, folder_id: str, recordings: list, depth: int = 0):
        """再帰的にフォルダをたどって録画ファイルを収集する"""
        if depth > 5:
            return

        page_token = None
        while True:
            query = f"'{folder_id}' in parents and trashed=false"
            response = self.service.files().list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
                pageSize=100,
                pageToken=page_token,
            ).execute()

            for f in response.get("files", []):
                if f["mimeType"] == "application/vnd.google-apps.folder":
                    self._collect_recordings(f["id"], recordings, depth + 1)
                else:
                    ext = Path(f["name"]).suffix.lower()
                    if ext in SUPPORTED_EXTENSIONS:
                        recordings.append(f)

            page_token = response.get("nextPageToken")
            if not page_token:
                break

    def download_file(self, file_id: str, dest_path: Path, file_name: str = "") -> Path:
        """ファイルをダウンロードする"""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        request = self.service.files().get_media(fileId=file_id)
        with open(dest_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request, chunksize=10 * 1024 * 1024)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    pct = int(status.progress() * 100)
                    print(f"\r  ダウンロード中... {pct}%", end="", flush=True)
        print()
        return dest_path
