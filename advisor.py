#!/usr/bin/env python3
"""
Sales Advisor AI — 商談録画分析による成約率向上アドバイザー

Usage:
  python advisor.py --list-folders
  python advisor.py --folder "商談録画"
  python advisor.py --folder-id "1abc...xyz"
  python advisor.py --folder "商談録画" --output reports/my_report.md
"""

import argparse
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.drive_client import DriveClient
from src.transcriber import Transcriber
from src.analyzer import Analyzer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Google Drive の商談録画を分析して成約率向上アドバイスを生成します"
    )
    parser.add_argument("--folder", metavar="NAME", help="Google Drive フォルダ名")
    parser.add_argument("--folder-id", metavar="ID", help="Google Drive フォルダID（直接指定）")
    parser.add_argument("--list-folders", action="store_true", help="利用可能なフォルダ一覧を表示")
    parser.add_argument("--output", metavar="PATH", help="レポート出力先（デフォルト: reports/report_<日時>.md）")
    parser.add_argument("--skip-transcription", action="store_true", help="キャッシュ済み文字起こしのみ使用（新規ダウンロードなし）")
    parser.add_argument("--limit", type=int, metavar="N", help="処理する録画数の上限")
    return parser.parse_args()


def select_folder(drive: DriveClient, folder_name: str) -> str:
    """フォルダ名からフォルダIDを解決する"""
    folders = drive.find_folder(folder_name)
    if not folders:
        print(f"エラー: フォルダ '{folder_name}' が見つかりません。")
        print("--list-folders で利用可能なフォルダを確認してください。")
        sys.exit(1)

    if len(folders) == 1:
        return folders[0]["id"]

    print(f"'{folder_name}' に一致するフォルダが複数あります:")
    for i, f in enumerate(folders):
        print(f"  [{i + 1}] {f['name']} (ID: {f['id']})")
    while True:
        try:
            choice = int(input("番号を選択してください: ")) - 1
            if 0 <= choice < len(folders):
                return folders[choice]["id"]
        except (ValueError, KeyboardInterrupt):
            pass
        print("有効な番号を入力してください。")


def list_folders(drive: DriveClient):
    print("Google Drive のフォルダ一覧:")
    folders = drive.list_folders()
    if not folders:
        print("  (フォルダが見つかりません)")
        return
    for f in folders:
        print(f"  - {f['name']} (ID: {f['id']})")


def generate_report(
    transcripts: list[dict],
    individual_analyses: list[str],
    aggregate: str,
    output_path: Path,
):
    lines = [
        "# 商談録画分析レポート",
        f"\n生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}",
        f"分析件数: {len(transcripts)}件",
        "\n---\n",
        "## 総合アドバイス・改善提案\n",
        aggregate,
        "\n---\n",
        "## 個別商談分析\n",
    ]

    for transcript, analysis in zip(transcripts, individual_analyses):
        lines.append(f"### {transcript['file_name']}\n")
        lines.append(analysis)
        lines.append("\n---\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nレポートを保存しました: {output_path}")


def main():
    args = parse_args()

    credentials_file = os.environ.get("GOOGLE_CREDENTIALS_FILE", "credentials.json")
    token_file = os.environ.get("GOOGLE_TOKEN_FILE", "token.json")

    print("Google Drive に接続中...")
    drive = DriveClient(credentials_file=credentials_file, token_file=token_file)

    if args.list_folders:
        list_folders(drive)
        return

    if not args.folder and not args.folder_id:
        print("エラー: --folder または --folder-id を指定してください。")
        print("       --list-folders でフォルダ一覧を確認できます。")
        sys.exit(1)

    folder_id = args.folder_id if args.folder_id else select_folder(drive, args.folder)

    print(f"\n録画ファイルを検索中 (フォルダID: {folder_id})...")
    recordings = drive.list_recordings(folder_id)
    if not recordings:
        print("録画ファイルが見つかりませんでした。")
        sys.exit(0)

    if args.limit:
        recordings = recordings[: args.limit]

    print(f"{len(recordings)}件の録画が見つかりました:")
    for r in recordings:
        size_mb = int(r.get("size", 0)) / (1024 * 1024)
        print(f"  - {r['name']} ({size_mb:.1f} MB)")

    transcriber = Transcriber()
    transcripts = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for rec in recordings:
            file_id = rec["id"]
            file_name = rec["name"]

            cached = transcriber.load_cached(file_id)
            if cached:
                print(f"\n[キャッシュ] {file_name}")
                transcripts.append(cached)
                continue

            if args.skip_transcription:
                print(f"\n[スキップ] {file_name} (--skip-transcription 指定のためキャッシュなしはスキップ)")
                continue

            print(f"\n[処理中] {file_name}")
            video_path = Path(tmpdir) / file_name
            print(f"  ダウンロード中...")
            drive.download_file(file_id, video_path, file_name)

            transcript = transcriber.transcribe(file_id, file_name, video_path)
            transcripts.append(transcript)

    if not transcripts:
        print("分析できる文字起こしデータがありません。")
        sys.exit(0)

    print(f"\n{len(transcripts)}件の文字起こしを分析します...")
    analyzer = Analyzer()
    individual_analyses, aggregate = analyzer.run(transcripts)

    output_path = Path(args.output) if args.output else Path(
        f"reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    generate_report(transcripts, individual_analyses, aggregate, output_path)


if __name__ == "__main__":
    main()
