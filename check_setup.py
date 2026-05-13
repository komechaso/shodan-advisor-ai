#!/usr/bin/env python3
"""セットアップ確認スクリプト: python3 check_setup.py"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

OK = "\033[92m✓\033[0m"
NG = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"

errors = []

def check(label: str, ok: bool, hint: str = ""):
    symbol = OK if ok else NG
    print(f"  {symbol} {label}")
    if not ok and hint:
        print(f"      → {hint}")
    if not ok:
        errors.append(label)


print("\n=== Sales Advisor AI — セットアップ確認 ===\n")

# Python バージョン
ver = sys.version_info
check(
    f"Python {ver.major}.{ver.minor}.{ver.micro}",
    ver >= (3, 9),
    "Python 3.9 以上が必要です",
)

# ffmpeg
check(
    "ffmpeg",
    shutil.which("ffmpeg") is not None,
    "brew install ffmpeg でインストールしてください",
)

# Pythonパッケージ
packages = {
    "anthropic": "pip3 install anthropic",
    "openai": "pip3 install openai",
    "googleapiclient": "pip3 install google-api-python-client",
    "google_auth_oauthlib": "pip3 install google-auth-oauthlib",
    "dotenv": "pip3 install python-dotenv",
}
for pkg, hint in packages.items():
    try:
        __import__(pkg)
        check(f"pip: {pkg}", True)
    except ImportError:
        check(f"pip: {pkg}", False, hint)

# .env
env_path = Path(".env")
check(".env ファイル", env_path.exists(), "cp .env.example .env を実行してください")

if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv()

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    check(
        "ANTHROPIC_API_KEY",
        bool(anthropic_key) and not anthropic_key.startswith("your_"),
        ".env の ANTHROPIC_API_KEY を設定してください",
    )

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    check(
        "OPENAI_API_KEY",
        bool(openai_key) and not openai_key.startswith("your_"),
        ".env の OPENAI_API_KEY を設定してください",
    )

# credentials.json
creds_file = os.environ.get("GOOGLE_CREDENTIALS_FILE", "credentials.json")
check(
    f"Google認証: {creds_file}",
    Path(creds_file).exists(),
    "Google Cloud Console から OAuth2 クライアントID (Desktop app) をダウンロードして\n"
    f"      {creds_file} として保存してください\n"
    "      https://console.cloud.google.com → APIとサービス → 認証情報",
)

# Claude API 疎通テスト
if not errors or errors == [e for e in errors if "ANTHROPIC" not in e]:
    try:
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if key and not key.startswith("your_"):
            client = anthropic.Anthropic(api_key=key)
            msg = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
            )
            check("Claude API 疎通", True)
    except Exception as e:
        check("Claude API 疎通", False, str(e)[:100])

print()
if errors:
    print(f"  {NG} {len(errors)}件の設定が未完了です。上記の案内に従って設定してください。")
else:
    print(f"  {OK} すべての設定が完了しています！")
    print()
    print("  実行例:")
    print("    python3 advisor.py --list-folders")
    print("    python3 advisor.py --folder '商談録画'")
print()
