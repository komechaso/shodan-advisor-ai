import os
from datetime import datetime

import anthropic

MODEL = "claude-opus-4-7"

SYSTEM_PROMPT = """あなたは日本のBtoB営業に精通した、トップクラスの営業コンサルタントです。
商談録画の文字起こしを分析し、成約率を向上させるための具体的で実践的なアドバイスを提供します。

## 分析の観点

### 1. オープニング・ラポール形成
- アイスブレイクの質と自然さ
- 信頼関係構築のスピードと深さ
- 顧客の状況・課題への共感

### 2. ヒアリング・課題発見
- SPIN質問（状況・問題・示唆・解決）の活用度
- 潜在課題の掘り起こし
- 顧客の言葉を引き出す傾聴スキル
- 課題の優先度・緊急度の確認

### 3. 提案・プレゼンテーション
- 顧客課題と提案内容の整合性
- ベネフィット訴求の明確さ（機能ではなく価値）
- 競合との差別化ポイントの提示
- 数字・事例・実績の活用

### 4. 反論・懸念への対応
- 反論処理のスムーズさと説得力
- 価格交渉・コスト懸念への対応
- 意思決定の障壁除去

### 5. クロージング
- クロージングのタイミングと手法
- 次のアクションの合意取得
- 決裁プロセスの把握と支援

### 6. 全体的なコミュニケーション
- 話す・聞くのバランス（理想: 営業40%：顧客60%）
- 専門用語の適切な使用
- 熱量・エネルギーレベル

## 出力形式
各録画を分析する際は、上記の観点ごとに具体的な発言を引用しながら評価し、
改善点と改善方法を明示してください。
複数の録画をまとめて分析する際は、共通のパターンと優先度の高い改善点を特定してください。"""


class Analyzer:
    def __init__(self, api_key: str = None):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"]
        )

    def analyze_single(self, transcript: dict) -> str:
        """1件の商談録画を分析する"""
        file_name = transcript["file_name"]
        text = transcript["text"]

        print(f"  分析中: {file_name}")

        user_content = f"""以下は商談録画「{file_name}」の文字起こしです。

---
{text}
---

この商談を詳細に分析し、営業担当者への具体的なフィードバックと改善アドバイスを提供してください。
良かった点と改善が必要な点の両方を、具体的な発言を引用しながら説明してください。"""

        with self.client.messages.stream(
            model=MODEL,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
        ) as stream:
            result = stream.get_final_message()

        texts = [b.text for b in result.content if b.type == "text"]
        return "\n".join(texts)

    def analyze_aggregate(self, transcripts: list[dict], individual_analyses: list[str]) -> str:
        """全商談を横断的に分析して総合アドバイスを生成する"""
        print("\n総合パターン分析中...")

        summary_parts = []
        for t, analysis in zip(transcripts, individual_analyses):
            summary_parts.append(
                f"## 商談: {t['file_name']}\n\n{analysis}\n"
            )
        combined = "\n---\n".join(summary_parts)

        user_content = f"""以下は{len(transcripts)}件の商談録画の個別分析結果です。

{combined}

---

これら全ての商談を横断的に分析して、以下を提供してください：

1. **共通の強み** — 複数の商談で一貫して見られる良い点
2. **共通の課題** — 複数の商談で繰り返し見られる改善が必要な点
3. **最優先改善アクション** — 成約率向上に最も効果的な改善項目（上位3つ）
4. **具体的なトレーニング計画** — 改善のための具体的なアクションプランと練習方法
5. **成約率向上の見込み** — 改善実施後に期待できる効果の見込み

営業チーム全体へのアドバイスとして、実践的で今すぐ取り組める内容を提示してください。"""

        full_text = ""
        with self.client.messages.stream(
            model=MODEL,
            max_tokens=8192,
            thinking={"type": "adaptive"},
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                full_text += text

        print()
        return full_text

    def run(self, transcripts: list[dict]) -> tuple[list[str], str]:
        """全録画を分析して個別分析と総合分析を返す"""
        individual_analyses = []
        for t in transcripts:
            analysis = self.analyze_single(t)
            individual_analyses.append(analysis)

        if len(transcripts) == 1:
            aggregate = individual_analyses[0]
        else:
            aggregate = self.analyze_aggregate(transcripts, individual_analyses)

        return individual_analyses, aggregate
