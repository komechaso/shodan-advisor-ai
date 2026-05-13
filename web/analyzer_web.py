import os
import anthropic

MODEL = "claude-opus-4-7"

SYSTEM_PROMPT = """あなたは日本のBtoB/BtoC営業に精通した、トップクラスの営業コンサルタントです。
商談録画の文字起こしを分析し、営業担当者への具体的・実践的なフィードバックとスコアを提供します。

## 評価カテゴリと採点基準（各0〜10点）

### ラポール形成（rapport）
- 0-3: 自己紹介だけで終わり、関係構築ゼロ
- 4-6: 雑談はあるが表面的。顧客の状況への共感が薄い
- 7-9: 自然な雑談→共感→信頼の流れが作れている
- 10: 顧客が自発的に話し始める雰囲気を作り出している

### ヒアリング（hearing）
- 0-3: 予算・ニーズ・緊急度を全く確認していない
- 4-6: 一部確認しているが表面的。深掘りが不足
- 7-9: SPIN的アプローチで潜在課題まで引き出せている
- 10: 顧客自身が気づいていなかった課題を言語化できた

### 提案力（proposal）
- 0-3: 機能説明のみ。顧客課題と紐付いていない
- 4-6: 課題との対応はあるが説得力に欠ける
- 7-9: 顧客の言葉を使いながらベネフィット訴求できている
- 10: 顧客が「これしかない」と感じる提案ができた

### 反論処理（objection_handling）
- 0-3: 反論で即撤退。処理スキルがほぼない
- 4-6: 一応返しているが説得力・共感が不足
- 7-9: 共感→理解→解決の流れで丁寧に処理
- 10: 反論をむしろ購入理由に変換できた

### クロージング（closing）
- 0-3: クロージングを試みていない、または完全に失敗
- 4-6: 試みているが弱い。タイミングや手法が不適切
- 7-9: 適切なタイミングで決断を促し次のアクションを合意
- 10: 顧客が自ら「お願いします」と言うクロージング

### コミュニケーション（communication）
- 0-3: 一方的で話す比率が90%以上。専門用語多用
- 4-6: やや一方的（70-80%）。時々顧客の話を遮断
- 7-9: 60:40程度。顧客の話をよく聞き相槌が自然
- 10: 理想の40:60。顧客が中心の会話設計

## スコアリングの注意事項
- 文字起こしの質（音声認識の精度）を考慮して採点する
- 発言が確認できない項目は中間点（5点）を基本とする
- 全体スコアは単純平均ではなく商談全体の質を総合評価する
  （ヒアリング25%・クロージング20%・反論処理20%・提案力20%・ラポール10%・コミュニケーション5%の重み）"""

ANALYSIS_TOOL = {
    "name": "submit_analysis",
    "description": "商談録音の分析結果を構造化されたデータとして提出します",
    "input_schema": {
        "type": "object",
        "required": [
            "overall_score", "grade", "summary",
            "category_scores", "strengths", "improvements",
            "knowledge_topics", "objection_scripts"
        ],
        "properties": {
            "overall_score": {
                "type": "integer",
                "description": "総合評価スコア（0〜100点）",
                "minimum": 0, "maximum": 100
            },
            "grade": {
                "type": "string",
                "enum": ["S", "A", "B", "C", "D"],
                "description": "S=85+, A=70-84, B=55-69, C=40-54, D=0-39"
            },
            "summary": {
                "type": "string",
                "description": "この商談の総評（150〜250字）"
            },
            "category_scores": {
                "type": "object",
                "required": ["rapport", "hearing", "proposal", "objection_handling", "closing", "communication"],
                "properties": {
                    "rapport": {
                        "type": "object",
                        "required": ["score", "comment", "evidence"],
                        "properties": {
                            "score": {"type": "integer", "minimum": 0, "maximum": 10},
                            "comment": {"type": "string", "description": "50字程度の評価コメント"},
                            "evidence": {"type": "string", "description": "スコアの根拠となる具体的な発言の引用（なければ空文字）"}
                        }
                    },
                    "hearing": {
                        "type": "object",
                        "required": ["score", "comment", "evidence"],
                        "properties": {
                            "score": {"type": "integer", "minimum": 0, "maximum": 10},
                            "comment": {"type": "string"},
                            "evidence": {"type": "string"}
                        }
                    },
                    "proposal": {
                        "type": "object",
                        "required": ["score", "comment", "evidence"],
                        "properties": {
                            "score": {"type": "integer", "minimum": 0, "maximum": 10},
                            "comment": {"type": "string"},
                            "evidence": {"type": "string"}
                        }
                    },
                    "objection_handling": {
                        "type": "object",
                        "required": ["score", "comment", "evidence"],
                        "properties": {
                            "score": {"type": "integer", "minimum": 0, "maximum": 10},
                            "comment": {"type": "string"},
                            "evidence": {"type": "string"}
                        }
                    },
                    "closing": {
                        "type": "object",
                        "required": ["score", "comment", "evidence"],
                        "properties": {
                            "score": {"type": "integer", "minimum": 0, "maximum": 10},
                            "comment": {"type": "string"},
                            "evidence": {"type": "string"}
                        }
                    },
                    "communication": {
                        "type": "object",
                        "required": ["score", "comment", "evidence"],
                        "properties": {
                            "score": {"type": "integer", "minimum": 0, "maximum": 10},
                            "comment": {"type": "string"},
                            "evidence": {"type": "string"}
                        }
                    }
                }
            },
            "strengths": {
                "type": "array",
                "description": "評価できる点（2〜4点）",
                "items": {"type": "string"},
                "minItems": 1, "maxItems": 5
            },
            "improvements": {
                "type": "array",
                "description": "優先度順の改善点（3〜5点）",
                "items": {
                    "type": "object",
                    "required": ["issue", "detail", "evidence", "how_to_fix", "better_script"],
                    "properties": {
                        "issue": {"type": "string", "description": "問題点のタイトル（20字以内）"},
                        "detail": {"type": "string", "description": "問題の詳細説明"},
                        "evidence": {"type": "string", "description": "問題が起きた場面の発言引用"},
                        "how_to_fix": {"type": "string", "description": "具体的な改善方法"},
                        "better_script": {"type": "string", "description": "改善後の理想的な発言例（セリフ形式）"}
                    }
                },
                "minItems": 2, "maxItems": 6
            },
            "knowledge_topics": {
                "type": "array",
                "description": "成約率向上のために学ぶべき知識・スキル（2〜4点）",
                "items": {
                    "type": "object",
                    "required": ["topic", "why_needed", "key_points", "recommended_action"],
                    "properties": {
                        "topic": {"type": "string", "description": "学ぶべきトピック名"},
                        "why_needed": {"type": "string", "description": "なぜ今この商談で必要だったか"},
                        "key_points": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "押さえるべきポイント（3〜5点）"
                        },
                        "recommended_action": {"type": "string", "description": "具体的な学習アクション"}
                    }
                },
                "minItems": 2, "maxItems": 5
            },
            "objection_scripts": {
                "type": "array",
                "description": "この商談で出た反論・懸念への推奨返し（1〜5点）",
                "items": {
                    "type": "object",
                    "required": ["objection", "actual_response", "better_response", "why_better"],
                    "properties": {
                        "objection": {"type": "string", "description": "顧客の反論・懸念（引用または要約）"},
                        "actual_response": {"type": "string", "description": "実際の返し方（引用または要約）"},
                        "better_response": {"type": "string", "description": "より効果的な返し方（セリフ形式）"},
                        "why_better": {"type": "string", "description": "なぜこの返しが効果的か"}
                    }
                },
                "minItems": 1, "maxItems": 6
            }
        }
    }
}


def analyze_transcript(transcript_text: str, file_name: str) -> dict:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        tools=[ANALYSIS_TOOL],
        tool_choice={"type": "tool", "name": "submit_analysis"},
        messages=[{
            "role": "user",
            "content": (
                f"以下の商談録音「{file_name}」を詳細に分析し、"
                f"スコアと具体的なフィードバックを提出してください。\n\n"
                f"---\n{transcript_text}\n---"
            ),
        }],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_analysis":
            return block.input

    raise RuntimeError("分析結果を取得できませんでした")
