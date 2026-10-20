# Lab Codex Scheduler

## 概要

研究用ML実験の決定的な実行制御を行うPythonサービスです。Codex app-serverは推論、コード理解、分析が必要な操作に限って利用します。

## 構成

- `src/lab_codex_scheduler/`：Controller、Discord adapter、Codex app-server client、解析CLI。
- `config/`：編集用の例。実際の秘密情報や運用設定をコミットしない。
- `docs/`：アーキテクチャと実装段階。
- `tests/`：外部GPUや実サーバーを使わない検証。

詳細設計は `docs/architecture.md` を参照してください。

## 開発

- Python 3.13以降、uvを使う。
- `uv sync --dev --extra analysis` で完全な開発依存を準備する。軽量構成では `uv sync --dev` も使えるが、TensorBoard統合テストはskipする。
- `uv run --extra analysis python -m unittest discover -s tests -v` で全テストを検証する。
- 外部サーバーへ接続するテストは追加しない。SSH層を注入可能にし、実ネットワークに依存させない。

## 安全な変更

- SSH接続先はユーザーがOpenSSH設定に登録する。ControllerやCodexが自動追加・変更しない。
- シークレット、研究データ、SQLite運用DB、実験成果物をコミットしない。
- 他ユーザーのプロセスやジョブに作用させない。
- リモート実行、停止、上書き、削除を伴う機能は対象をジョブIDで限定する。
- 仕様や受け入れ条件が変わった場合は関連する設計文書を同じ変更で更新する。
