# 実装計画と受け入れ条件

各Phaseは前段の永続形式を壊さず追加します。単体・ローカル統合検証は実行済みです。開発端末では実Codexログインを使ったTurn/resumeも確認しました。実ホスト、Discord Gateway、GPU利用、配備後の再起動や運用データを使う受け入れ試験は未実施です。

## Phase 1：基礎設計とSSH実行基盤

状態：オフライン実装とローカル検証を完了。実GPUサーバーへの疎通、配備先での再起動・終了状態照合は未実施。

実装：OpenSSH Host解決、TOML環境定義、拡張登録可能なExecutionAdapter ProtocolとDocker/uv/venv/native adapter、SQLiteスキーマ、`db-backup`によるOnline Backup API経由の整合snapshot、integrity check、0600作成、既存ファイル非上書き、ジョブID単位のDetached Supervisor、stdout/stderr、終了コード、状態不明、再照会。GPU割当後に実行環境のGit SHAと追跡ファイルのdirty状態をbest-effort取得し、job/attemptに保存。Git情報を取得できなくても起動は継続します。独自adapterの登録・設定読込・実行argvをGPUなしテストで検証。

受け入れ：不正設定は接続前に拒否。Host表示はSSH設定を変更しない。同一UUIDで二重起動しない。通信断を即時失敗とせずUNKNOWNに保つ。ジョブIDに限定した停止。偽transportで起動・照会・停止・ログ解析を検証。

## Phase 2：GPU管理とキュー

状態：実装済み、偽Probe/transportによるローカル検証済み。freshなGPUプロセス一覧から管理対象外プロセスを検知し依頼者へ通知する。Docker内プロセスはホスト側`/proc/<pid>/environ`の`LAB_CDS_JOB_ID`で照合し、読めない場合は分類を保留する。実機照合、長時間運転、systemd配備は未検証。

実装：nvidia-smiからUUID単位でスナップショット保存、プロセス一覧の不明状態を安全側で保留、連続空きポリシー、SQLite予約、起動直前再確認、優先度、依存DAG（CLI/libraryとDiscord `/depend`）、常駐ポーリング。

受け入れ：Controller管理ジョブ間でGPUを二重予約しない。GPU履歴不足、共有プロセス、空き不足、通信断、直前競合で安全に待機。外部プロセスを停止しない。

実装済み：待機ジョブのtimezone付き`not_before`時刻とDiscord `/schedule`による一回限りの遅延開始。依頼者はDiscord `/pause`・`/resume`またはCLI `job-pause`・`job-resume`で待機ジョブを一時停止・再開できます。Pauseとdispatch開始はDB transactionで競合を防ぎます。実行中ジョブもSupervisorへジョブID限定のSIGSTOP/SIGCONTを送り、一時停止中もGPU予約を維持します。Dockerはコンテナ全体へ作用させず、`LAB_CDS_JOB_ID`が一致するコンテナ内プロセスだけをsignalします。pause中の停止要求は対象をresumeしてから終了させます。Fake Supervisor、native process group、Dockerコマンドを模したローカルプロセスでpause/resume/cancelと予約保持を検証。GPU実機/Docker実環境での確認は未実施です。

実装済み：期限付きジョブ、24時間前/待機中期限切れ/実行中期限超過通知、確定失敗だけを対象とする上限付き再試行と試行別リモートUUID/履歴、最大100件のatomic条件スイープ（CLI/Discord）。systemd user serviceの雛形は`docs/deployment.md`にありますが、実環境への登録・再起動試験は未実施。

## Phase 3：Discord連携

状態：Bot、認可、Commands、スレッド会話処理、通知outbox worker、成果物取得、永続計画承認を実装。ローカルでClient構築を検証。Discordへの接続とGuild上のInteractions動作確認は未実施。

実装：`/status`、`/gpu`、`/queue`、`/submit`、`/sweep`、`/job`、`/depend`、`/cancel`、`/pause`、`/resume`、`/schedule`、`/logs`、`/plot`、`/tb-media`、`/artifacts`、`/artifact`、`/audio-utterances`、`/compare-audio`、`/plan`、`/report`、`/new-conversation`、`/instructions`、`/instructions-history`、`/instructions-restore`。`/new-conversation`は許可されたText/Forum Channelまたはその配下のThreadから、最初の依頼付きで新しいDiscord/Codex会話を作ります。許可IDによる認可、Message IDとSlash Command Interaction IDの永続的な重複受付防止、既存Codex会話のmarker復旧、`LAB_CDS_DISCORD_REQUIRE_MENTION`による既存Thread限定のメンション省略、受信添付の署名/Guildサイズ確認、依頼者限定の停止・承認ボタン。開始/期限/終端/日次通知はoutboxを経由して依頼者へ限定メンション。

成果物取得はジョブUUID配下、登録環境のartifact_dir、拡張子allowlist、サイズ制限、symlink拒否を適用。Checkpointは許可していません。成功通知では許可済み小型成果物を最大3件、自動添付します。

実装済み：Bot自身が投稿したEmbed footerの識別markerをThread履歴から照合し、対応を欠損DBから保守的に復旧。複数markerが競合する場合は復旧しません。Codex Threadのresumeが失敗した場合も、依頼者確認後にだけ新Threadへ再送し、DiscordイベントIDを失敗状態から一度再claimします。対応markerを更新し、DB Thread bindingは旧Codex IDとの比較更新にします。`/new-conversation`はInteraction IDをThread作成前にinboxへ記録し、再配送による二重作成を防ぎます。受信はGuildの`filesize_limit`、送信はInteractionの`attachment_size_limit`と設定上限を使います。音声は`audio_manifest.json` v1で発話ID検索し、3役のbasenameを検証して同時添付します。TensorBoard image/audioイベントは読み取り専用で抽出し、サイズ上限付きキャッシュから`/tb-media`で取得できます。outbox通知はstable nonceと60秒leaseでDiscordの短時間重複排除を使いますが、exactly-onceは保証しません。未完了：Discord Gateway結合試験。

受け入れ：User/Guild/Channel allowlistなしは拒否。曖昧なThread IDは新しいCodex会話へ接続しない。計画ボタンと停止操作は依頼者に限定。既知イベントキーを重複登録しない。実Discordへ接続せずローカルテストを通過。

## Phase 4：Codex app-server

状態：stdio JSONL JSON-RPC Client、initialize/thread start/resume/turn、streaming、Thread単位直列化、期限、承認応答、構造化実験計画を実装。偽app-serverでプロトコルテスト済み。開発環境のCodex CLI 0.162.1が生成するJSON Schemaと照合し、`thread/start.sandbox`のread-only値の不一致を修正。2026-10-21に実ログイン済みCLIでread-onlyのTurnを実行し、別app-serverプロセスで同一Threadをresumeして2つ目のTurn完了まで確認。Turn中のapp-server EOFが最長2時間待ちになる経路を修正し、即時失敗テストを追加。偽app-serverで承認要求の非同期応答も検証。Discord上のApproval UI結合と制御サーバーへの配備は未確認。

実装：会話用指示を各Turnに適用、画像/音声をlocalImage/localAudioとして送信、テキスト添付は文字数上限付きでinline、turn output schemaに沿って計画候補を作る。shellコマンド、未登録環境、環境IDだけでなくproposal時点から変更された実行環境を拒否。承認前にジョブを作らない。Codexのワークスペース書込みは既定無効、ネットワークアクセス無効。

受け入れ：Codex停止後は保存Threadをresumeする。自然言語をShellとして実行しない。Codex提案だけではGPUジョブを起動しない。ThreadごとにTurnを直列化。読み取り/書込み承認を依頼者経由で扱い、権限拡張要求を拒否する。

## Phase 5：研究ワークフロー

状態：ローカルTensorBoard解析、限定的なリモートイベント回収と常駐監視、期限・再試行、依存ジョブ、日次レポート、発話ID音声manifestを実装。偽SSH transportとsynthetic event fileで検証。実ホストでのパス・帯域確認は未実施。

実装済みコマンド：`analyze <logdir>`、`plot --run NAME=LOGDIR ...`。`analysis` extraでTensorBoardとmatplotlibを導入します。集計と図生成でCodexを呼びません。

期限通知は依存する後続ジョブを最大12件の作業リストとして表示します。同一環境・設定・UUID正規化argvの成功実行が3件以上ある場合、中央値から依存クリティカルパスの計算時間を推定し、期限までの残時間と比較します。GPU待機・共有競合は含めず、履歴不足では予測しません。依存ジョブ100件超では全体予測を省略します。`propose_next_after_evaluation=true`を明示した環境では、評価成功後にProposal Serviceが成功状態・評価DAG・依頼者を再検証し、Codexが次実験案を1件生成して依頼者へ承認ボタン付きで提示します。提案元evaluation job IDに一意に紐付き、承認前にはキュー登録しません。Codex提案の自動承認はありません。偽CodexとSQLiteでThread再利用、依頼者照合、重複処理、承認前の未登録、承認後の登録を検証します。環境ごとの`validation_tags`と`/plot`引数上書きに対応。環境に`evaluation_command`を明示した場合、成功したexperimentに対する評価ジョブを永続的に一度だけ自動生成します。`/tb-media`はtag候補表示と指定tagの最新media添付に対応します。条件スイープはCLI `enqueue-sweep`とDiscord `/sweep`で最大100件まで登録できます。評価コマンドは、環境設定による自動生成か、`/submit`またはCLI `enqueue --depends-on`による手動登録を選べます。日次レポートは`LAB_CDS_DAILY_REPORT_TIME=HH:MM`（任意で`LAB_CDS_REPORT_TIMEZONE`）で有効化します。TensorBoard回収は環境の`log_dir/<job UUID>`にある小さなevent fileに限定し、1ファイル16 MiB、1 polling cycle 32 MiBまでです。学習コマンドは`LAB_CDS_TENSORBOARD_LOGDIR`をwriterの保存先として使い、Dockerでは同一絶対パスをGPUホストから読めるvolumeへbind mountする必要があります。ログ更新が止まっただけで学習停止とは判断しません。

## 次の運用前作業

1. 実ホストのOpenSSH aliasと環境定義を設定し、GPUなしの管理用コマンドから疎通を確認。
2. Discord Botのtoken、Message Content Intent、User/Guild/Channel allowlistを設定し、検証用GuildでCommandを同期。
3. 制御サーバーのCodex CLIを認証し、read-only状態でapp-serverを接続確認。
4. GPU取得とDiscord停止操作は、専用の検証ジョブと小さい空き閾値で運用者が確認。
5. 手動のDB snapshot機能はある。定期実行、別媒体への複製、保持期間、復元訓練、障害通知を運用要件に合わせて確定してから常時稼働へ移す。

Slurm、Kubernetes、Redis、PostgreSQL、GitHub Issues連携は、SQLite単一制御ノードの制約が確認されるまで導入しません。
