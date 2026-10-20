# Lab Codex Scheduler

研究用ML実験の登録、GPU待ち、実行、状態保存、結果確認を行うPythonサービスです。定期監視やジョブ制御は通常のPython処理で行います。Codex app-serverはDiscord上の自然言語相談、実験計画、必要な分析に限って利用します。

アーキテクチャと受け入れ状況は [`docs/architecture.md`](docs/architecture.md) と [`docs/implementation-plan.md`](docs/implementation-plan.md) を参照してください。
systemd user serviceの雛形は [`docs/deployment.md`](docs/deployment.md) にあります。

## 開発状態

Phase 1〜5のローカル実装を進めており、SSH実行基盤、GPUキュー、Discord・Codex接続コード、TensorBoard同期・解析・media取得、期限・再試行・依存ジョブ、日次通知、条件スイープ、音声manifestによる発話比較を含みます。偽SSH transport、偽app-server、SQLite、synthetic TensorBoard event fileで検証します。開発端末では実Codexログインによるモデル応答とapp-server再起動後のThread resumeも確認済みです。Discord Gateway、実SSH/GPUホストでの結合・配備試験は未実施です。評価ジョブは環境の`evaluation_command`で成功後に自動登録でき、期限通知は依存する後続ジョブを作業リストとして表示します。同一環境・設定・UUID正規化argvの成功履歴が3件以上ある場合、中央値から依存経路の計算時間を推定します。GPU待ち時間は含めず、履歴不足時は予測しません。依存ジョブが100件を超える場合は安全側で全体予測を省略します。Discord履歴からの会話復旧はBotが投稿した識別markerがある場合に限り、Forum Gatewayの結合試験は未実施です。

## セットアップ

Python 3.13以降とuvを用意します。

```sh
uv sync --dev --extra analysis
uv run python -m unittest discover -s tests -v
```

TensorBoard解析・グラフ・media機能を使わない場合は、軽量構成として`uv sync --dev`だけでも起動できます。ただし、その構成ではTensorBoard依存を使うテストはskipされます。

```sh
uv sync --dev
```

Codex連携は制御サーバー上のCodex CLI `codex app-server --listen stdio://` を起動します。Codex CLIのインストールとログインは別途必要です。Codexの編集権限は既定で読み取り専用です。ワークスペース書き込みを明示的に有効化する場合は `LAB_CDS_CODEX_WORKSPACE_WRITE=1` と `LAB_CDS_CODEX_WORKDIR` を設定します。ネットワークアクセスはTurn sandboxで無効です。

## 接続先と環境

OpenSSHの `~/.ssh/config` に接続先を手動登録し、[`config/environments.example.toml`](config/environments.example.toml) を `~/.config/lab-codex-scheduler/environments.toml` へコピーして編集します。ControllerやCodexはSSH接続先を作成・変更しません。環境定義にはHost別名、Docker/uv/venv/native、作業ディレクトリ、環境変数、GPU識別対応、ログ・成果物ディレクトリと任意の`validation_tags`を登録します。Discord `/plot`はタグ引数を省略すると`validation_tags`を使い、タグ引数を指定した場合はその値で上書きします。

成功した実験の後に評価を自動登録する場合は、環境へ`evaluation_command`をargv配列で追加します。テンプレートには`{job_id}`と、設定済みの場合に限り`{artifact_dir}`・`{log_dir}`を使えます。評価ジョブは親実験のGPU要求・依頼者・通知先を継承し、親成功後に通常のGPU待ちと直前確認を経て起動します。設定が未指定なら自動評価は行いません。

Codexによる次実験の提案は既定で無効です。環境に`propose_next_after_evaluation = true`を明示すると、評価成功後にだけCodexを呼び出し、TensorBoard scalarの要約とジョブ状態を使って次案を作ります。提案は依頼者へのメンションと承認ボタン付きで届き、承認前はジョブ登録も実行もしません。同じ評価ジョブから作る提案はDBで一意にし、Bot再起動後もpending提案を復元します。保存済みのCodex会話Threadと有効なCodex app-server認証が必要です。

```sh
uv run lab-codex-scheduler config-check ~/.config/lab-codex-scheduler/environments.toml
uv run lab-codex-scheduler hosts
uv run lab-codex-scheduler db-init
```

`hosts` はローカルOpenSSH設定を表示します。`host-test`、`gpu`、`scheduler` は実ホストへ接続する操作です。GPUなしテストはネットワーク接続しません。

SQLiteのバックアップには`db-backup DESTINATION`を使います。SQLite Online Backup APIで整合性を保って複製し、integrity checkに合格した後だけ保存します。出力先が既に存在する場合は上書きしません。定期実行や保持期間は運用環境で決めてください。

## ジョブ実行

```sh
uv run lab-codex-scheduler enqueue --environment pro6000-uv \
  --model-contains "PRO 6000" --min-free-vram-gib 40 --idle-seconds 120 \
  --retry-limit 2 --retry-delay-seconds 300 --deadline-at 2026-10-21T09:00:00+09:00 -- \
  python train.py --config config.yaml
uv run lab-codex-scheduler schedule-once
uv run --extra analysis lab-codex-scheduler scheduler --interval-seconds 30
```

条件スイープJSONは`command_argv`の`{parameter}`を`parameters`の値で直積展開します。最大100件を1 transactionで登録し、全コマンドはargvのまま実行されます。

```json
{"command_argv":["python","train.py","--lr","{lr}","--seed","{seed}"],"parameters":{"lr":[0.001,0.0001],"seed":[1,2]}}
```

```sh
uv run lab-codex-scheduler enqueue-sweep --environment pro6000-uv --spec sweep.json
```

常駐`scheduler`を起動すると、登録HostのGPU状態をポーリングし、空き判定、予約、直前再確認を経てキューから最大1件ずつ起動します。共有GPUは他ユーザーによって競合し得ます。Controllerは他ユーザーのプロセスに作用しません。依存関係のあるジョブは依存先が成功するまで待ちます。依頼者はDiscord `/pause`・`/resume`またはCLI `job-pause`・`job-resume`で待機中または実行中の管理対象ジョブを一時停止・再開できます。実行中pause中もGPU予約を保持します。native/uv/venvはジョブ固有のプロセスグループ、Dockerは`LAB_CDS_JOB_ID`が一致するコンテナ内プロセスだけをsignalします。Dockerコンテナ全体には作用しません。評価コマンドを環境設定した場合、成功したexperimentから一度だけ評価ジョブを自動登録し、親ジョブ依存としてGPUキューへ送ります。`{job_id}`、`{artifact_dir}`、`{log_dir}`をargv内で利用できます。設定変更後に既存ジョブから自動評価は行いません。手動の後続ジョブはDiscord `/submit`の`depends_on_job_id`またはCLI `--depends-on JOB_UUID`でも登録できます。確定した異常終了だけを指定回数再試行し、通信断で状態不明のジョブは再実行しません。期限を過ぎた待機ジョブは起動せず、実行中ジョブは停止せず期限超過を通知します。期限通知には親ジョブと依存する後続ジョブの状態を最大12件表示します。十分な履歴がある場合は依存経路の計算時間を推定しますが、GPU待ち時間や競合は含めず、実完了時刻を保証しません。

`job-status <UUID>` と `job-cancel <UUID>` は管理対象ジョブの状態確認・停止です。`job-pause <UUID>` と `job-resume <UUID>` は待機中または実行中の一時停止・再開です。実行中pauseではGPU予約を保持します。`submit` CLIはGPUキューを通さないため、手動でリソース確認後に `--confirm-resource-check` を指定します。

## TensorBoard解析

```sh
uv run --extra analysis lab-codex-scheduler analyze /path/to/tensorboard/logdir
uv run --extra analysis lab-codex-scheduler plot \
  --run baseline=/path/to/run-a --run candidate=/path/to/run-b \
  --tag train/loss --tag validation/loss --output /path/to/compare.png
```

解析はEventAccumulatorを使い、Scalar SummaryとScalar pluginのTensor Summaryを個別に読み取ります。NaN/Infを検出し、最終値・最良値・更新停滞の候補を集計します。イベント更新停止だけでプロセス停止とは判断しません。`analyze`と`plot`はローカルのイベントファイルを対象にし、常駐schedulerは設定済み`log_dir/<job UUID>`から上限付きでイベントを同期します。

## DiscordとCodex

許可リストはfail-closedです。Bot起動には `DISCORD_BOT_TOKEN` と `LAB_CDS_DISCORD_USER_IDS` が必要です。任意でGuild/Channel IDを `LAB_CDS_DISCORD_GUILD_IDS` / `LAB_CDS_DISCORD_CHANNEL_IDS` にカンマ区切りで設定します。BotメンションにはDeveloper PortalのMessage Content Intentが必要です。`LAB_CDS_DISCORD_REQUIRE_MENTION`は既定`true`で、`false`にすると保存済みCodex会話のあるDiscordスレッド内だけメンションなしで応答します。

```sh
uv run lab-codex-scheduler discord \
  --config ~/.config/lab-codex-scheduler/environments.toml
```

利用可能な主な操作は `/status`、`/gpu`、`/queue`、`/submit`、`/sweep`、`/job`、`/depend`、`/cancel`、`/pause`、`/resume`、`/schedule`、`/logs`、`/plot`、`/tb-media`、`/artifacts`、`/artifact`、`/audio-utterances`、`/compare-audio`、`/plan`、`/report`、`/new-conversation`、`/instructions`、`/instructions-history`、`/instructions-restore` です。`/pause`と`/resume`は依頼者自身の待機中・実行中ジョブに使えます。実行中ジョブはジョブ固有のプロセスだけを一時停止し、GPU予約を保持します。`/plan` は登録済み環境から構造化案を作りますが、ボタン承認までジョブを作成しません。承認後はGPU待ちキューへ登録されます。計画ボタンはBot再起動後もpending状態から復元します。

通常Channelのメンションはスレッドを作り、登録済みDiscord ThreadとCodex Threadの対応を再利用します。Botが投稿した識別Embedから履歴復旧を試み、複数IDが見つかるなど曖昧な場合は別会話へ接続せず、元の依頼者だけが押せる確認ボタンから新しいCodex会話を開始できます。CodexのTurnはThread単位に直列化し、進捗をDiscordへ表示します。ファイル操作承認は依頼者限定UIを使い、権限拡張要求は拒否します。

長い会話を切り替えるときは、許可されたTextまたはForum Channelで `/new-conversation title prompt` を実行します。既存Thread内から実行した場合は同じ親Channelに新しいThreadを作ります。最初の依頼を指定すると、新しいDiscord ThreadとCodex Threadを作成して会話を開始します。

会話用AGENTS.mdはデータディレクトリの `conversation/AGENTS.md` に一度だけ生成されます。ユーザーが直接編集でき、初期化では既存ファイルを上書きしません。添付は専用inboxへ検証保存し、既定10 MiB上限とGuildの実ファイルサイズ上限の小さい方を適用します。成果物取得は登録環境の `artifact_dir/<job UUID>/` 配下に限定し、Checkpoint等と許可外形式、設定上限を超えるファイルは取得しません。成功通知では画像・音声・CSV・JSON・Markdownから最大3件の小さな成果物を自動添付し、容量超過や接続失敗時はテキスト通知を優先します。

Discord `/plot` はローカルへ同期済みの `LAB_CDS_TENSORBOARD_CACHE_DIR/<job UUID>/` を対象とします。常駐schedulerは`log_dir/<job UUID>`のTensorBoard event fileを読み取り専用で同期し、scalarのNaN/Infと停滞を決定的に通知します。1ファイル16 MiB、1回32 MiBが上限です。学習コードは`LAB_CDS_TENSORBOARD_LOGDIR`をwriterの保存先に使ってください。Dockerでは同じ絶対パスをGPUホストから読めるvolumeへbind mountする必要があります。 `/tb-media`はTensorBoardのImage/Audio summaryから対応形式（PNG/JPEG/WebP、WAV/MP3/OGG/FLAC）の原bytesを抽出します。`job_id`と`kind`を指定し、`tag`を省略するとtagとstepの候補を表示、指定すると最新イベントを添付します。1回の抽出は最大100ファイル・24 MiBで、過去の抽出キャッシュを置き換えて膨張を防ぎます。`/schedule` はキュー済みジョブをISO-8601の指定時刻まで待機させます。定期繰り返しには対応していません。

`/audio-utterances`と`/compare-audio utterance_id:...`は、ジョブ成果物内の`audio_manifest.json`を使います。manifestは`{"version":1,"utterances":{"utt-001":{"ground_truth":"gt.wav","baseline":"baseline.wav","generated":"generated.wav"}}}`形式で、参照音声は同じ成果物ディレクトリの別々のbasenameにします。manifest上限は1 MiBです。`/report`は手動要約、日次レポートは`LAB_CDS_DAILY_REPORT_TIME=HH:MM`と任意の`LAB_CDS_REPORT_TIMEZONE`を設定するとoutbox経由で配信します。

DiscordとCodexを起動する前に、Botの権限、許可ID、成果物パス、Codex CLI認証、ファイル保持場所を運用者が設定してください。実接続を伴う起動コマンドはこの開発中には実行していません。

## データ保護と安全境界

SQLite、イベントinbox、会話指示、通知状態はユーザーデータディレクトリに保存します。SSH鍵とBot Tokenを設定ファイルへ複製しません。ジョブ停止・ログ・成果物取得は管理対象UUIDに限定します。外部タスク管理、Slurm、Kubernetes、Redis、PostgreSQLは現段階では導入しません。
