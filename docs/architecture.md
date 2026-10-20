# アーキテクチャ設計

更新日：2026-10-21

## 設計方針

通常の監視と制御を**Deterministic Controller**が担い、Codexは文章理解や原因分析が要るときだけ呼び出します。Controllerは入力を検証し、永続状態を更新し、決定的な処理を実行します。自然言語をShellへ直接渡さず、Codexが提案する実験仕様をスキーマ検証してからControllerのコマンドへ変換します。

制御サーバーは単一プロセスのPythonサービスから始め、SQLiteを状態の正本にします。複数ワーカーへの水平分割は初期構成に含めません。SQLiteのトランザクションと一意制約で重複受付を抑え、外部操作とDB更新の境界ではリカバリー可能な状態遷移を使います。

## コンポーネント

| コンポーネント | 責務 | 主な境界 |
| --- | --- | --- |
| Discord Adapter | Slash Command、メンション、スレッド、添付、通知 | Discordイベントを認可済みの内部Commandに変換 |
| Application Service | 権限確認、入力検証、ユースケース調整 | DiscordやCodex SDKに依存しない |
| Proposal Service | 完了済み実験の確認、Codex次案、構造化検証、提案の重複防止 | Discord送信・承認UIを持たず、依頼者・evaluation DAG・環境opt-inをDBから再検証 |
| Scheduler | 優先度、依存DAG、予約、期限、再試行、待機ジョブの一時停止 | DB上の状態遷移をトランザクションで処理 |
| Resource Monitor | SSH越しのGPUスナップショットと履歴 | `nvidia-smi` JSON/CSVを正規化。NVMLは必要時に追加 |
| Environment Adapter | Docker、uv、venv、nativeのコマンド構築 | 型付きargvと環境変数を受け、任意Shell文字列を受けない |
| Remote Job Supervisor | SSH経由の開始、照会、停止、ログ取得 | リモートのジョブID付きラッパーとローカルDBを照合 |
| Metrics / Artifacts | TensorBoard、評価、グラフ、成果物索引 | 通常解析はPython処理。大きなファイルを自動転送しない |
| Codex Adapter | app-server Thread/Turn、ストリーム、承認 | JSON-RPC stdioをプロトコル境界とし、Controller機能は限定公開 |
| SQLite Repository | ジョブ、イベント、予約、スナップショット、Thread対応 | WAL、外部キー、スキーマバージョンを使用 |

## SSH接続先

接続先はユーザーが `~/.ssh/config` または `ssh_config` 互換の専用ファイルへ手動登録します。ControllerはOpenSSHの `ssh -G <alias>` を使い、Host別名の解決結果を取得します。接続・ProxyJump・IdentityFile・known_hostsの意味を独自実装しないためです。さらに環境定義にHost別名を明示し、実験計画が未登録の別名を指定した場合は拒否します。

CLIにはローカル設定を表示する `hosts` と、ユーザーが指定した接続先だけへ `ssh -T` する `host-test` を設けます。Host鍵検証を無効化するオプションは設けません。接続先をCodexが追加・変更する経路は作りません。

## 実行環境定義

TOMLを採用します。人が編集しやすく、Python標準ライブラリで厳密に読み取れるため、Phase 1の設定処理に追加runtime依存が要りません。環境定義は接続先と分離し、環境ごとにHost、方式、作業ディレクトリ、argv形式のコマンド接頭辞、環境変数、GPU可視化方式、ログ先、成果物先を持ちます。Docker方式だけ `container` を必須とします。

Docker、uv、venv、nativeは`ExecutionAdapter` Protocolに登録します。各adapterが設定検証、argv構築、Supervisor用workdir、job環境変数、GPU可視化を定義します。追加adapterは`register_execution_adapter(kind, adapter)`で環境設定を読み込む前に登録できます。アダプターは例としてDockerなら `docker exec -w ... <container> ...`、uvなら設定済みの `uv run -- ...` をargvとして組み立てます。環境変数の補間やShell評価は許可リスト方式で扱い、ユーザー指定のargvを `shell=True` で実行しません。

Dockerの`workdir`はコンテナ内パスです。リモートSupervisor自体はホスト上で実行するため、ホストの`cwd`には`/`を使い、コンテナ内の作業ディレクトリは`docker exec -w`だけで指定します。

## リモートジョブの生存性

SSHセッション内で学習プロセスを直接実行しません。リモートの軽量SupervisorがジョブIDごとに `state.json`、stdout/stderr、終了コードを保管し、独立したセッションで学習を起動します。開始は冪等にし、同じジョブIDで既存状態を見つけたら二つ目を起動しません。Controller再起動後はDBのジョブIDでSupervisorへ照会します。

通信断は `UNKNOWN` として扱い、即座に失敗や再試行へ遷移させません。再接続後にリモート状態、PIDのプロセス識別情報、終了コードを照合します。native/uv/venvはSupervisorが作成したジョブID固有のプロセスグループを操作します。Dockerはコンテナ全体をpause/停止せず、`LAB_CDS_JOB_ID`環境変数が一致するコンテナ内プロセスだけを操作します。Supervisor上の状態とGPU予約はpause中も保持し、resume後に学習を続けます。GPUサーバーに軽量常駐Agentを置けない場合に備え、SSHで呼ぶ単一のPythonラッパー方式を初期案とします。

## SQLiteの主要データ

- `jobs`：ID、状態、依頼者、優先度、環境ID、argv、設定ハッシュ、現在試行のコードRevision、選択GPU、開始・終了時刻、終了コード、再試行回数、待機/実行中のpause flag。
- `job_attempts`：試行別remote UUID、状態、選択GPU、コードRevision、開始・終了時刻、終了コード。
- `job_dependencies`：有向辺。挿入時に循環を拒否します。
- `experiment_plans`：Codexの構造化提案、依頼者、実行環境、承認状態、発行したジョブID。自動次案は元evaluation job IDを一意に保持。
- `job_events`：状態遷移と外部イベントの監査履歴。
- `gpu_snapshots`：Host、GPU UUID、型式、VRAM、利用率、プロセス、観測時刻。
- `discord_sessions`：Guild、親Channel、Discord Thread、Codex Thread、開始ユーザー、最終時刻、設定ハッシュ、状態。
- `discord_inbox`：Discord Message IDによる重複受付防止と処理結果。
- `discord_interactions`：Slash Command Interaction IDの一意受付。再配送時に`/submit`などの操作を二重実行しません。
- `notifications`：一意イベントキー、送信先、本文、配送状態、lease時刻。障害後にlease期限切れを再試行します。

待機ジョブのpauseは`jobs.paused`に保存し、dispatch候補から除外します。実行中ジョブは依頼者のDiscord `/pause`・`/resume`またはCLI操作で、管理対象プロセスだけへSIGSTOP/SIGCONTを送ります。実行中のpauseではGPU予約を保持するため、他ジョブへ同じGPUを割り当てません。Dockerは`LAB_CDS_JOB_ID`が一致するプロセスだけへsignalを送り、コンテナ全体には作用させません。停止操作もpause中のジョブをresumeしてから管理対象プロセスを終了します。

SQLiteは初期の単一制御ノードで十分です。DBの書込みロック競合が継続する、または複数制御ノードが必要になった段階でPostgreSQLを再評価します。

状態DBは`db-backup`でSQLite Online Backup APIから整合したsnapshotを作成し、`db-restore`でintegrity/schema version/必須テーブル/外部キーを検証したsnapshotを別の新規ファイルへ復元します。両コマンドは出力を上書きしません。運用DBへの切替はservice停止と現行DB退避を伴うため、CLIでは自動化せず運用者が手順に沿って行います。

## GPU選択と共有GPU

GPU UUIDを永続識別子にします。NVIDIAの列挙順は再起動で変わり得るため、indexを識別子として保存しません。`nvidia-smi`でGPU一覧と使用量を取得し、プロセス照会はUUIDを指定してGPUごとに行います。プロセス照会に失敗したGPUは状態不明として割当対象から外します。選択条件は要求型式、必要VRAM、GPU使用率、VRAM空き量、連続空き観測、Controller内の予約を含む設定可能なポリシーにします。予約取得後、起動直前に再照会します。

自システムの予約は他ユーザーの排他を意味しません。起動後にもプロセスと空き量を監視し、競合を通知します。他ユーザープロセスを終了・優先度変更する機能は持たせません。nvidia-smiの値が取得できない、MIG/vGPU識別が曖昧な場合は自動割当を保留します。競合検知では管理ジョブのプロセス系譜に加えて`LAB_CDS_JOB_ID`を照合し、Docker内PIDの`/proc`情報を読めない場合は誤分類を避けて通知を保留します。

ジョブ起動直前にSupervisorが実作業ディレクトリで`git rev-parse HEAD`と追跡ファイルのdirty状態を読み、SHAと任意の`-dirty`をjobおよびattemptへ保存します。Dockerでは登録コンテナ内の登録workdirを対象にします。Git repositoryでない場合、Gitがない場合、通信が失敗した場合はrevisionをnullとして起動を継続します。

## Discord会話とCodex Thread

正本はSQLiteの `discord_sessions` です。新規メンションでは通常テキストChannelにスレッドを作成してからCodex `thread/start` を行います。Forumでは投稿ThreadのIDを会話キーにします。既存Threadのメンションは保存済み対応を優先し、Codex Threadをresumeします。

DB対応がないスレッドは別のCodex会話へ自動接続しません。Botが投稿した識別Embed markerが履歴に見つかる場合だけ対応を復旧し、複数markerなど曖昧な場合は復旧しません。markerのない旧スレッド、または保存したCodex Threadをapp-serverが再開できない場合は、依頼者限定の確認ボタンから新Threadを作り、元の依頼を再送できます。Threadを付け替える際はDB bindingを旧IDとの比較更新にし、Bot自身のmarkerも更新して履歴上のID競合を避けます。再送は失敗扱いのDiscord Message IDだけを明示確認後に一度再claimします。Forum投稿を含むDiscord Gatewayの実イベント結合試験は未実施です。

長期会話を手動で切り替える`/new-conversation title prompt`を用意します。Text/Forum Channelから新規Threadを作成し、既存Thread内から実行した場合は親Channelに新しいThreadを作ります。Discord Interaction IDをThread作成前にinboxへ記録し、再配送では二つ目のThreadを作りません。指定された初回依頼と新しいCodex Threadの対応をSQLiteへ保存してからTurnを開始します。

メンション必須は既定です。`LAB_CDS_DISCORD_REQUIRE_MENTION=false`の場合も、DBにCodex対応を持つ既存Discord Threadだけを対象とし、未登録ThreadやForum投稿には自動参加しません。TurnはCodex Threadごとに直列化します。Discord Message IDを一意キーにし、Thread lockを待つ前に`discord_inbox`へ記録します。Bot起動後は未処理messageをDiscordから再取得し、依頼者・Guild・Channelを再検証して順番に処理します。イベントは外部操作を始める前に`processing`へ移します。再起動時に`processing`だった依頼は処理結果不明として依頼者へ通知し、重複処理を避けるため自動再送しません。Slash CommandはInteraction ID一意制約でも重複処理を拒否します。Codex app-serverはstdioのJSONL JSON-RPCを初期transportにします。Thread IDを永続化し、再起動後に `thread/resume` を呼びます。app-server仕様変更は実装時にプロトコル版と公式スキーマを再確認します。

## Discord通知と添付

ジョブ受付など同期応答はコマンド発行者へ返します。ジョブ完了、異常、GPU競合、承認待ちなどはoutboxへ一度記録し、依頼者が判明するときだけユーザーをメンションします。小さな進捗更新にはメンションを付けません。通知先はジョブ依頼者、会話開始者、明示担当者の順で解決し、曖昧な場合は全員通知しません。Discord送信にはoutbox event key由来の安定nonceを渡し、短時間の再試行をサーバー側で重複排除します。Codex次実験提案も提案元evaluation ID由来のnonceを再利用します。送信中leaseは60秒で回収します。Discord APIのnonce重複確認は直近数分に限られるため、長時間停止後などのexactly-onceは保証できません。[Create Message API](https://docs.discord.com/developers/resources/message#create-message)。discord.py 2.7.1はnonce指定時に`enforce_nonce`を送ります。

フォーラム投稿作成と通常Channelのスレッド作成はDiscord APIの異なる形式に従います。添付上限は受信ではGuildの`filesize_limit`、Interaction送信では`attachment_size_limit`を優先し、上限を超えるファイルは変換せず拒否します。成功通知ではallowlist内の画像・音声・CSV・JSON・Markdownから最大3件を合計16 MiB（さらにGuild上限で制限）以内で添付します。Checkpointとログ全体は自動添付しません。受信ファイルはサイズ、拡張子、MIME、ファイル名を検査して専用inboxへ保存し、Codexへは安全な作業領域のファイル参照だけを渡します。

現行の成果物取得は、登録環境で設定した絶対artifact_dirの下に `<job UUID>/` を作る配置を対象とします。remote helperはUUIDとbasenameを受け取り、symlink、許可外suffix、最大サイズ超過を拒否します。`audio_manifest.json` v1は発話IDごとにGround Truth/Baseline/Generatedのbasenameを指定し、Discord `/audio-utterances`と`/compare-audio`で検索・取得できます。TensorBoard image/audioイベントはEventAccumulatorから対応形式の原bytesをサイズ制限付きで抽出し、ローカルcacheからDiscord `/tb-media`でtag候補表示・添付取得できます。変換は行いません。

## 会話用AGENTS.md

開発用 `AGENTS.md` はリポジトリ直下に置きます。Discord会話指示のユーザー編集版は `~/.config/lab-codex-scheduler/conversation/AGENTS.md` に配置し、SQLiteやアプリ更新と分離します。初回起動時に同梱テンプレートを `O_EXCL` 相当で一度だけ作成し、既存ファイルがあれば内容に触れません。

更新時は現在版を日時付き履歴へコピーしてから置換します。Discordの `/instructions show` と `/instructions history` は内容と保存済み履歴を返し、`/instructions restore <version>` は復元前の版も履歴へ保存します。Codex Thread開始時に内容を明示的な開発者指示として読み込み、適用したSHA-256とversionを会話対応情報へ記録します。編集は次のTurnから有効にし、進行中Turnには差し替えません。

## 承認境界

事前承認なしで行えるのは、登録済みジョブの起動、事前承認済み範囲での空き待ち、評価、グラフ生成、レポートです。新規実験計画、コード変更、大規模GPU使用、学習打切り、既存成果物の上書きや削除はDiscord承認イベントを必要とします。提案内容、対象、推定時間、対象成果物を承認画面で確認できるようにします。

現行実装ではCodexワークスペース書込みは既定で無効です。明示して有効化した場合も、File Change approvalではapp-serverのpatchUpdated通知からitem IDで差分を結び、差分を提示できない場合、変更先が許可Workspace外の場合、差分添付上限を超える場合は拒否します。Command approvalでは実行コマンド、cwd、追加権限要求を表示します。Permission profile要求は現在のBot UIでは承認せず拒否します。

実験環境の`propose_next_after_evaluation`は既定でfalseです。trueにした場合だけ、正常終了した評価ジョブから保存済みDiscord/Codex Threadへ次案生成を依頼します。Proposal Serviceは依頼者、成功状態、評価DAG、同一環境、opt-inを再検証します。Codexにはジョブ状態と上限付きTensorBoard scalar要約を渡し、生ログやコマンド引数は渡しません。提案元evaluation job IDに一意制約を置き、pending計画は既存の依頼者限定承認UIで扱います。提案は自動でキューへ入らず、再起動後も同じ提案を再表示します。分析要求が計画保存前に失敗した場合、Codex Turn自体が再試行される可能性はありますが、重複計画はDBで拒否されます。偽CodexとSQLiteを使い、Thread再利用、依頼者照合、重複イベント、承認前の未登録、承認後の登録をテストします。

## 現行仕様の根拠

2026-10-21に確認した公式資料を、各プロトコル境界の判断に使います。実装時にもバージョンと仕様を再確認します。

- Codex app-serverはstdio上のJSONLメッセージを使い、initialize後にThreadを開始し、保存したThread IDをresumeできます。現行CLIが生成したJSON Schemaと実際の`turn/start`、プロセス再起動後の`thread/resume`を照合しました。[OpenAI Codex app-server連携ガイド](https://developers.openai.com/siwc/token-sharing-open-source/codex-app-server)、[Codex app-server README](https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md)
- 現行app-serverのTurn入力型はtext、localImage、localAudioを含み、Turn開始にはJSON Schemaによる出力制約があります。[Codex app-server Turn protocol](https://github.com/openai/codex/blob/main/codex-rs/app-server-protocol/src/protocol/v2/turn.rs)
- Discord Slash CommandはCHAT_INPUTのApplication Commandであり、Forum投稿作成は通常のChannel thread作成とは別のAPI形式です。[Application Commands](https://docs.discord.com/developers/interactions/application-commands)、[Channel resource](https://docs.discord.com/developers/resources/channel)、[Threads](https://docs.discord.com/developers/topics/threads)
- Discord添付上限は利用状況で変わり、Interactionには `attachment_size_limit` が渡されます。[Discord API reference](https://docs.discord.com/developers/reference)
- NVIDIAはGPU UUIDを一意識別子として示し、nvidia-smiのquery形式からGPUとcompute app情報を取得できます。[nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
- Dockerの `exec` は起動中コンテナに新しいプロセスを実行し、`--workdir` と環境変数を指定できます。[Docker exec](https://docs.docker.com/reference/cli/docker/container/exec)
- `uv run` はプロジェクト環境を準備してコマンドを実行し、引数は `--` で区切れます。[uv run](https://docs.astral.sh/uv/concepts/projects/run/)
- TensorBoard EventAccumulatorはscalarとtensorを別のSummary型として保持し、タグ別に取得できます。Scalarの読込はイベント形式の違いを考慮します。[TensorBoard EventAccumulator](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/backend/event_processing/event_accumulator.py)、[PyTorch TensorBoard](https://docs.pytorch.org/docs/stable/tensorboard.html)
- OpenSSHの `ssh -G` と `ssh_config` を設定解決の基準にします。[OpenSSH ssh(1)](https://man.openbsd.org/ssh.1)、[OpenSSH ssh_config(5)](https://man.openbsd.org/ssh_config.5)

## 未決事項

- 制御サーバーのOSユーザーとsystemd unit名、バックアップ先。
- Discordの許可Guild、Channel、User/Role IDと通知Channel。
- GPU空き判定の既定閾値と連続観測時間。
- 各GPUホストでSupervisorに利用できるPythonの最小バージョン。
- 研究データの保持期間、添付ファイルの容量上限、個人情報を含む音声の扱い。
- Codex app-server認証方式とモデル提供元。既存のCodex CLIログインを使うかAPI credentialを使うかは運用環境確認後に決めます。

## 実装状況と未接続境界

Controller、GPUキュー、SQLite、試行履歴付き期限/再試行、依存ジョブ登録、CLI、Discord Bot、Codex app-server JSONL client、承認後の計画登録、開始/期限/終端/日次通知、bounded artifact fetch、発話単位音声manifest、TensorBoard event fileの限定的なリモート同期と常駐scalar監視、EventAccumulator解析・図生成を実装しました。テストは偽SSH transport、偽app-server、synthetic TensorBoard event fileで実行します。

開発端末では実Codexログインを使ったモデル応答、app-server再起動後のThread resumeを確認しました。実Discord Gateway、Discord Approval UI、実SSH Host/GPU、Dockerコンテナ、長時間連続稼働への接続確認はしていません。期限通知は依存DAGの後続ジョブを最大12件表示します。同一環境・設定・UUID正規化argvの成功履歴が3件以上あるジョブについて、中央値から依存クリティカルパスの計算時間を見積もります。GPU待機・競合は予測対象外で、履歴不足の場合は見込みを出しません。Codexの次案生成は環境ごとの明示opt-inで評価完了後に動作し、提案元evaluation IDに一意に紐付き、承認前にジョブを作りません。次案の自動採用は未実装です。条件スイープはCLI/Discordから最大100件を原子的に登録できます。TensorBoard image/audioは原形式のままサイズ制限付きでDiscord取得できます。環境に任意の`evaluation_command` argvテンプレートを設定すると、成功したexperimentから一度だけ評価ジョブを作成します。親成功状態と追跡行をSQLite transactionで照合し、再起動後も重複しません。Discord履歴markerによる復旧はBotが投稿したmarkerに限られ、旧スレッドの曖昧な対応は復旧せず、依頼者確認ボタンで元の依頼を新しいCodex Threadへ送ります。TensorBoard同期は環境の`log_dir/<job UUID>`のみを読み、1ファイル16 MiB・1 cycle 32 MiBを上限とします。
