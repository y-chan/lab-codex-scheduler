# 制御サーバーへの配備例

この雛形はLinuxのsystemd user service向けです。パス、Unix user、環境ファイルは配備先で編集してください。ユニットの起動・有効化はこのリポジトリでは行いません。

## 起動前の設定

リポジトリの配置先で依存を準備します。実行環境設定はサンプルをコピーして、ユーザーが登録したOpenSSH aliasと実際の作業ディレクトリに合わせます。`config-check`は構文・値を検証し、`host-test`は指定Hostへ実際にSSH接続します。

```sh
install -d -m 700 ~/.config/lab-codex-scheduler
install -m 600 config/environments.example.toml ~/.config/lab-codex-scheduler/environments.toml
uv sync --locked --extra analysis
uv run --locked lab-codex-scheduler config-check ~/.config/lab-codex-scheduler/environments.toml
uv run --locked lab-codex-scheduler hosts
uv run --locked lab-codex-scheduler host-test <registered-host-alias>
uv run --locked lab-codex-scheduler db-init
```

`host-test`の前に、aliasが対象ホストを指すこととknown_hostsに登録済みであることを確認してください。`gpu`はNVIDIA GPU情報をリモート取得する読み取り操作です。GPU予約や学習の起動は行いません。

Discord用の`secrets.env`には`DISCORD_BOT_TOKEN`だけを保存します。User/Guild/Channel allowlistはsystemd drop-in等のアクセス制御された設定から渡します。トークンをGit、issue、会話、ログへ転記しません。

```sh
install -m 600 /dev/null ~/.config/lab-codex-scheduler/secrets.env
${EDITOR:-vi} ~/.config/lab-codex-scheduler/secrets.env
```

Discord Developer PortalではBotのMessage Content Intentを有効にし、Guildへ招待します。制御サーバーでsystemdを使うUnix userとしてCodex CLIのログインを確認し、`command -v codex`の場所がユニットの`PATH`に含まれるようにしてください。

```sh
codex login status
command -v codex
```

systemdへ登録する前に、検証用Guild、限定allowlist、GPUを使わないDiscord操作を確認してください。`gpu`や`scheduler`を使う場合は、最初に設定済みGPU Hostで状態取得だけを行い、専用の小さい検証ジョブの開始・停止は運用者が対象IDと条件を確認してから実行します。

## Scheduler

`~/.config/systemd/user/lab-codex-scheduler.service`:

```ini
[Unit]
Description=Lab Codex Scheduler GPU monitor and queue dispatcher
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/src/lab-codex-scheduler
Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin
Environment=XDG_CONFIG_HOME=%h/.config
Environment=XDG_DATA_HOME=%h/.local/share
UMask=0077
ExecStart=%h/.local/bin/uv run --locked --no-sync lab-codex-scheduler scheduler --interval-seconds 30
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

## Discord Bot

`~/.config/systemd/user/lab-codex-discord.service`:

```ini
[Unit]
Description=Lab Codex Scheduler Discord adapter
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/src/lab-codex-scheduler
Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin
Environment=XDG_CONFIG_HOME=%h/.config
Environment=XDG_DATA_HOME=%h/.local/share
EnvironmentFile=%h/.config/lab-codex-scheduler/secrets.env
UMask=0077
ExecStart=%h/.local/bin/uv run --locked --no-sync lab-codex-scheduler discord --config %h/.config/lab-codex-scheduler/environments.toml
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

`secrets.env`はownerだけが読める権限（`0600`）にし、`DISCORD_BOT_TOKEN`を設定します。許可IDなどはユニットに直接書かず、systemd drop-inまたはアクセス制限された別設定から渡します。両サービスは同じSQLite DBを使います。

TensorBoard同期・scalar監視・グラフ・media取得には`analysis` extraが必要です。配備時に一度、リポジトリで`uv sync --locked --extra analysis`を実行して依存を準備します。サービス起動時は`--no-sync`を使ってネットワーク上の依存同期を避けます。アプリ更新時は新しいlockfileとextraで明示的に再同期してからサービスを再起動してください。

```sh
systemctl --user daemon-reload
systemctl --user enable --now lab-codex-scheduler.service lab-codex-discord.service
systemctl --user status lab-codex-scheduler.service lab-codex-discord.service
```

常時稼働にはユーザーマネージャーのlinger、SSH鍵の読み取り、known_hosts登録、Codex CLI認証、systemdの書込パス/サンドボックス設定が必要です。実GPU/Discord接続を伴うため、検証Guildと検証ジョブで確認してから有効化してください。

## SQLite状態DBのバックアップ

`db-backup`はSQLite Online Backup APIで整合したスナップショットを作成し、integrity checkを通したファイルだけを保存します。出力先は既存ファイルを上書きせず、owner限定の`0600`で作成します。バックアップ先ディレクトリは事前に作成してください。

```sh
install -d -m 700 ~/.local/share/lab-codex-scheduler/backups
uv run --locked lab-codex-scheduler db-backup \
  ~/.local/share/lab-codex-scheduler/backups/scheduler-$(date +%Y%m%d-%H%M%S).sqlite3
```

このコマンドは手動スナップショットです。バックアップ先の別媒体への同期、定期実行、保持期間、復元手順は運用環境の保存要件を確認して設定してください。source DBをサービス停止なしで読むため、OS上でファイルを直接copyする方法は使いません。
