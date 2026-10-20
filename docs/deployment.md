# 制御サーバーへの配備例

この雛形はLinuxのsystemd user service向けです。パス、Unix user、環境ファイルは配備先で編集してください。ユニットの起動・有効化はこのリポジトリでは行いません。

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
