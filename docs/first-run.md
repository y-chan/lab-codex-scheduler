# 初回設定

この手順では、制御サーバーに設定を置き、DiscordやGPUサーバーへ接続する前に読み取り確認を行います。

## どのサーバーに何を置くか

設定と常駐プロセスは、常時起動するLinux制御サーバーに置きます。
以降のシェルコマンドは、特に断りがない限り制御サーバーへSSHログインした状態で実行します。

GPUサーバーにはOpenSSHで接続できることが必要です。
GPUサーバーにはBotやSchedulerをインストールしません。

| 用意するもの | 保存場所または設定場所 | 用途 |
| --- | --- | --- |
| GPUサーバーのSSH Host aliasと鍵 | 制御サーバーの`~/.ssh/config`と`~/.ssh/` | SSH接続 |
| 実行環境の設定 | 制御サーバーの`~/.config/lab-codex-scheduler/environments.toml` | Host、Dockerまたはuv、作業場所など |
| Discord Bot token | 制御サーバーの`~/.config/lab-codex-scheduler/secrets.env` | Botのログイン |
| Discord許可ID | 制御サーバーの`~/.config/lab-codex-scheduler/discord-policy.env` | Botを使えるユーザー、サーバー、チャンネルの制限 |
| Codex CLIの認証 | 制御サーバーでBotを動かすUnixユーザーのCodex認証ストア | app-serverのログイン |

SSH秘密鍵、Discord token、Codex認証情報はチャットやGitへ貼りません。
後で設定内容の相談をするときは、Host alias、実行方式、作業ディレクトリ、コンテナ名だけを共有すれば足ります。

## 制御サーバーへ配置する

制御サーバーにPython 3.13以降、uv、OpenSSHクライアント、Gitを用意します。
uvの導入方法は[公式インストール手順](https://docs.astral.sh/uv/getting-started/installation/)を参照してください。
プライベートリポジトリをcloneするため、そのサーバーからGitHubへSSH接続できる鍵も必要です。
GPU用のSSH鍵とは別にし、GPUサーバーの秘密鍵をGitHubへ登録したり、個人PCの秘密鍵を制御サーバーへコピーしたりしないでください。
接続確認はclone前に`ssh -T git@github.com`で行います。

次の例ではリポジトリを`~/src/lab-codex-scheduler`へ配置します。
すでに別の場所へcloneしてある場合は、そのディレクトリを以降の`cd`とsystemd設定に使います。

```sh
mkdir -p ~/src
cd ~/src
git clone git@github.com:y-chan/lab-codex-scheduler.git
cd lab-codex-scheduler
uv sync --locked --extra analysis
```

Codex CLIが未導入なら、Node.js/npmを用意してから次を実行します。
Codex CLIのバージョンは`codex --version`で記録してください。
公式のnpm導入例は[OpenAI Codex CLI資料](https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex)を参照してください。

```sh
npm install -g @openai/codex@latest
codex --version
```

`uv sync`は初回に依存パッケージを取得します。
外部通信が制限されている場合は、制御サーバーからPyPIへ接続できる時間帯に実行してください。

## GPUサーバーへのSSH接続を登録する

Host aliasは制御サーバーの実行ユーザーのOpenSSH設定へ登録します。
ノートPC側の`~/.ssh/config`に登録しただけでは、制御サーバーから使えません。

GPUサーバーで許可された鍵を制御サーバー上の専用鍵として用意し、公開鍵をGPUサーバーの管理者が登録します。
接続先の指紋を既知の値と照合してから`known_hosts`へ登録してください。
`StrictHostKeyChecking=no`は設定しません。
systemdを使う場合、再起動後もSSH鍵を読める必要があります。
対話ログイン中だけ動くSSH agentに鍵を入れた構成では、ログアウトや再起動後に接続できないことがあります。
研究室の運用規則に従って、常駐agentまたは権限を絞ったサービス用鍵を用意してください。

```sh
install -d -m 700 ~/.ssh
${EDITOR:-vi} ~/.ssh/config
chmod 600 ~/.ssh/config
```

設定例です。
`HostName`、`User`、鍵の場所、踏み台を実環境に合わせます。

```sshconfig
Host gpu-a100
    HostName gpu-a100.example.internal
    User researcher
    IdentityFile ~/.ssh/id_ed25519_lab_gpu
    IdentitiesOnly yes

Host gpu-pro6000
    HostName gpu-pro6000.example.internal
    User researcher
    IdentityFile ~/.ssh/id_ed25519_lab_gpu
    IdentitiesOnly yes
    ProxyJump gateway
```

`Host`に書いた名前がHost aliasです。
`ProxyJump`もOpenSSHが処理するため、踏み台を独自設定へ書き直す必要はありません。

## 実行環境を登録する

サンプルを制御サーバーの設定ディレクトリへコピーし、実際に使う環境だけ残します。
サンプルにはDocker用とuv用の架空Hostが両方入っています。
一台だけ使う場合は、使わない環境ブロックを削除するか、Hostと全パスを実在する値へ変更してください。

```sh
install -d -m 700 ~/.config/lab-codex-scheduler
install -m 600 config/environments.example.toml \
  ~/.config/lab-codex-scheduler/environments.toml
${EDITOR:-vi} ~/.config/lab-codex-scheduler/environments.toml
```

既存の起動済みDockerコンテナを使う設定例です。

```toml
[environments.lab-docker]
host = "gpu-a100"
type = "docker"
container = "research-container"
workdir = "/workspace/project"
python = ["python"]
gpu_visibility = "cuda_visible_devices"
log_dir = "/workspace/project/runs"
artifact_dir = "/workspace/project/artifacts"
```

`host`には上で登録したHost aliasを指定します。
`container`はGPUサーバー上ですでに起動しているコンテナ名またはIDです。
`workdir`、`log_dir`、`artifact_dir`はコンテナ内の絶対パスを指定します。
TensorBoardを使う場合、`log_dir`と同じ絶対パスをGPUサーバーから読めるvolumeへbind mountしてください。

uvで直接実行する設定例です。

```toml
[environments.lab-uv]
host = "gpu-pro6000"
type = "uv"
workdir = "/home/researcher/project"
command_prefix = ["uv", "run", "--"]
python = ["python"]
gpu_visibility = "cuda_visible_devices"
log_dir = "/home/researcher/project/runs"
artifact_dir = "/home/researcher/project/artifacts"
```

`log_dir`と`artifact_dir`はGPUサーバー上の実在するディレクトリにします。
学習ユーザーから書き込める必要があります。
環境変数が必要なら、その環境の`.env`テーブルへ文字列で設定します。
`gpu_id_map`はDocker内とホストのGPU IDが一致しない場合だけ追加します。

実行前に次の順で確認します。
`config-check`はローカル設定を検証し、`hosts`はOpenSSHが解決した宛先を表示します。
`host-test`は指定したaliasへSSH接続します。

```sh
uv run --locked lab-codex-scheduler config-check \
  ~/.config/lab-codex-scheduler/environments.toml
uv run --locked lab-codex-scheduler hosts
uv run --locked lab-codex-scheduler host-test gpu-a100
```

GPU情報の取得は読み取り操作です。
学習ジョブは登録・起動しません。
観測履歴はSQLiteへ保存するため、初回は先にDBを初期化します。

```sh
uv run --locked lab-codex-scheduler db-init
```

```sh
uv run --locked lab-codex-scheduler gpu \
  --host gpu-a100 \
  --config ~/.config/lab-codex-scheduler/environments.toml
```

## Discord Botを用意する

Discord Developer PortalでApplicationを作り、Bot userを追加します。
Bot設定の「Privileged Gateway Intents」でMessage Content Intentを有効にします。
このBotは通常メッセージ中のメンションを受け取り、会話を続けるためにメッセージ本文と添付情報を扱います。
DiscordではこのIntentをDeveloper Portalで有効にする必要があります。
検証済みApplicationでは追加審査が必要になる場合があります。

Installation設定またはOAuth2 URL Generatorで、`bot`と`applications.commands`のscopeを選びます。
Bot permissionsは次に限定します。

- View Channels
- Send Messages
- Send Messages in Threads
- Create Public Threads
- Read Message History
- Attach Files
- Embed Links

Administrator、Manage Server、Manage Messages、Manage Threadsは付けません。
Botを検証用Guildへ招待し、利用するテキストチャンネルとForum channelだけに上記権限を与えます。

DiscordのDeveloper Modeを有効にすると、ユーザー、Guild、チャンネルのコンテキストメニューからIDをコピーできます。
Discordの「User Settings」から「Advanced」を開き、「Developer Mode」を有効にします。
最初はBotを使う自分のUser ID、検証用Guild ID、Botを使わせる親チャンネルIDを控えます。
Thread内の操作も親チャンネルIDを許可すれば通ります。

## Discordの設定を保存する

Bot tokenはDeveloper PortalのBot設定から取得し、制御サーバーだけに保存します。
tokenを含む`secrets.env`はGitへ置きません。

```sh
install -m 600 /dev/null ~/.config/lab-codex-scheduler/secrets.env
${EDITOR:-vi} ~/.config/lab-codex-scheduler/secrets.env
chmod 600 ~/.config/lab-codex-scheduler/secrets.env
```

`secrets.env`には次の一行だけを書き、`Developer-Portalで取得したtoken`を実際の値に置き換えます。

```dotenv
DISCORD_BOT_TOKEN=Developer-Portalで取得したtoken
```

許可IDはtokenと別のファイルへ保存します。
値はDiscordからコピーした数値IDへ置き換えてください。
複数のIDはカンマで区切ります。

```sh
cat > ~/.config/lab-codex-scheduler/discord-policy.env <<'EOF'
LAB_CDS_DISCORD_USER_IDS=自分のUser-ID
LAB_CDS_DISCORD_GUILD_IDS=検証用Guild-ID
LAB_CDS_DISCORD_CHANNEL_IDS=検証用親チャンネルID
EOF
chmod 600 ~/.config/lab-codex-scheduler/discord-policy.env
```

User IDは必須です。
Guild IDとChannel IDも設定すると、その範囲の操作に限定できます。
DMから操作する設定は既定で無効です。

## Codex CLIを認証する

Codex CLIはDiscord Botと同じLinux制御サーバーへインストールします。
systemd user serviceを使う場合、対話ログインと確認コマンドはそのserviceを動かすUnixユーザーで実行します。
ブラウザーを開けないサーバーではdevice codeログインを使います。

```sh
codex --version
codex login --device-auth
codex login status
command -v codex
```

ログインで表示された認証手順を手元のブラウザーで完了します。
`codex login status`が成功し、`command -v codex`のパスがsystemd serviceの`PATH`に含まれることを確認してください。
Codexの認証情報をBot tokenファイルやリポジトリへコピーする必要はありません。

## 最初の確認

`db-init`は前段で実行済みです。
ここでは許可IDファイルを読み込み、Botだけを前景起動します。
Schedulerは起動しないため、GPUジョブのdispatchは始まりません。

```sh
set -a
. ~/.config/lab-codex-scheduler/secrets.env
. ~/.config/lab-codex-scheduler/discord-policy.env
set +a
uv run --locked lab-codex-scheduler discord \
  --config ~/.config/lab-codex-scheduler/environments.toml
```

Discord上で`/status`を実行します。
応答が返ればBot token、Gateway、Command同期、allowlistを確認できています。
Codexを確認するときは許可チャンネルでBotをメンションして短い質問を送り、応答後に`/new-conversation`を試します。
確認中は`/submit`を使わないでください。

終了するときは前景プロセスで`Ctrl-C`を押します。
systemdへ登録する前に、下記の運用設定とバックアップ先を決めます。

## systemdを登録する前に決めること

サービス例は`docs/deployment.md`にあります。
制御サーバー上でuser unit用ディレクトリを作り、例を2つのunitファイルへコピーします。
`WorkingDirectory`を実際のclone先へ合わせてください。

```sh
mkdir -p ~/.config/systemd/user
${EDITOR:-vi} ~/.config/systemd/user/lab-codex-scheduler.service
${EDITOR:-vi} ~/.config/systemd/user/lab-codex-discord.service
```

Discord serviceには次の行を追加し、許可IDファイルを読み込ませます。

```ini
EnvironmentFile=%h/.config/lab-codex-scheduler/discord-policy.env
```

SchedulerとDiscordを同じUnixユーザーで動かし、同じSQLite DBを使います。
systemd user managerのlinger、起動時のCodex認証、GPU上のSSH接続、ログと成果物の保存先を確認してから有効化します。
GPU利用を確認するまではScheduler serviceを起動しません。

Discordだけを先に起動して確認する場合は、BotとSlash Commandが動作するか検証用Guildで`/status`を実行します。
起動ログは`journalctl --user -u lab-codex-discord.service -n 100 --no-pager`で確認できます。
確認後は`systemctl --user stop lab-codex-discord.service`で停止します。
Schedulerを起動すると、キューに登録済みのジョブをGPU条件に従ってdispatchします。
意図しないジョブがないことを`queue`で確認してから起動してください。

## 公式手順

DiscordのscopeとBot権限は[Application Commands](https://docs.discord.com/developers/interactions/application-commands)、[Permissions](https://docs.discord.com/developers/topics/permissions)、Message Content Intentは[Gateway documentation](https://docs.discord.com/developers/events/gateway)を参照してください。
Codex CLIのdevice authと`codex login status`は[OpenAI Codex CLI reference](https://developers.openai.com/codex/cli/reference)に記載されています。
