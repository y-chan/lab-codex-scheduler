"""Initial offline-friendly administration commands."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
import os
import sys
import uuid
from pathlib import Path

from lab_codex_scheduler import __version__
from lab_codex_scheduler.config import ConfigError, load_environments
from lab_codex_scheduler.db import backup_database, connect, initialize
from lab_codex_scheduler.gpu import GPURequest, GPUReservationStore, NVIDIAProbe, persist_snapshots
from lab_codex_scheduler.jobs import JobRepository, JobSpec, RemoteSupervisor, config_digest
from lab_codex_scheduler.jobs import OpenSSHTransport
from lab_codex_scheduler.ssh import OpenSSH, SSHError
from lab_codex_scheduler.scheduler import QueueDispatcher
from lab_codex_scheduler.service import SchedulerService
from lab_codex_scheduler.sweeps import expand_sweep


def default_config_path() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / \
        "lab-codex-scheduler" / "environments.toml"


def default_db_path() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / \
        "lab-codex-scheduler" / "scheduler.sqlite3"


def default_data_dir() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / \
        "lab-codex-scheduler"


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="lab-codex-scheduler")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    check = commands.add_parser("config-check", help="環境定義を検証")
    check.add_argument("path", nargs="?", type=Path, default=default_config_path())
    commands.add_parser("hosts", help="環境定義で参照されるOpenSSH Hostを表示")
    test = commands.add_parser("host-test", help="指定HostへのSSH接続を確認")
    test.add_argument("alias", help="登録済みOpenSSH Host別名")
    submit = commands.add_parser("submit", help="指定実行環境へジョブを登録して起動")
    submit.add_argument("--environment", required=True)
    submit.add_argument("--config", type=Path, default=default_config_path())
    submit.add_argument("--db", type=Path, default=default_db_path())
    submit.add_argument("--job-id", default=None)
    submit.add_argument(
        "--confirm-resource-check", action="store_true",
        help="GPU予約キューを通さない直接起動のため、GPU利用状況を手動確認済みであることを明示",
    )
    submit.add_argument("command_argv", nargs=argparse.REMAINDER)
    enqueue = commands.add_parser("enqueue", help="GPU要求付きジョブを永続キューへ登録")
    enqueue.add_argument("--environment", required=True)
    enqueue.add_argument("--config", type=Path, default=default_config_path())
    enqueue.add_argument("--db", type=Path, default=default_db_path())
    enqueue.add_argument("--job-id", default=None)
    enqueue.add_argument("--priority", type=int, default=0)
    enqueue.add_argument("--model-contains", default=None)
    enqueue.add_argument("--min-free-vram-gib", type=float, default=0.0)
    enqueue.add_argument("--max-utilization", type=float, default=5.0)
    enqueue.add_argument("--idle-seconds", type=float, default=120.0)
    enqueue.add_argument("--max-sample-gap-seconds", type=float, default=45.0)
    enqueue.add_argument("--max-sample-age-seconds", type=float, default=30.0)
    enqueue.add_argument("--retry-limit", type=int, default=0)
    enqueue.add_argument("--retry-delay-seconds", type=int, default=60)
    enqueue.add_argument("--deadline-at", default=None, help="完了期限（timezone付きISO-8601）")
    enqueue.add_argument("--depends-on", action="append", default=[], metavar="JOB_UUID")
    enqueue.add_argument("command_argv", nargs=argparse.REMAINDER)
    sweep = commands.add_parser("enqueue-sweep", help="条件値の直積をGPUキューへ一括登録")
    sweep.add_argument("--environment", required=True)
    sweep.add_argument("--spec", type=Path, required=True, help="command_argv/parametersを含むJSON")
    sweep.add_argument("--config", type=Path, default=default_config_path())
    sweep.add_argument("--db", type=Path, default=default_db_path())
    sweep.add_argument("--priority", type=int, default=0)
    sweep.add_argument("--model-contains", default=None)
    sweep.add_argument("--min-free-vram-gib", type=float, default=0.0)
    sweep.add_argument("--retry-limit", type=int, default=0)
    sweep.add_argument("--retry-delay-seconds", type=int, default=60)
    sweep.add_argument("--deadline-at", default=None)
    status = commands.add_parser("job-status", help="ジョブ状態をリモートSupervisorと照合")
    status.add_argument("job_id")
    status.add_argument("--db", type=Path, default=default_db_path())
    cancel = commands.add_parser("job-cancel", help="管理対象ジョブへ停止要求")
    cancel.add_argument("job_id")
    cancel.add_argument("--db", type=Path, default=default_db_path())
    for command, help_text in (("job-pause", "待機中または実行中の管理対象ジョブを一時停止"),
                               ("job-resume", "一時停止した管理対象ジョブを再開")):
        control = commands.add_parser(command, help=help_text)
        control.add_argument("job_id")
        control.add_argument("--db", type=Path, default=default_db_path())
    gpu = commands.add_parser("gpu", help="登録HostのGPU状態を取得して履歴に保存")
    gpu.add_argument("--host", required=True)
    gpu.add_argument("--config", type=Path, default=default_config_path())
    gpu.add_argument("--db", type=Path, default=default_db_path())
    queue = commands.add_parser("queue", help="依存関係が満たされた待機ジョブを表示")
    queue.add_argument("--db", type=Path, default=default_db_path())
    tick = commands.add_parser("schedule-once", help="条件を満たす待機GPUジョブを1件まで起動")
    tick.add_argument("--config", type=Path, default=default_config_path())
    tick.add_argument("--db", type=Path, default=default_db_path())
    service = commands.add_parser("scheduler", help="定期監視とGPUジョブdispatchを実行")
    service.add_argument("--config", type=Path, default=default_config_path())
    service.add_argument("--db", type=Path, default=default_db_path())
    service.add_argument("--interval-seconds", type=float, default=30.0)
    init = commands.add_parser("db-init", help="SQLite状態DBを初期化")
    init.add_argument("--path", type=Path, default=default_db_path())
    backup = commands.add_parser("db-backup", help="SQLite状態DBの検証済みバックアップを作成")
    backup.add_argument("destination", type=Path, help="既存ファイルと重ならないバックアップ先")
    backup.add_argument("--db", type=Path, default=default_db_path())
    discord = commands.add_parser("discord", help="Discord Botを起動")
    discord.add_argument("--db", type=Path, default=default_db_path())
    discord.add_argument("--data-dir", type=Path, default=default_data_dir())
    discord.add_argument("--config", type=Path, default=default_config_path())
    analyze = commands.add_parser("analyze", help="TensorBoard scalarを決定的に集計")
    analyze.add_argument("logdir", type=Path)
    analyze.add_argument("--tag", action="append", default=[])
    analyze.add_argument("--window", type=int, default=20)
    plot = commands.add_parser("plot", help="TensorBoard scalarの比較グラフを生成")
    plot.add_argument("--run", action="append", required=True, metavar="NAME=LOGDIR")
    plot.add_argument("--tag", action="append", default=[])
    plot.add_argument("--output", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "config-check":
            environments = load_environments(args.path)
            print(f"設定OK: {len(environments)} environments ({args.path})")
        elif args.command == "hosts":
            environments = load_environments(default_config_path())
            ssh = OpenSSH()
            seen: set[str] = set()
            for environment in environments.values():
                if environment.host not in seen:
                    host = ssh.resolve(environment.host)
                    print(f"{host.alias}\t{host.user}@{host.hostname}:{host.port}\tProxyJump={host.proxyjump}")
                    seen.add(host.alias)
        elif args.command == "host-test":
            environments = load_environments(default_config_path())
            if args.alias not in {environment.host for environment in environments.values()}:
                raise ConfigError(f"環境定義にないHostです: {args.alias}")
            OpenSSH().resolve(args.alias)
            OpenSSH().test(args.alias)
            print(f"接続OK: {args.alias}")
        elif args.command == "db-init":
            initialize(args.path)
            print(f"SQLite初期化OK: {args.path}")
        elif args.command == "db-backup":
            destination = backup_database(args.db, args.destination)
            print(json.dumps({"status": "verified", "backup": str(destination)}, ensure_ascii=False))
        elif args.command == "submit":
            if not args.confirm_resource_check:
                raise ConfigError(
                    "直接起動はGPU予約を行いません。GPU状況を確認した場合だけ "
                    "--confirm-resource-check を付けてください"
                )
            argv = args.command_argv[1:] if args.command_argv[:1] == ["--"] else args.command_argv
            if not argv:
                raise ConfigError("実行コマンドを -- の後に指定してください")
            environments = load_environments(args.config)
            if args.environment not in environments:
                raise ConfigError(f"未登録の実行環境です: {args.environment}")
            if not -1000 <= args.priority <= 1000:
                raise ConfigError("priorityは-1000から1000の範囲で指定してください")
            try:
                job_id = str(uuid.UUID(args.job_id)) if args.job_id else str(uuid.uuid4())
            except ValueError as exc:
                raise ConfigError("job-idはUUID形式で指定してください") from exc
            spec = JobSpec.create(environments[args.environment], argv, job_id)
            repo = JobRepository(str(args.db))
            repo.create(spec, config_digest(str(args.config)))
            repo.transition(spec.id, "starting")
            supervisor = RemoteSupervisor()
            code_revision = None
            try:
                code_revision = supervisor.revision(spec)
                repo.store_code_revision(spec.id, code_revision)
            except (ConnectionError, OSError, RuntimeError) as exc:
                print(f"Git revision unavailable: {exc}", file=sys.stderr)
            try:
                remote = supervisor.start(spec)
            except (ConnectionError, OSError, RuntimeError) as exc:
                repo.transition(spec.id, "unknown")
                print(f"ジョブ状態不明: {spec.id}: {exc}", file=sys.stderr)
                return 2
            state = str(remote.get("status", "unknown"))
            repo.transition(spec.id, state if state in {"running", "failed", "succeeded", "unknown"} else "unknown",
                            remote.get("exit_code"))
            print(json.dumps({**remote, "code_revision": code_revision}, ensure_ascii=False))
        elif args.command == "enqueue":
            argv = args.command_argv[1:] if args.command_argv[:1] == ["--"] else args.command_argv
            if not argv:
                raise ConfigError("実行コマンドを -- の後に指定してください")
            environments = load_environments(args.config)
            if args.environment not in environments:
                raise ConfigError(f"未登録の実行環境です: {args.environment}")
            try:
                job_id = str(uuid.UUID(args.job_id)) if args.job_id else str(uuid.uuid4())
                request = GPURequest(
                    model_contains=args.model_contains,
                    min_free_vram_bytes=int(args.min_free_vram_gib * 1024**3),
                    max_utilization_percent=args.max_utilization,
                    idle_seconds=args.idle_seconds,
                    max_sample_gap_seconds=args.max_sample_gap_seconds,
                    max_sample_age_seconds=args.max_sample_age_seconds,
                )
            except (ValueError, OverflowError) as exc:
                raise ConfigError(f"ジョブIDまたはGPU条件が不正です: {exc}") from exc
            environment = environments[args.environment]
            if environment.gpu_visibility == "none":
                raise ConfigError("gpu_visibility=noneの環境にはGPUジョブを登録できません")
            spec = JobSpec.create(environment, argv, job_id)
            JobRepository(str(args.db)).create(
                spec, config_digest(str(args.config)), priority=args.priority,
                gpu_request=asdict(request), retry_limit=args.retry_limit,
                retry_delay_seconds=args.retry_delay_seconds, deadline_at=args.deadline_at,
                depends_on=args.depends_on,
            )
            print(json.dumps({"job_id": spec.id, "status": "queued", "priority": args.priority},
                             ensure_ascii=False))
        elif args.command == "enqueue-sweep":
            environments = load_environments(args.config)
            environment = environments.get(args.environment)
            if environment is None:
                raise ConfigError(f"未登録の実行環境です: {args.environment}")
            if environment.gpu_visibility == "none":
                raise ConfigError("gpu_visibility=noneの環境にはGPUジョブを登録できません")
            if not -100 <= args.priority <= 100:
                raise ValueError("priorityは-100から100の範囲で指定してください")
            if not 0 <= args.min_free_vram_gib <= 1024:
                raise ValueError("min_free_vram_gibは0から1024の範囲で指定してください")
            if args.spec.stat().st_size > 1024 * 1024:
                raise ValueError("sweep specification exceeds 1 MiB")
            commands_argv = expand_sweep(environment, args.spec.read_bytes())
            specs = [JobSpec.create(environment, list(argv)) for argv in commands_argv]
            request = GPURequest(
                model_contains=args.model_contains,
                min_free_vram_bytes=int(args.min_free_vram_gib * 1024**3),
            )
            JobRepository(str(args.db)).create_many(
                specs, config_digest(str(args.config)), priority=args.priority,
                gpu_request=asdict(request), retry_limit=args.retry_limit,
                retry_delay_seconds=args.retry_delay_seconds, deadline_at=args.deadline_at,
            )
            print(json.dumps({"status": "queued", "count": len(specs),
                              "job_ids": [spec.id for spec in specs]}, ensure_ascii=False))
        elif args.command in {"job-pause", "job-resume"}:
            repository = JobRepository(str(args.db))
            paused = args.command == "job-pause"
            result = repository.set_paused(args.job_id, paused)
            print(json.dumps({**result, "paused": paused}, ensure_ascii=False))
        elif args.command in {"job-status", "job-cancel"}:
            repo = JobRepository(str(args.db))
            spec, local_status, _ = repo.get(args.job_id)
            if args.command == "job-cancel" and local_status in {"queued", "waiting_gpu"}:
                repo.transition(args.job_id, "canceled")
                GPUReservationStore(args.db).release(args.job_id)
                remote = {"job_id": args.job_id, "status": "canceled"}
            elif args.command == "job-cancel" and local_status in {"succeeded", "failed", "canceled"}:
                remote = {"job_id": args.job_id, "status": local_status}
            else:
                supervisor = RemoteSupervisor()
                remote = supervisor.status(spec) if args.command == "job-status" else supervisor.stop(spec)
            remote_status = str(remote.get("status", "unknown"))
            if remote_status in {"running", "succeeded", "failed", "canceled", "unknown"} and \
                    remote_status != local_status:
                repo.transition(args.job_id, remote_status, remote.get("exit_code"))
            if remote_status in {"running", "paused"}:
                repo.set_active_paused(args.job_id, remote_status == "paused")
            if args.command == "job-status" and remote_status in {"succeeded", "failed", "canceled"}:
                GPUReservationStore(args.db).release(args.job_id)
            with connect(args.db) as db:
                metadata = db.execute(
                    "SELECT code_revision,paused FROM jobs WHERE id=?", (args.job_id,),
                ).fetchone()
            print(json.dumps({**remote, "paused": bool(metadata["paused"]) if metadata else False,
                              "code_revision": metadata["code_revision"] if metadata else None},
                             ensure_ascii=False))
        elif args.command == "gpu":
            environments = load_environments(args.config)
            if args.host not in {environment.host for environment in environments.values()}:
                raise ConfigError(f"環境定義にないHostです: {args.host}")
            snapshots = NVIDIAProbe(OpenSSHTransport()).collect(args.host)
            persist_snapshots(args.db, snapshots)
            print(json.dumps([
                {"host": point.host, "uuid": point.uuid, "model": point.model,
                 "memory_total_bytes": point.memory_total_bytes,
                 "memory_free_bytes": point.memory_free_bytes,
                 "utilization_percent": point.utilization_percent,
                 "processes": [process.__dict__ for process in point.processes],
                 "processes_known": point.processes_known,
                 "observed_at": point.observed_at}
                for point in snapshots
            ], ensure_ascii=False))
        elif args.command == "queue":
            for job in JobRepository(str(args.db)).next_queued():
                print(json.dumps({
                    "id": job["id"], "status": job["status"],
                    "environment_id": job["environment_id"], "priority": job["priority"],
                    "not_before": job["not_before"], "deadline_at": job["deadline_at"],
                    "retry_count": job["retry_count"], "retry_limit": job["retry_limit"],
                    "gpu_request": json.loads(str(job["gpu_request_json"])),
                }, ensure_ascii=False))
        elif args.command == "schedule-once":
            result = QueueDispatcher(args.db, args.config).dispatch_once()
            print(json.dumps(result or {"status": "no_dispatchable_job"}, ensure_ascii=False))
        elif args.command == "scheduler":
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
            SchedulerService(args.db, args.config, args.interval_seconds).run()
        elif args.command == "discord":
            from lab_codex_scheduler.discord_bot import run_from_env

            run_from_env(args.db, args.data_dir, args.config)
        elif args.command == "analyze":
            from dataclasses import asdict
            from lab_codex_scheduler.analysis import read_tensorboard, summarize

            series = read_tensorboard(args.logdir, args.tag or None)
            result = {}
            for tag, points in series.items():
                try:
                    result[tag] = asdict(summarize(points, window=args.window))
                except ValueError as exc:
                    result[tag] = {"error": str(exc), "count": len(points)}
            print(json.dumps({"logdir": str(args.logdir), "tags": result}, ensure_ascii=False, indent=2))
        elif args.command == "plot":
            from lab_codex_scheduler.analysis import plot_scalars, read_tensorboard

            runs = {}
            for item in args.run:
                if "=" not in item:
                    raise ConfigError("--runは NAME=LOGDIR 形式で指定してください")
                name, logdir = item.split("=", 1)
                if not name or not logdir or name in runs:
                    raise ConfigError("--runの名前とLOGDIRは空にせず、名前を重複させないでください")
                runs[name] = read_tensorboard(logdir, args.tag or None)
            output = plot_scalars(runs, args.output, args.tag or None)
            print(json.dumps({"output": str(output)}, ensure_ascii=False))
    except (ConfigError, SSHError, OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
