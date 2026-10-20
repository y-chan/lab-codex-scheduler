import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from lab_codex_scheduler.config import load_environments
from lab_codex_scheduler.db import connect, initialize
from lab_codex_scheduler.discord_state import DiscordStore
from lab_codex_scheduler.jobs import JobRepository, JobSpec, now
from lab_codex_scheduler.plans import PlanRepository
from lab_codex_scheduler.proposals import ProposalNotEligible, create_next_experiment_proposal


class FakeCodex:
    running = True

    def __init__(self):
        self.turns = 0
        self.loaded: list[str] = []

    async def start(self):
        self.running = True

    async def ensure_thread_loaded(self, thread_id: str):
        self.loaded.append(thread_id)

    async def run_turn(self, thread_id: str, prompt: str, **kwargs):
        self.turns += 1
        self.thread_id = thread_id
        self.prompt = prompt
        self.schema = kwargs["output_schema"]
        return {"finalResponse": json.dumps({
            "environment_id": "train",
            "command_argv": ["python", "train.py", "--lr", "0.0001"],
            "priority": 0,
            "gpu": {"min_free_vram_bytes": 0, "max_utilization_percent": 5, "idle_seconds": 120},
            "rationale": "評価指標に基づく次の候補です",
        })}


class NextExperimentProposalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "state.sqlite3"
        self.config = self.root / "environments.toml"
        self.config.write_text('''
[environments.train]
host = "gpu-a"
type = "native"
workdir = "/workspace"
propose_next_after_evaluation = true
''', encoding="utf-8")
        initialize(self.db)
        self.environment = load_environments(self.config)["train"]
        self.experiment_id = "experiment-1"
        self.evaluation_id = "evaluation-1"
        repository = JobRepository(str(self.db))
        experiment = JobSpec(self.experiment_id, "train", "gpu-a", ("python", "train.py"),
                             "/workspace", {})
        evaluation = JobSpec(self.evaluation_id, "train", "gpu-a", ("python", "evaluate.py"),
                             "/workspace", {})
        repository.create(experiment, "config", requester_id="user-1",
                          notification_channel_id="thread-1")
        repository.transition(self.experiment_id, "starting")
        repository.transition(self.experiment_id, "running")
        repository.transition(self.experiment_id, "succeeded", exit_code=0)
        repository.create(evaluation, "config", requester_id="user-1",
                          notification_channel_id="thread-1", depends_on=[self.experiment_id],
                          job_kind="evaluation")
        repository.transition(self.evaluation_id, "starting")
        repository.transition(self.evaluation_id, "running")
        repository.transition(self.evaluation_id, "succeeded", exit_code=0)
        with connect(self.db) as db:
            db.execute(
                "INSERT INTO evaluation_followups(parent_job_id,evaluation_job_id,created_at) VALUES(?,?,?)",
                (self.experiment_id, self.evaluation_id, now()),
            )
        DiscordStore(self.db).create_session(
            thread_id="thread-1", guild_id="guild-1", parent_channel_id="channel-1",
            starter_user_id="user-1", settings_sha256="sha", codex_thread_id="codex-1",
        )
        self.codex = FakeCodex()

    def test_proposal_is_scoped_persisted_once_and_not_queued_until_approved(self):
        kwargs = dict(
            db_path=self.db, environment_config=self.config, thread_id="thread-1",
            requester_id="user-1", evaluation_job_id=self.evaluation_id,
            experiment_job_id=self.experiment_id, tensorboard_cache=self.root / "tb",
            codex=self.codex, instructions="research instructions",
        )
        first = asyncio.run(create_next_experiment_proposal(**kwargs))
        replay = asyncio.run(create_next_experiment_proposal(**kwargs))
        self.assertFalse(first.replay)
        self.assertTrue(replay.replay)
        self.assertEqual(first.plan_id, replay.plan_id)
        self.assertEqual(self.codex.turns, 1)
        self.assertEqual(self.codex.thread_id, "codex-1")
        self.assertEqual(self.codex.schema["properties"]["environment_id"]["enum"], ["train"])
        with connect(self.db) as db:
            count = db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            plan = db.execute("SELECT state,job_id FROM experiment_plans WHERE plan_id=?",
                              (first.plan_id,)).fetchone()
        self.assertEqual(count, 2)
        self.assertEqual(tuple(plan), ("pending", None))

        job_id = PlanRepository(self.db).decide(
            first.plan_id, "user-1", True, self.environment, self.config,
        )
        self.assertTrue(job_id)
        with connect(self.db) as db:
            count = db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        self.assertEqual(count, 3)

    def test_event_cannot_submit_for_another_requester_or_thread(self):
        kwargs = dict(
            db_path=self.db, environment_config=self.config, thread_id="thread-1",
            requester_id="someone-else", evaluation_job_id=self.evaluation_id,
            experiment_job_id=self.experiment_id, tensorboard_cache=self.root / "tb",
            codex=self.codex, instructions="research instructions",
        )
        with self.assertRaisesRegex(ProposalNotEligible, "依頼者"):
            asyncio.run(create_next_experiment_proposal(**kwargs))
        self.assertEqual(self.codex.turns, 0)
        with connect(self.db) as db:
            count = db.execute("SELECT COUNT(*) FROM experiment_plans").fetchone()[0]
        self.assertEqual(count, 0)

    def test_event_requires_bound_thread_and_current_opt_in(self):
        common = dict(
            db_path=self.db, environment_config=self.config, requester_id="user-1",
            evaluation_job_id=self.evaluation_id, experiment_job_id=self.experiment_id,
            tensorboard_cache=self.root / "tb", codex=self.codex, instructions="instructions",
        )
        with self.assertRaisesRegex(ProposalNotEligible, "Thread"):
            asyncio.run(create_next_experiment_proposal(**common, thread_id="unbound-thread"))
        self.config.write_text(
            '[environments.train]\nhost="gpu-a"\ntype="native"\nworkdir="/workspace"\n'
            'propose_next_after_evaluation=false\n', encoding="utf-8",
        )
        with self.assertRaisesRegex(ProposalNotEligible, "有効化"):
            asyncio.run(create_next_experiment_proposal(**common, thread_id="thread-1"))
        self.assertEqual(self.codex.turns, 0)


if __name__ == "__main__":
    unittest.main()
