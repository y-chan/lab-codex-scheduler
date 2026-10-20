import tempfile
import unittest
from pathlib import Path

from lab_codex_scheduler.db import initialize
from lab_codex_scheduler.gpu import (
    GPUAllocator, GPUProcess, GPURequest, GPUReservationStore, GPUSnapshot,
    NVIDIAProbe, choose_gpu, persist_snapshots,
)
from lab_codex_scheduler.jobs import JobRepository, JobSpec


class FakeRunner:
    def __init__(self, processes: str = "") -> None:
        self.processes = processes
        self.calls: list[list[str]] = []

    def run(self, host: str | None, argv: list[str]) -> str:
        self.calls.append(argv)
        if "--query-gpu=" in argv[1]:
            return "GPU-a, NVIDIA A100-SXM4-80GB, 81920, 2048, 0\n"
        return self.processes


class GPUTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db_path = Path(self.temp.name) / "scheduler.sqlite3"
        initialize(self.db_path)

    def point(self, observed: float, util: float = 0, used: int = 2_048 * 1024**2,
              processes: tuple[GPUProcess, ...] = (), known: bool = True) -> GPUSnapshot:
        return GPUSnapshot("gpu-a", "GPU-a", "NVIDIA A100", 80 * 1024**3,
                           used, util, processes, known, observed)

    def create_job(self, job_id: str) -> None:
        JobRepository(str(self.db_path)).create(
            JobSpec(job_id, "env", "gpu-a", ("true",), "/tmp", {}), "config-hash"
        )

    def test_probe_parses_inventory_and_maps_process_per_uuid(self) -> None:
        runner = FakeRunner("1234, python, 512\n")
        snapshots = NVIDIAProbe(runner).collect("gpu-a", observed_at=500)
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].uuid, "GPU-a")
        self.assertEqual(snapshots[0].memory_total_bytes, 81920 * 1024**2)
        self.assertEqual(snapshots[0].processes[0].pid, 1234)
        self.assertEqual(snapshots[0].processes[0].used_memory_bytes, 512 * 1024**2)
        self.assertIn("GPU-a", runner.calls[1])

    def test_history_requires_a_continuous_recent_idle_window(self) -> None:
        request = GPURequest(min_free_vram_bytes=50 * 1024**3, idle_seconds=60,
                             max_sample_gap_seconds=25, max_sample_age_seconds=15)
        history = [self.point(t) for t in (930, 950, 970, 990)]
        selected = choose_gpu(history, request, now=1000)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.snapshot.uuid, "GPU-a")

        occupied = [self.point(930), self.point(950), self.point(970, processes=(
            GPUProcess(2345, "python", 1024),
        )), self.point(990)]
        self.assertIsNone(choose_gpu(occupied, request, now=1000))
        self.assertIsNone(choose_gpu([self.point(970)], request, now=1000))

    def test_unknown_process_inventory_blocks_shared_gpu_selection(self) -> None:
        request = GPURequest(idle_seconds=60, max_sample_gap_seconds=25)
        history = [self.point(t, known=(t != 970)) for t in (930, 950, 970, 990)]
        self.assertIsNone(choose_gpu(history, request, now=1000))

    def test_sqlite_reservations_exclude_other_jobs_and_release(self) -> None:
        store = GPUReservationStore(self.db_path)
        self.create_job("job-one")
        self.create_job("job-two")
        self.assertTrue(store.acquire("gpu-a", "GPU-a", "job-one", at=1000))
        self.assertFalse(store.acquire("gpu-a", "GPU-a", "job-two", at=1000))
        self.assertEqual(store.active(), {("gpu-a", "GPU-a")})
        store.release("job-one", at=1001)
        self.assertTrue(store.acquire("gpu-a", "GPU-a", "job-two", at=1002))

    def test_allocator_reserves_then_rechecks_before_dispatch(self) -> None:
        request = GPURequest(idle_seconds=60, max_sample_gap_seconds=25, max_sample_age_seconds=15)
        persist_snapshots(self.db_path, [self.point(t) for t in (930, 950, 970, 990)])
        allocator = GPUAllocator(self.db_path)
        self.create_job("job-one")
        self.create_job("job-two")
        selected = allocator.reserve("job-one", "gpu-a", request,
                                     lambda host: [self.point(1000)], now=1000)
        self.assertIsNotNone(selected)
        self.assertEqual(allocator.reservations.for_job("job-one"), ("gpu-a", "GPU-a"))
        self.assertIsNone(allocator.reserve("job-two", "gpu-a", request,
                                            lambda host: [self.point(1000)], now=1000))

    def test_preflight_race_releases_reservation(self) -> None:
        request = GPURequest(idle_seconds=60, max_sample_gap_seconds=25, max_sample_age_seconds=15)
        persist_snapshots(self.db_path, [self.point(t) for t in (930, 950, 970, 990)])
        allocator = GPUAllocator(self.db_path)
        self.create_job("job-one")
        busy = self.point(1000, processes=(GPUProcess(321, "python", 1024),))
        self.assertIsNone(allocator.reserve("job-one", "gpu-a", request, lambda host: [busy], now=1000))
        self.assertIsNone(allocator.reservations.for_job("job-one"))


if __name__ == "__main__":
    unittest.main()
