import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_crl_scaling_grid_4gpu.py"
SPEC = importlib.util.spec_from_file_location("run_crl_scaling_grid_4gpu", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
parallel = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = parallel
SPEC.loader.exec_module(parallel)


class FourGpuScalingGridTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan_path = parallel.DEFAULT_PLAN
        cls.plan = parallel.load_plan(cls.plan_path)
        cls.experiments = parallel.build_experiments(cls.plan)

    def test_parse_gpu_ids(self):
        self.assertEqual(parallel.parse_gpu_ids("0, 1,2,3"), ["0", "1", "2", "3"])
        with self.assertRaises(Exception):
            parallel.parse_gpu_ids("0,1,1")

    def test_defaults_to_slurm_visible_devices(self):
        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "2,3,5,7"}):
            self.assertEqual(parallel.default_gpu_ids(), ["2", "3", "5", "7"])

    def test_preheldout_has_smoke_main_barrier(self):
        batches = parallel.experiment_batches(self.experiments, "pre-heldout")
        self.assertEqual([[item.grid_id for item in batch] for batch in batches], [
            ["S04", "S08", "S16", "S32", "S64"],
            ["M04", "M08", "M16", "M32"],
        ])

    def test_smoke_initial_placement_and_queue(self):
        smoke = parallel.experiment_batches(self.experiments, "smoke")[0]
        self.assertEqual([item.grid_id for item in smoke[:4]], ["S04", "S08", "S16", "S32"])
        self.assertEqual(smoke[4].grid_id, "S64")

    def test_single_run_command_targets_exact_experiment(self):
        experiment = self.experiments[2]
        command = parallel.single_run_command(experiment, self.plan_path, None)
        self.assertEqual(command[command.index("--stage") + 1], "smoke")
        self.assertEqual(command[command.index("--start-at") + 1], "S16")
        self.assertEqual(command[command.index("--stop-after") + 1], "S16")
        self.assertIn("--execute", command)

    def test_heldout_requires_prediction(self):
        heldout = self.experiments[-1]
        with self.assertRaises(ValueError):
            parallel.single_run_command(heldout, self.plan_path, None)

    def test_successful_batch_runs_every_experiment(self):
        smoke = parallel.experiment_batches(self.experiments, "smoke")[0]
        calls = []

        def successful_run(experiment, gpu, _plan, _artifact, log_dir, _registry):
            calls.append((experiment.grid_id, gpu))
            return parallel.RunResult(experiment, gpu, 0, log_dir / f"{experiment.grid_id}.log")

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            parallel, "run_one", side_effect=successful_run
        ):
            succeeded = parallel.run_batch(
                smoke,
                ["0", "1", "2", "3"],
                self.plan_path,
                None,
                Path(directory),
                parallel.ProcessRegistry(),
            )

        self.assertTrue(succeeded)
        self.assertEqual({grid_id for grid_id, _gpu in calls}, {"S04", "S08", "S16", "S32", "S64"})
        initial_mapping = {grid_id: gpu for grid_id, gpu in calls if grid_id != "S64"}
        self.assertEqual(initial_mapping, {"S04": "0", "S08": "1", "S16": "2", "S32": "3"})

    def test_failure_stops_new_work_on_that_queue(self):
        smoke = parallel.experiment_batches(self.experiments, "smoke")[0]
        calls = []

        def failed_run(experiment, gpu, _plan, _artifact, log_dir, _registry):
            calls.append(experiment.grid_id)
            return parallel.RunResult(experiment, gpu, 1, log_dir / f"{experiment.grid_id}.log")

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            parallel, "run_one", side_effect=failed_run
        ), mock.patch("sys.stderr"):
            succeeded = parallel.run_batch(
                smoke,
                ["0"],
                self.plan_path,
                None,
                Path(directory),
                parallel.ProcessRegistry(),
            )

        self.assertFalse(succeeded)
        self.assertEqual(calls, ["S04"])


if __name__ == "__main__":
    unittest.main()
