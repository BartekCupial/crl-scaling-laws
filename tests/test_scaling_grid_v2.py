import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_crl_scaling_grid_v2.py"
SPEC = importlib.util.spec_from_file_location("run_crl_scaling_grid_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
v2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = v2
SPEC.loader.exec_module(v2)


class ExpandedScalingGridTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = v2.load_plan(v2.DEFAULT_PLAN)
        cls.experiments = v2.build_experiments(cls.plan)
        cls.seeds = cls.plan["axes"]["seeds"]

    def test_total_job_counts(self):
        smoke = [item for item in self.experiments if item.stage == "smoke"]
        fitting = [item for item in self.experiments if item.stage == "fit"]
        heldout = [item for item in self.experiments if item.stage == "heldout"]
        self.assertEqual(len(smoke), 27)
        self.assertEqual(len(fitting), 75)
        self.assertEqual(len(heldout), 6)
        self.assertEqual(len(self.experiments), 108)
        self.assertEqual({item.model_family for item in self.experiments}, {"simba"})

    def test_seed_first_batches_and_heldout_gate(self):
        preheldout = v2.experiment_batches(self.experiments, "pre-heldout", self.seeds)
        self.assertEqual(
            [(label, len(batch)) for label, batch in preheldout],
            [
                ("smoke-seed-1", 27),
                ("fit-seed-1", 25),
                ("fit-seed-2", 25),
                ("fit-seed-3", 25),
            ],
        )
        for label, batch in preheldout[1:]:
            seed = int(label[-1])
            self.assertEqual({item.seed for item in batch}, {seed})
            self.assertTrue(
                all(
                    (item.width, item.depth)
                    not in {(256, 128), (512, 32)}
                    for item in batch
                )
            )
        heldout = v2.experiment_batches(self.experiments, "heldout", self.seeds)
        self.assertEqual([len(batch) for _label, batch in heldout], [2, 2, 2])
        self.assertEqual(
            {(item.width, item.depth) for _label, batch in heldout for item in batch},
            {(256, 128), (512, 32)},
        )

    def test_exact_known_parameter_counts(self):
        residual = {
            4: 982_691,
            8: 1_778_339,
            16: 3_369_635,
            32: 6_552_227,
            64: 12_917_411,
        }
        simba = {
            4: 3_605_155,
            8: 8_074_915,
            16: 17_014_435,
            32: 34_893_475,
            64: 70_651_555,
        }
        for depth, expected in residual.items():
            self.assertEqual(v2.expected_trainable_params("residual", 256, depth), expected)
        for depth, expected in simba.items():
            self.assertEqual(v2.expected_trainable_params("simba", 256, depth), expected)

    def test_initial_order_uses_parameter_proxy(self):
        smoke = v2.experiment_batches(self.experiments, "smoke", self.seeds)[0][1]
        with tempfile.TemporaryDirectory() as directory:
            ordered = v2.sort_experiments(smoke, self.experiments, Path(directory))
        self.assertEqual(ordered[0].grid_id, "S-S1-SIM-W128-D004")
        self.assertEqual(ordered[-1].grid_id, "S-S1-SIM-W256-D128")
        self.assertEqual(
            [item.expected_trainable_params for item in ordered],
            sorted(item.expected_trainable_params for item in ordered),
        )

    def test_next_seed_uses_observed_previous_seed_runtime(self):
        seed2 = v2.experiment_batches(self.experiments, "seed2", self.seeds)[0][1]
        low_cost = next(item for item in seed2 if item.grid_id == "M-S2-SIM-W128-D004")
        high_cost = next(item for item in seed2 if item.grid_id == "M-S2-SIM-W128-D192")
        lookup = v2.experiment_lookup(self.experiments)
        low_reference = lookup[("fit", 1, *low_cost.architecture_key)]
        high_reference = lookup[("fit", 1, *high_cost.architecture_key)]
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            for reference, runtime in ((low_reference, 100.0), (high_reference, 1.0)):
                run_dir = run_root / reference.run_name
                run_dir.mkdir()
                (run_dir / "runtime.json").write_text(
                    json.dumps({"runtime_seconds": runtime}), encoding="utf-8"
                )
            ordered = v2.sort_experiments([low_cost, high_cost], self.experiments, run_root)
        self.assertEqual([item.grid_id for item in ordered], [high_cost.grid_id, low_cost.grid_id])

    def test_command_contains_all_grid_axes(self):
        experiment = next(
            item for item in self.experiments if item.grid_id == "M-S3-SIM-W512-D024"
        )
        command = v2.build_command(
            self.plan["runner"], self.plan["fixed_args"], experiment, Path("/tmp/v2-run")
        )
        expected = {
            "--seed": "3",
            "--use-simba": "1",
            "--critic-network-width": "512",
            "--actor-network-width": "512",
            "--critic-depth": "24",
            "--actor-depth": "24",
            "--total-env-steps": "443000000",
        }
        for flag, value in expected.items():
            self.assertEqual(command[command.index(flag) + 1], value)

    def test_extended_horizons_scale_from_two_billion_to_four_hundred_million(self):
        main = [
            item for item in self.experiments
            if item.stage in {"fit", "heldout"} and item.seed == 1
        ]
        ordered = sorted(main, key=lambda item: item.expected_trainable_params)
        self.assertEqual(ordered[0].total_env_steps, 2_000_000_000)
        self.assertEqual(ordered[-1].total_env_steps, 400_000_000)
        self.assertEqual(ordered[0].num_epochs, 400)
        self.assertEqual(ordered[-1].num_epochs, 100)
        self.assertTrue(
            all(left.total_env_steps >= right.total_env_steps for left, right in zip(ordered, ordered[1:]))
        )
        self.assertTrue(
            all(item.total_env_steps / item.num_epochs <= 5_000_000 for item in main)
        )

    def test_resume_skips_only_completed_directories(self):
        selected = self.experiments[:2]
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            completed_dir = run_root / selected[0].run_name
            completed_dir.mkdir()
            (completed_dir / "COMPLETE").write_text("ok\n", encoding="utf-8")
            pending, completed = v2.classify_existing(selected, run_root, resume_completed=True)
            self.assertEqual(pending, [selected[1]])
            self.assertEqual(completed, [selected[0]])
            with self.assertRaises(RuntimeError):
                v2.classify_existing(selected, run_root, resume_completed=False)

    def test_skip_existing_skips_active_and_failed_directories(self):
        selected = self.experiments[:3]
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            (run_root / selected[0].run_name).mkdir()
            failed = run_root / selected[1].run_name
            failed.mkdir()
            (failed / "FAILED").write_text("exit_code=130\n", encoding="utf-8")
            pending, skipped = v2.classify_existing(
                selected, run_root, resume_completed=True, skip_existing=True
            )
            self.assertEqual(pending, [selected[2]])
            self.assertEqual(skipped, selected[:2])

    def test_seed_and_heldout_prerequisites_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            with self.assertRaises(RuntimeError):
                v2.validate_prerequisites(self.experiments, "seed2", run_root)

            prerequisites = v2.prerequisite_experiments(self.experiments, "seed2")
            self.assertEqual(len(prerequisites), 52)
            for experiment in prerequisites:
                run_dir = run_root / experiment.run_name
                run_dir.mkdir()
                (run_dir / "COMPLETE").write_text("ok\n", encoding="utf-8")
            v2.validate_prerequisites(self.experiments, "seed2", run_root)

            with self.assertRaises(RuntimeError):
                v2.validate_prerequisites(self.experiments, "heldout", run_root)

    def test_max_jobs_must_be_positive(self):
        with self.assertRaises(ValueError):
            v2.main(["--max-jobs", "0"])

    def test_worker_continues_queue_after_one_run_fails(self):
        selected = self.experiments[:2]
        results = [
            v2.RunResult(selected[0], "0", 130, 1.0, Path("failed.log")),
            v2.RunResult(selected[1], "0", 0, 1.0, Path("ok.log")),
        ]
        with mock.patch.object(v2, "run_one", side_effect=results) as run_one:
            succeeded = v2.run_batch(selected, ["0"], {})
        self.assertFalse(succeeded)
        self.assertEqual(run_one.call_count, 2)


if __name__ == "__main__":
    unittest.main()
