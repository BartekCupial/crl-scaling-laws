import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_crl_scaling_grid.py"
SIMBA_PLAN = REPO_ROOT / "configs" / "crl_scaling_humanoid_simba_v1.json"
SPEC = importlib.util.spec_from_file_location("run_crl_scaling_grid", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
grid = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = grid
SPEC.loader.exec_module(grid)


class ScalingGridTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = grid.load_plan(grid.DEFAULT_PLAN)
        cls.experiments = grid.build_experiments(cls.plan)

    def test_grid_is_in_required_cost_order(self):
        self.assertEqual(
            [experiment.grid_id for experiment in self.experiments],
            ["S04", "S08", "S16", "S32", "S64", "M04", "M08", "M16", "M32", "M64"],
        )

    def test_smoke_overrides_only_budget_stage_fields(self):
        smoke = self.experiments[:5]
        self.assertEqual([experiment.depth for experiment in smoke], [4, 8, 16, 32, 64])
        self.assertTrue(all(experiment.total_env_steps == 1_000_000 for experiment in smoke))
        self.assertTrue(all(experiment.num_epochs == 5 for experiment in smoke))
        self.assertTrue(all(experiment.wandb_group == "humanoid_residual_smoke_v1_s2" for experiment in smoke))

    def test_main_and_heldout_naming_and_budget(self):
        main = self.experiments[5:]
        self.assertEqual([experiment.depth for experiment in main], [4, 8, 16, 32, 64])
        self.assertTrue(all(experiment.total_env_steps == 100_000_000 for experiment in main))
        self.assertTrue(all(experiment.num_epochs == 100 for experiment in main))
        self.assertEqual(main[-1].stage, "heldout")
        self.assertEqual(main[-1].run_name, "humanoid_residual_d064_s2_100m")

    def test_fixed_scientific_configuration(self):
        fixed = self.plan["fixed_args"]
        self.assertEqual(fixed["env_id"], "humanoid")
        self.assertEqual(fixed["eval_env_id"], "humanoid")
        self.assertEqual(fixed["seed"], 2)
        self.assertEqual(fixed["num_envs"], 512)
        self.assertEqual(fixed["max_replay_size"], 10_000)
        self.assertEqual(fixed["wandb_project_name"], "crl_scaling_laws")
        self.assertEqual(fixed["wandb_entity"], "ideas-ncbr")
        self.assertNotIn("use_simba", fixed)
        self.assertFalse(fixed["capture_vis"])

    def test_command_pairs_actor_and_critic_depth(self):
        experiment = self.experiments[2]
        command = grid.build_command(
            ["uv", "run", "--no-sync", "train.py"],
            self.plan["fixed_args"],
            experiment,
            Path("/tmp/scaling-run"),
            slurm_job_id="169548",
        )
        self.assertEqual(command[:6], ["srun", "--jobid=169548", "--overlap", "--nodes=1", "--ntasks=1", "uv"])
        self.assertEqual(command[command.index("--critic-depth") + 1], "16")
        self.assertEqual(command[command.index("--actor-depth") + 1], "16")
        self.assertEqual(command[command.index("--total-env-steps") + 1], "1000000")
        self.assertEqual(command[command.index("--num-epochs") + 1], "5")
        self.assertIn("--track", command)
        self.assertEqual(command[command.index("--track") + 1], "--wandb-entity")
        self.assertIn("--no-capture-vis", command)

    def test_heldout_is_separate_from_preheldout_queue(self):
        selected = grid.select_experiments(self.experiments, "pre-heldout", None, None)
        self.assertEqual(selected[-1].grid_id, "M32")
        self.assertNotIn("M64", [experiment.grid_id for experiment in selected])


class SimbaScalingGridTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = grid.load_plan(SIMBA_PLAN)
        cls.experiments = grid.build_experiments(cls.plan)

    def test_simba_grid_preserves_residual_budget_and_order(self):
        self.assertEqual(
            [experiment.grid_id for experiment in self.experiments],
            ["S04", "S08", "S16", "S32", "S64", "M04", "M08", "M16", "M32", "M64"],
        )
        self.assertTrue(all(experiment.total_env_steps == 1_000_000 for experiment in self.experiments[:5]))
        self.assertTrue(all(experiment.total_env_steps == 100_000_000 for experiment in self.experiments[5:]))

    def test_simba_grid_is_separate_and_explicit(self):
        fixed = self.plan["fixed_args"]
        self.assertEqual(fixed["use_simba"], 1)
        self.assertEqual(fixed["wandb_project_name"], "crl_scaling_laws")
        self.assertEqual(fixed["wandb_entity"], "ideas-ncbr")
        self.assertTrue(all("_simba_" in experiment.run_name for experiment in self.experiments))
        self.assertNotEqual(
            self.experiments[0].wandb_group,
            ScalingGridTest.plan["stages"]["smoke"]["wandb_group"],
        )

    def test_simba_command_selects_encoder(self):
        command = grid.build_command(
            self.plan["runner"],
            self.plan["fixed_args"],
            self.experiments[0],
            Path("/tmp/simba-scaling-run"),
        )
        self.assertEqual(command[command.index("--use-simba") + 1], "1")
        self.assertEqual(command[command.index("--critic-depth") + 1], "4")
        self.assertEqual(command[command.index("--actor-depth") + 1], "4")


if __name__ == "__main__":
    unittest.main()
