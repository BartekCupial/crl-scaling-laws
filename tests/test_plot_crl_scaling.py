import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "plot_crl_scaling.py"
SPEC = importlib.util.spec_from_file_location("plot_crl_scaling", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
plotting = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plotting
SPEC.loader.exec_module(plotting)


class ScalingPlotTest(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame(
            {
                "run_id": ["a", "a", "b", "b"],
                "run_name": ["run-a", "run-a", "run-b", "run-b"],
                "run_group": ["humanoid_crl_v2_fit_s1"] * 4,
                "run_state": ["finished", "finished", "running", "running"],
                "training/envsteps": [1_000_000, 10_000_000, 900_000, 9_500_000],
                "training/walltime": [100.0, 1_000.0, 90.0, 950.0],
                "training/critic_loss": [4.0, 2.0, 5.0, 2.5],
                "config.model_family": ["simba"] * 4,
                "config.critic_network_width": [128, 128, 256, 256],
                "config.critic_depth": [4, 4, 8, 8],
                "config.seed": [1] * 4,
                "config.total_env_steps": [100_000_000] * 4,
                "config.total_trainable_params": [None, None, 8_074_915, 8_074_915],
                "config.batch_size": [256] * 4,
                "config.num_sgd_batches_per_training_step": [800] * 4,
                "config.num_envs": [512] * 4,
                "config.unroll_length": [62] * 4,
                "config.min_replay_size": [1000] * 4,
            }
        )
        self.flop_profile = plotting.load_flop_profile(plotting.DEFAULT_FLOP_PROFILE)

    def test_normalization_reconstructs_params(self):
        history = plotting.normalize_history(
            self.raw, "training/critic_loss", "humanoid_crl_v2"
        )
        run_a = history[history["run_id"] == "a"].sort_values("env_steps")
        self.assertEqual(int(run_a["trainable_params"].iloc[0]), 950_691)
        self.assertEqual(run_a["is_latest"].tolist(), [False, True])
        self.assertEqual(set(history["stage"]), {"fit"})

    def test_flop_estimate_reconstructs_exact_gradient_update_count(self):
        raw = self.raw.iloc[[0]].copy()
        raw["training/envsteps"] = 1_523_712
        history = plotting.add_flop_estimates(
            plotting.normalize_history(
                raw, "training/critic_loss", "humanoid_crl_v2"
            ),
            self.flop_profile,
        )
        row = history.iloc[0]
        profile = next(
            item
            for item in self.flop_profile["profiles"]
            if item["width"] == 128 and item["depth"] == 4
        )
        self.assertEqual(int(row["gradient_updates"]), 31 * 800)
        expected = (
            31 * 800 * profile["sgd_update_flops"]
            + 1_523_712 * profile["rollout_flops_per_env_step"]
        )
        self.assertEqual(int(row["estimated_training_flops"]), expected)

    def test_flop_profile_covers_the_27_architecture_grid(self):
        keys = {
            (item["width"], item["depth"])
            for item in self.flop_profile["profiles"]
        }
        self.assertEqual(len(keys), 27)
        self.assertIn((128, 192), keys)
        self.assertIn((256, 128), keys)
        self.assertIn((512, 32), keys)

    def test_checkpoint_selection_keeps_only_reached_observations(self):
        history = plotting.normalize_history(
            self.raw, "training/critic_loss", "humanoid_crl_v2"
        )
        history = plotting.add_flop_estimates(history, self.flop_profile)
        selected = plotting.select_checkpoint_observations(
            history, [1_000_000, 10_000_000, 30_000_000]
        )
        self.assertEqual(set(selected["checkpoint_env_steps"]), {1_000_000, 10_000_000})
        self.assertEqual(len(selected), 4)

    def test_isoflop_interpolation_is_log_linear_and_never_extrapolates(self):
        history = plotting.add_flop_estimates(
            plotting.normalize_history(
                self.raw[self.raw["run_id"].eq("a")],
                "training/critic_loss",
                "humanoid_crl_v2",
            ),
            self.flop_profile,
        ).sort_values("estimated_training_flops")
        low, high = history["estimated_training_flops"].iloc[[0, -1]]
        budget = int((low * high) ** 0.5)
        observations = plotting.interpolate_isoflop_observations(
            history, [int(low / 2), budget, int(high * 2)]
        )
        self.assertEqual(len(observations), 1)
        self.assertAlmostEqual(observations.iloc[0]["loss"], (4.0 * 2.0) ** 0.5)
        self.assertTrue(observations.iloc[0]["is_interpolated"])
        self.assertEqual(observations.iloc[0]["run_status"], "finished")

    def test_budget_minima_select_lowest_observed_loss_without_future_leakage(self):
        raw = self.raw[self.raw["run_id"].eq("a")].copy()
        extra = raw.iloc[[-1]].copy()
        extra["training/envsteps"] = 20_000_000
        extra["training/critic_loss"] = 3.0
        history = plotting.add_flop_estimates(
            plotting.normalize_history(
                pd.concat([raw, extra], ignore_index=True),
                "training/critic_loss", "humanoid_crl_v2",
            ),
            self.flop_profile,
        ).sort_values("estimated_training_flops")
        middle_compute = int(history.iloc[1]["estimated_training_flops"])
        final_compute = int(history.iloc[2]["estimated_training_flops"])
        observations = plotting.select_budget_minimum_observations(
            history, [middle_compute, final_compute]
        )
        self.assertEqual(observations["loss"].tolist(), [2.0, 2.0])
        self.assertEqual(
            observations["best_loss_observation_env_steps"].tolist(),
            [10_000_000, 10_000_000],
        )

    def test_budget_optima_keep_one_lowest_loss_run_per_budget(self):
        observations = pd.DataFrame(
            {
                "training_budget_flops": [1, 1, 2, 2],
                "loss": [3.0, 2.0, 1.5, 1.75],
                "run_id": ["a", "b", "a", "b"],
            }
        )
        optima = plotting.select_budget_optimal_observations(observations)
        self.assertEqual(optima["run_id"].tolist(), ["b", "a"])
        self.assertEqual(optima["loss"].tolist(), [2.0, 1.5])

    def test_extended_run_supersedes_100m_run_after_reaching_budget(self):
        raw = pd.concat([self.raw[self.raw["run_id"].eq("a")]] * 2, ignore_index=True)
        raw.loc[2:, "run_id"] = "a-extended"
        raw.loc[2:, "run_name"] = "run-a-extended"
        raw.loc[2:, "config.total_env_steps"] = 2_000_000_000
        history = plotting.add_flop_estimates(
            plotting.normalize_history(
                raw, "training/critic_loss", "humanoid_crl_v2"
            ),
            self.flop_profile,
        )
        budget = int(
            history.groupby("run_id")["estimated_training_flops"].max().min()
        )
        candidates = plotting.select_budget_minimum_observations(history, [budget])
        self.assertEqual(candidates["run_id"].tolist(), ["a-extended"])

    def test_group_filter_excludes_smoke_and_covers_scientific_stages(self):
        groups = plotting.experiment_groups("humanoid_crl_v2", (1, 2, 3))
        self.assertEqual(len(groups), 6)
        self.assertNotIn("humanoid_crl_v2_smoke_s1", groups)
        self.assertIn("humanoid_crl_v2_fit_s1", groups)
        self.assertIn("humanoid_crl_v2_heldout_s3", groups)

    def test_running_wandb_state_becomes_ongoing_plot_status(self):
        history = plotting.normalize_history(
            self.raw, "training/critic_loss", "humanoid_crl_v2"
        )
        statuses = history.groupby("run_id")["run_status"].first().to_dict()
        self.assertEqual(statuses, {"a": "finished", "b": "ongoing"})
        self.assertEqual(plotting.RUN_STATUS_MARKERS["ongoing"], "^")

    def test_normalization_excludes_smoke_rows_from_cached_data(self):
        smoke = self.raw.copy()
        smoke["run_id"] = "smoke"
        smoke["run_name"] = "smoke-run"
        smoke["run_group"] = "humanoid_crl_v2_smoke_s1"
        history = plotting.normalize_history(
            pd.concat([self.raw, smoke], ignore_index=True),
            "training/critic_loss",
            "humanoid_crl_v2",
        )
        self.assertEqual(set(history["run_id"]), {"a", "b"})
        self.assertEqual(set(history["stage"]), {"fit"})

    def test_run_variant_filter_selects_only_extended_names(self):
        raw = self.raw.copy()
        extended = self.raw.copy()
        extended["run_id"] = "extended"
        extended["run_name"] = "humanoid_sim_w128_d004_s1_2000m_ext_v2"
        history = plotting.normalize_history(
            pd.concat([raw, extended], ignore_index=True),
            "training/critic_loss", "humanoid_crl_v2",
        )
        selected = plotting.filter_run_variant(history, "extended")
        self.assertEqual(set(selected["run_id"]), {"extended"})
        self.assertTrue(selected["run_name"].str.endswith("_ext_v2").all())

    def test_eleven_standalone_plots_are_written(self):
        history = plotting.normalize_history(
            self.raw, "training/critic_loss", "humanoid_crl_v2"
        )
        history = plotting.add_flop_estimates(history, self.flop_profile)
        checkpoints = plotting.select_checkpoint_observations(
            history, [1_000_000, 10_000_000]
        )
        isoflops = plotting.interpolate_isoflop_observations(
            history,
            [int(history["estimated_training_flops"].median())],
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            plotting.plot_compute_trajectories(
                history, "training/critic_loss", output_dir, ("png",)
            )
            plotting.plot_parameters(
                history, "training/critic_loss", output_dir, ("png",)
            )
            plotting.plot_dataset_trajectories(
                history, "training/critic_loss", output_dir, ("png",)
            )
            plotting.plot_parameters_vs_compute(
                history, "training/critic_loss", output_dir, ("png",)
            )
            plotting.plot_samples_vs_compute(history, output_dir, ("png",))
            budget_minima = plotting.select_budget_minimum_observations(
                history, [int(history["estimated_training_flops"].max())]
            )
            plotting.plot_budget_minimum_points(
                budget_minima, "training/critic_loss", output_dir, ("png",)
            )
            plotting.plot_parameter_checkpoints(
                checkpoints, "training/critic_loss", output_dir, ("png",)
            )
            plotting.plot_isoflop_profiles(
                isoflops, "training/critic_loss", output_dir, ("png",)
            )
            self.assertEqual(
                {path.name for path in output_dir.glob("*.png")},
                {
                    "loss_vs_compute.png",
                    "loss_vs_parameters.png",
                    "loss_vs_dataset_size.png",
                    "loss_vs_parameters_checkpoints.png",
                    "loss_vs_parameters_isoflops.png",
                    "parameters_vs_compute.png",
                    "samples_vs_compute.png",
                    "best_loss_vs_parameters_budget_points.png",
                    "best_loss_vs_compute_budget_points.png",
                    "parameters_vs_compute_budget_points.png",
                    "samples_vs_compute_budget_points.png",
                },
            )

    def test_nonpositive_loss_is_rejected_for_log_plot(self):
        raw = self.raw.copy()
        raw["training/critic_loss"] = 0.0
        with self.assertRaisesRegex(ValueError, "No positive finite observations"):
            plotting.normalize_history(raw, "training/critic_loss", "humanoid_crl_v2")

    def test_isoflop_profiles_by_width_write_one_plot_per_width(self):
        history = plotting.add_flop_estimates(
            plotting.normalize_history(
                self.raw, "training/critic_loss", "humanoid_crl_v2"
            ),
            self.flop_profile,
        )
        observations = plotting.interpolate_isoflop_observations(
            history, [int(history["estimated_training_flops"].median())]
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            plotting.plot_isoflop_profiles_by_width(
                observations, "training/critic_loss", output_dir, ("png",)
            )
            expected = {
                f"loss_vs_parameters_isoflops_w{int(width)}.png"
                for width in observations["width"].unique()
            }
            self.assertEqual(
                {path.name for path in output_dir.glob("*.png")}, expected
            )


if __name__ == "__main__":
    unittest.main()
