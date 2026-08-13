from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "scripts" / "helios_w512_4gpu.sbatch"
SUBMIT = ROOT / "scripts" / "submit_helios_w512_4gpu.sh"
SMOKE = ROOT / "scripts" / "helios_multigpu_smoke.sbatch"
ANT = ROOT / "scripts" / "helios_ant_u4_w512_4gpu.sbatch"
ARM = ROOT / "scripts" / "helios_arm_push_hard_w512_4gpu.sbatch"
RUNNER = ROOT / "scripts" / "run_helios_w512_4gpu_task.sh"
ALL_ENVS_SUBMIT = ROOT / "scripts" / "submit_helios_all_envs_w512_4gpu.sh"


class HeliosWidth512GridTest(unittest.TestCase):
    def test_full_node_resources_and_four_device_training(self):
        text = SBATCH.read_text(encoding="utf-8")
        for directive in (
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --cpus-per-task=260",
            "#SBATCH --mem=400G",
            "#SBATCH --time=47:59:59",
            "#SBATCH --gres=gpu:4",
            "#SBATCH --account=plgllmreasoning-gpu-gh200",
            "#SBATCH --partition=plgrid-gpu-gh200",
        ):
            self.assertIn(directive, text)
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertIn("--num-devices 4", runner)
        self.assertIn("NCCL/2.26.2-CUDA-12.8.0", runner)

    def test_architectures_and_budgets_are_locked(self):
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn("DEPTHS=(4 8 12 16 24 32 40 48 64)", text)
        self.assertIn(
            "HUMANOID_STEPS=(842000000 647000000 561000000 508000000 "
            "443000000 403000000 400000000 400000000 400000000)",
            text,
        )
        self.assertIn("HUMANOID_EPOCHS=(169 130 113 102 100 100 100 100 100)", text)

    def test_three_seed_arrays_are_afterok_gated(self):
        text = SUBMIT.read_text(encoding="utf-8")
        self.assertEqual(len(re.findall(r"sbatch --parsable --array=0-8", text)), 3)
        self.assertIn('afterok:${SEED1_JOB}', text)
        self.assertIn('afterok:${SEED2_JOB}', text)
        for seed in (1, 2, 3):
            self.assertIn(f"CRL_SEED={seed}", text)

    def test_debug_smoke_defaults_to_two_devices(self):
        text = SMOKE.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:2", text)
        self.assertIn("#SBATCH --time=00:30:00", text)
        self.assertIn('NUM_DEVICES="${CRL_NUM_DEVICES:-2}"', text)
        self.assertIn('--num-devices "${NUM_DEVICES}"', text)
        self.assertIn("--total-env-steps 1000000", text)
        self.assertIn("--no-track", text)

    def test_environment_scripts_have_paper_horizons(self):
        ant = ANT.read_text(encoding="utf-8")
        arm = ARM.read_text(encoding="utf-8")
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertIn("CRL_ENV_ID=ant_u4_maze", ant)
        self.assertIn("CRL_EVAL_ENV_ID=ant_u4_maze_eval", ant)
        self.assertIn("CRL_FIXED_STEPS=400000000", ant)
        self.assertIn("CRL_ENV_ID=arm_push_hard", arm)
        self.assertIn("CRL_FIXED_STEPS=100000000", arm)
        self.assertIn("--batch-size 512", runner)

    def test_multienv_submission_smoke_gates_81_runs(self):
        text = ALL_ENVS_SUBMIT.read_text(encoding="utf-8")
        self.assertEqual(text.count("--array=5-5"), 3)
        self.assertEqual(text.count("--array=0-8"), 9)
        self.assertIn('afterok:${H_SMOKE}', text)
        self.assertIn('afterok:${A_SMOKE}', text)
        self.assertIn('afterok:${P_SMOKE}', text)
        self.assertIn('afterok:${H1}:${A1}:${P1}', text)
        self.assertIn('afterok:${H2}:${A2}:${P2}', text)


if __name__ == "__main__":
    unittest.main()
