import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "resume_interrupted_ext_run.py"
SPEC = importlib.util.spec_from_file_location("resume_interrupted_ext_run", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
resume = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = resume
SPEC.loader.exec_module(resume)


class ResumeInterruptedRunTest(unittest.TestCase):
    def test_checkpoint_epoch_requires_exact_env_step_match(self):
        text = "\n".join(
            [
                "epoch 10 out of 20 complete. metrics: {'training/envsteps': 100.0}",
                "epoch 11 out of 20 complete. metrics: {'training/envsteps': 200.0}",
            ]
        )
        self.assertEqual(resume.checkpoint_epoch(text, 200), 11)
        with self.assertRaises(ValueError):
            resume.checkpoint_epoch(text, 150)

    def test_wandb_run_id_uses_existing_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "wandb" / "run-20260811_165720-abc123").mkdir(parents=True)
            self.assertEqual(resume.wandb_run_id(run_dir), "abc123")


if __name__ == "__main__":
    unittest.main()
