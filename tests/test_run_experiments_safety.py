from pathlib import Path
import unittest

from run_experiments import clean_run_dir


class RunExperimentCleanSafetyTests(unittest.TestCase):
    def test_refuses_to_delete_experiments_root(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        experiments_root = project_root / "outputs" / "experiments"

        with self.assertRaises(ValueError):
            clean_run_dir(experiments_root, experiments_root)

    def test_refuses_run_dir_outside_experiments_root(self) -> None:
        project_root = Path(__file__).resolve().parents[1]

        with self.assertRaises(ValueError):
            clean_run_dir(project_root / "outputs" / "not_experiments" / "run_a", project_root / "outputs" / "experiments")


if __name__ == "__main__":
    unittest.main()
