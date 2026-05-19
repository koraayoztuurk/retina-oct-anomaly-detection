from pathlib import Path
import unittest

from utils import validate_clean_output_root


class CleanOutputSafetyTests(unittest.TestCase):
    def test_refuses_project_root(self) -> None:
        project_root = Path(__file__).resolve().parents[1]

        with self.assertRaises(ValueError):
            validate_clean_output_root(project_root)

    def test_refuses_shared_outputs_root(self) -> None:
        project_root = Path(__file__).resolve().parents[1]

        with self.assertRaises(ValueError):
            validate_clean_output_root(project_root / "outputs")

    def test_allows_nested_tmp_folder(self) -> None:
        project_root = Path(__file__).resolve().parents[1]

        resolved = validate_clean_output_root(project_root / "tmp" / "smoke_ae")

        self.assertEqual(resolved, (project_root / "tmp" / "smoke_ae").resolve())

    def test_allows_nested_experiment_folder(self) -> None:
        project_root = Path(__file__).resolve().parents[1]

        resolved = validate_clean_output_root(project_root / "outputs" / "experiments" / "run_a")

        self.assertEqual(resolved, (project_root / "outputs" / "experiments" / "run_a").resolve())


if __name__ == "__main__":
    unittest.main()
