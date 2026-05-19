import unittest

import pandas as pd

from score_ablation import aggregate_patient_scores


class PatientAggregationTests(unittest.TestCase):
    def test_rejects_mixed_class_patient_group(self) -> None:
        frame = pd.DataFrame(
            [
                {"path": "a", "patient_id": "101", "class_name": "NORMAL", "label": 0, "mse": 0.1},
                {"path": "b", "patient_id": "101", "class_name": "CNV", "label": 1, "mse": 0.2},
            ]
        )

        with self.assertRaises(ValueError):
            aggregate_patient_scores(frame, ["mse"], "mean")

    def test_keeps_class_specific_patient_groups(self) -> None:
        frame = pd.DataFrame(
            [
                {"path": "a", "patient_id": "NORMAL-101", "class_name": "NORMAL", "label": 0, "mse": 0.1},
                {"path": "b", "patient_id": "CNV-101", "class_name": "CNV", "label": 1, "mse": 0.2},
            ]
        )

        result = aggregate_patient_scores(frame, ["mse"], "mean")

        self.assertEqual(len(result), 2)
        self.assertEqual(set(result["class_name"]), {"NORMAL", "CNV"})


if __name__ == "__main__":
    unittest.main()
