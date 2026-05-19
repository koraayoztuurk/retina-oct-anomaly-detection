from pathlib import Path
import unittest

from data_utils import parse_patient_id, split_normal_train_val


class PatientIdParsingTests(unittest.TestCase):
    def test_patient_id_keeps_class_prefix(self) -> None:
        cnv = Path("data/oct2017/test/CNV/CNV-0101-1.jpeg")
        normal = Path("data/oct2017/test/NORMAL/NORMAL-0101-1.jpeg")

        self.assertEqual(parse_patient_id(cnv), "CNV-0101")
        self.assertEqual(parse_patient_id(normal), "NORMAL-0101")
        self.assertNotEqual(parse_patient_id(cnv), parse_patient_id(normal))

    def test_patient_id_fallback_uses_parent_class(self) -> None:
        path = Path("data/oct2017/test/DRUSEN/0101-1.jpeg")

        self.assertEqual(parse_patient_id(path), "DRUSEN-0101")

    def test_train_validation_split_has_no_patient_overlap(self) -> None:
        paths = [
            Path(f"data/oct2017/train/NORMAL/NORMAL-{patient_id:04d}-{image_id}.jpeg")
            for patient_id in range(10)
            for image_id in range(2)
        ]

        train_paths, val_paths = split_normal_train_val(paths, val_ratio=0.2, seed=42)
        train_patients = {parse_patient_id(path) for path in train_paths}
        val_patients = {parse_patient_id(path) for path in val_paths}

        self.assertTrue(train_patients)
        self.assertTrue(val_patients)
        self.assertFalse(train_patients & val_patients)


if __name__ == "__main__":
    unittest.main()
