import unittest

from src.mask_models.training_schedule import (
    expected_optimizer_steps,
    optimizer_steps_per_epoch,
)


class MaskTrainingScheduleTests(unittest.TestCase):
    def test_partial_accumulation_is_flushed(self):
        self.assertEqual(optimizer_steps_per_epoch(314, 8), 40)

    def test_full_training_schedule_has_1600_updates(self):
        self.assertEqual(expected_optimizer_steps(314, 8, 40), 1600)

    def test_invalid_accumulation_is_rejected(self):
        with self.assertRaises(ValueError):
            optimizer_steps_per_epoch(314, 0)


if __name__ == "__main__":
    unittest.main()
