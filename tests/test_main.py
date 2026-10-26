import unittest
from main import quality_class

class QualityClassTests(unittest.TestCase):
    def test_below_threshold_is_negative(self):
        self.assertEqual(quality_class(6, 7), 0)
    def test_at_threshold_is_positive(self):
        self.assertEqual(quality_class(7, 7), 1)
    def test_invalid_quality_is_rejected(self):
        with self.assertRaises(ValueError):
            quality_class(11, 7)
    def test_invalid_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            quality_class(6, 0)

if __name__ == "__main__":
    unittest.main()
