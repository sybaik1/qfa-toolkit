import unittest

import src.recognition_strategy as recognition_strategy

Result = recognition_strategy.RecognitionStrategy.Result


class TestRecognitionStrategy(unittest.TestCase):
    def test_recognition_strategy(self):
        s = recognition_strategy.RecognitionStrategy(0.25, 0.75, False, False)
        self.assertEqual(s(0.125), Result.REJECT)
        self.assertEqual(s(0.5), Result.INVALID)
        self.assertEqual(s(0.875), Result.ACCEPT)

    def test_cutpoint(self):
        s = recognition_strategy.CutPoint(0.5)
        self.assertEqual(s(0.25), Result.REJECT)
        self.assertEqual(s(0.75), Result.ACCEPT)

    def test_isolated_cutpoint(self):
        s = recognition_strategy.IsolatedCutPoint(0.5, 0.125)
        self.assertEqual(s(0.25), Result.REJECT)
        self.assertEqual(s(0.5), Result.INVALID)
        self.assertEqual(s(0.75), Result.ACCEPT)

    def test_positive_one_sided_bounded_error(self):
        s = recognition_strategy.PositiveOneSidedBoundedError(0.25)
        self.assertEqual(s(0.0), Result.REJECT)
        self.assertEqual(s(0.5), Result.INVALID)
        self.assertEqual(s(0.875), Result.ACCEPT)

    def test_negative_one_sided_bounded_error(self):
        s = recognition_strategy.NegativeOneSidedBoundedError(0.25)
        self.assertEqual(s(0.125), Result.REJECT)
        self.assertEqual(s(0.5), Result.INVALID)
        self.assertEqual(s(1.0), Result.ACCEPT)


if __name__ == '__main__':
    unittest.main()
