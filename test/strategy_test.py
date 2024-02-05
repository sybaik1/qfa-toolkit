import unittest

import strategy


class TestStrategy(unittest.TestCase):
    def test_strategy(self):
        s = strategy.RecognitionStrategy(0.3, 0.5, False, False)
        self.assertEqual(s(0.2), strategy.RecognitionStrategy.Result.REJECT)


if __name__ == '__main__':
    unittest.main()
