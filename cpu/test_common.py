"""Synthetic checks for the port, without touching any references."""
import unittest
import numpy as np
from common import compare


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.ref = np.zeros((1, 300, 6), dtype=np.float32)
        self.ref[0, :2] = [[10, 10, 100, 100, .8, 0], [200, 200, 300, 300, .7, 1]]

    def test_permutation(self):
        actual = self.ref.copy()
        actual[0, :2] = actual[0, :2][::-1]
        result = compare(actual, self.ref)
        self.assertTrue(result['ok'])
        self.assertEqual(result['boxes_max'], 0)
        self.assertGreater(result['overall_max'], 0)

    def test_strict_failures(self):
        for column, delta in ((0, 10), (4, .02), (5, 1)):
            with self.subTest(column=column):
                actual = self.ref.copy()
                actual[0, 0, column] += delta
                self.assertFalse(compare(actual, self.ref)['ok'])
        actual = self.ref.copy()
        actual[0, 0, 4] = .1
        result = compare(actual, self.ref)
        self.assertFalse(result['ok'])
        self.assertEqual(result['min_iou'], 0)
        self.assertEqual(result['boxes_max'], 0)

    def test_empty(self):
        result = compare(np.zeros_like(self.ref), np.zeros_like(self.ref))
        self.assertTrue(result['ok'])
        self.assertEqual(result['min_iou'], 1)

    def test_int8_unmatched(self):
        for score, accepted in ((.29, True), (.30, False)):
            actual = self.ref.copy()
            actual[0, 2] = [400, 400, 500, 500, score, 2]
            result = compare(actual, self.ref, int8=True)
            self.assertEqual(result['ok'], accepted)
            self.assertEqual(len(result['unmatched']), 1)
        actual = self.ref.copy()
        actual[0, 0, 4] -= .06
        self.assertFalse(compare(actual, self.ref, int8=True)['ok'])


class MapPreprocessingTests(unittest.TestCase):
    def test_undo_letterbox_and_threshold(self):
        from eval_map import original_boxes
        output = np.zeros((1, 300, 6), dtype=np.float32)
        output[0, 0] = [160, 0, 480, 640, .001, 0]
        output[0, 1] = [160, 0, 480, 640, .0009, 0]
        rows = original_boxes(output, 320, 640)
        self.assertEqual(rows.shape, (1, 6))
        np.testing.assert_array_equal(rows[0, :4], [0, 0, 320, 640])
        output[0, 0, :4] = [0, 160, 640, 480]
        rows = original_boxes(output, 640, 320)
        np.testing.assert_array_equal(rows[0, :4], [0, 0, 640, 320])


if __name__ == '__main__':
    unittest.main()
