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


class Int8PairingTests(unittest.TestCase):
    def candidates(self, score, shift, different_class=False):
        reference = np.zeros((1, 300, 6), dtype=np.float32)
        actual = np.zeros_like(reference)
        reference[0, 0] = [0, 0, 100, 100, score, 0]
        actual[0, 0] = [shift, 0, 100 + shift, 100, score, int(different_class)]
        return actual, reference

    def test_low_overlap_low_scores_are_acceptable_unmatched(self):
        actual, reference = self.candidates(.26, 90)  # IoU = 10/190 ≈ 0.0526
        result = compare(actual, reference, int8=True)
        self.assertTrue(result['ok'])
        self.assertEqual([r['side'] for r in result['unmatched']], ['actual', 'expected'])
        self.assertEqual(result['boxes_max'], 0)
        # FP32 must still force the pair and reject its IoU/box differences.
        self.assertFalse(compare(actual, reference)['ok'])

    def test_low_overlap_high_scores_fail_as_unmatched(self):
        actual, reference = self.candidates(.40, 90)
        result = compare(actual, reference, int8=True)
        self.assertFalse(result['ok'])
        self.assertEqual(len(result['unmatched']), 2)
        self.assertIn('unmatched score >= 0.30', result['reason'])
        self.assertNotIn('IoU below threshold', result['reason'])

    def test_matching_geometry_still_requires_same_class(self):
        actual, reference = self.candidates(.26, 100 * .15 / 1.85, different_class=True)
        result = compare(actual, reference, int8=True)
        self.assertAlmostEqual(result['min_iou'], .85, places=6)
        self.assertFalse(result['ok'])
        self.assertEqual(result['unmatched'], [])
        self.assertIn('class mismatch', result['reason'])


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
