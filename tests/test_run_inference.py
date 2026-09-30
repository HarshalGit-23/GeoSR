import unittest

import numpy as np

from src.super_resolution.run_inference import super_resolve, tile_starts


class RepeatModel:
    def __init__(self):
        self.shapes = []

    def predict(self, tile):
        self.shapes.append(tile.shape)
        return np.repeat(np.repeat(tile, 4, axis=1), 4, axis=2)


class InferenceTilingTests(unittest.TestCase):
    def test_edge_aligned_windows_do_not_create_one_pixel_tile(self):
        self.assertEqual(tile_starts(129), [0, 1])

    def test_small_non_multiple_image_has_exact_cropped_output(self):
        image = np.arange(4 * 87 * 129, dtype=np.float32).reshape(4, 87, 129)
        model = RepeatModel()

        result = super_resolve(image, model)

        self.assertEqual(result.shape, (4, 348, 516))
        self.assertEqual(model.shapes, [(4, 128, 128), (4, 128, 128)])
        expected_first_column = np.repeat(
            np.repeat(image[:, :, :1], 4, axis=1), 4, axis=2
        )
        np.testing.assert_allclose(result[:, :, :4], expected_first_column, atol=0.01)


if __name__ == "__main__":
    unittest.main()
