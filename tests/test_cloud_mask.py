import numpy as np

from src.preprocessing.cloud_mask import fill_invalid_pixels


def test_cloud_pixels_are_median_filled_without_nonfinite_values():
    rgbn = np.array(
        [
            [[1, 3, 999], [5, 7, 9]],
            [[10, 20, 999], [30, 40, 50]],
            [[2, 4, 999], [6, 8, 10]],
            [[100, 200, 999], [300, 400, 500]],
        ],
        dtype=np.float32,
    )
    scl = np.array([[4, 3, 4], [8, 9, 11]], dtype=np.uint8)

    filled, invalid, medians = fill_invalid_pixels(rgbn, scl)

    assert invalid.tolist() == [[False, True, False], [True, True, True]]
    assert medians == (500.0, 504.5, 500.5, 549.5)
    assert np.isfinite(filled).all()
    assert np.allclose(filled[:, invalid], np.array(medians)[:, None])
