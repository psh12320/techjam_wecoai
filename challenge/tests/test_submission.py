from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import pytest

from challenge.run_enriched_fm import encode_fields
from challenge.train_submission import verify_submission, write_submission


def test_encoder_maps_validation_only_category_to_unknown():
    train = pd.DataFrame({"field": ["known_a", "known_b", "known_a"]})
    valid = pd.DataFrame({"field": ["known_b", "unseen"]})

    train_x, valid_x, dimension, cardinalities = encode_fields(train, valid, ["field"])

    assert dimension == 3
    assert cardinalities == {"field": 3}
    assert set(train_x[:, 0]) == {0, 1}
    assert valid_x[0, 0] in {0, 1}
    assert valid_x[1, 0] == 2


def test_submission_writer_preserves_row_alignment(tmp_path):
    path = tmp_path / "submission.csv"
    write_submission(
        path,
        np.asarray([7, 8]),
        np.asarray([70, 80]),
        np.asarray([0.25, -1.5]),
    )

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows == [
        ["row_id", "user_id", "video_id", "score"],
        ["0", "7", "70", "0.25"],
        ["1", "8", "80", "-1.5"],
    ]
    assert verify_submission(path, np.asarray([7, 8]), np.asarray([70, 80])) == {
        "rows": 2,
        "finite_scores": True,
        "aligned": True,
    }


def test_submission_writer_rejects_non_finite_scores(tmp_path):
    with pytest.raises(RuntimeError, match="NaN or infinite"):
        write_submission(
            tmp_path / "bad.csv",
            np.asarray([1]),
            np.asarray([2]),
            np.asarray([np.nan]),
        )
