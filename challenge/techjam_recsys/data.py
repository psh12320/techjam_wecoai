"""Leakage-safe KuaiRand-Pure loading and temporal feature construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_END = 20220421
VALID_START = 20220422
VALID_END = 20220428
EXPECTED_ROWS = {"train": 1_141_112, "valid": 124_909}

LOG_COLUMNS = [
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "time_ms",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "play_time_ms",
    "duration_ms",
    "tab",
]

VIDEO_COLUMNS = [
    "video_id",
    "author_id",
    "video_type",
    "upload_dt",
    "upload_type",
    "video_duration",
    "server_width",
    "server_height",
    "music_id",
    "music_type",
    "tag",
]

CATEGORICAL_FEATURES = [
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "hour",
    "weekday",
    "music_id",
    "music_type",
    "video_type",
    "upload_type",
    "primary_tag",
]

BASE_FEATURES = CATEGORICAL_FEATURES + [
    "log_duration",
    "upload_age_days",
    "aspect_ratio",
]


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    valid: pd.DataFrame
    valid_users: np.ndarray
    valid_labels: np.ndarray


def _read_logs(data_dir: Path) -> pd.DataFrame:
    dtype = {
        "user_id": "int32",
        "video_id": "int32",
        "date": "int32",
        "hourmin": "int16",
        "time_ms": "int64",
        "is_click": "int8",
        "is_like": "int8",
        "is_follow": "int8",
        "is_comment": "int8",
        "is_forward": "int8",
        "is_hate": "int8",
        "long_view": "int8",
        "play_time_ms": "float32",
        "duration_ms": "float32",
        "tab": "int8",
    }
    first = pd.read_csv(
        data_dir / "log_standard_4_08_to_4_21_pure.csv",
        usecols=LOG_COLUMNS,
        dtype=dtype,
    )
    second = pd.read_csv(
        data_dir / "log_standard_4_22_to_5_08_pure.csv",
        usecols=LOG_COLUMNS,
        dtype=dtype,
    )
    # Hard privacy boundary: dates after the public validation period are
    # discarded immediately and cannot enter feature engineering or metrics.
    second = second.loc[second["date"] <= VALID_END].copy()
    logs = pd.concat([first, second], ignore_index=True)
    if int(logs["date"].max()) > VALID_END:
        raise RuntimeError("Held-out test dates crossed the development boundary")
    return logs


def _read_video_features(data_dir: Path) -> pd.DataFrame:
    video = pd.read_csv(
        data_dir / "video_features_basic_pure.csv",
        usecols=VIDEO_COLUMNS,
    )
    video["primary_tag"] = video["tag"].fillna("UNK").astype(str).str.split(",").str[0]
    video["upload_dt"] = pd.to_datetime(video["upload_dt"], errors="coerce")
    return video.drop(columns="tag")


def _factorize_joint(train: pd.DataFrame, valid: pd.DataFrame, column: str) -> None:
    combined = pd.concat([train[column], valid[column]], ignore_index=True)
    codes, _ = pd.factorize(combined.fillna("__UNK__"), sort=True)
    train[column] = codes[: len(train)].astype(np.int32)
    valid[column] = codes[len(train) :].astype(np.int32)


def load_development_splits(data_dir: Path) -> DatasetSplits:
    logs = _read_logs(Path(data_dir))
    video = _read_video_features(Path(data_dir))
    logs = logs.merge(video, on="video_id", how="left", validate="many_to_one")

    train = logs.loc[logs["date"] <= TRAIN_END].copy()
    valid = logs.loc[(logs["date"] >= VALID_START) & (logs["date"] <= VALID_END)].copy()
    del logs
    actual = {"train": len(train), "valid": len(valid)}
    if actual != EXPECTED_ROWS:
        raise RuntimeError(f"Unexpected official split sizes: {actual}")

    train["_row_id"] = np.arange(len(train), dtype=np.int32)
    valid["_row_id"] = np.arange(len(valid), dtype=np.int32)
    valid_users = valid["user_id"].to_numpy(copy=True)
    valid_labels = valid["long_view"].to_numpy(dtype=np.int8, copy=True)

    duration_edges = np.unique(
        np.quantile(train["duration_ms"].to_numpy(), np.linspace(0.0, 1.0, 21)[1:-1])
    )
    for frame in (train, valid):
        frame["duration_bucket"] = np.searchsorted(
            duration_edges, frame["duration_ms"].to_numpy(), side="right"
        ).astype(np.int16)
        frame["log_duration"] = np.log1p(frame["duration_ms"].clip(lower=0)).astype(
            np.float32
        )
        frame["hour"] = (frame["hourmin"] // 100).clip(0, 23).astype(np.int8)
        frame["weekday"] = (
            pd.to_datetime(frame["date"].astype(str), format="%Y%m%d").dt.weekday
        ).astype(np.int8)
        interaction_date = pd.to_datetime(frame["date"].astype(str), format="%Y%m%d")
        frame["upload_age_days"] = (
            (interaction_date - frame["upload_dt"])
            .dt.days.fillna(-1)
            .clip(-1, 10_000)
            .astype(np.float32)
        )
        height = frame["server_height"].replace(0, np.nan)
        frame["aspect_ratio"] = (
            (frame["server_width"] / height)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .astype(np.float32)
        )

    for column in CATEGORICAL_FEATURES:
        _factorize_joint(train, valid, column)

    train = train.sort_values(["time_ms", "_row_id"], kind="stable").reset_index(
        drop=True
    )
    valid = valid.sort_values("_row_id", kind="stable").reset_index(drop=True)
    return DatasetSplits(
        train=train,
        valid=valid,
        valid_users=valid_users,
        valid_labels=valid_labels,
    )
