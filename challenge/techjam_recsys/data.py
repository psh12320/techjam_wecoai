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
FINAL_TRAIN_END = 20220428
TEST_START = 20220429
TEST_END = 20220508

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

CONTEXT_COLUMNS = [
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "time_ms",
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

USER_COLUMNS = [
    "user_id",
    "user_active_degree",
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
] + [f"onehot_feat{index}" for index in range(12)]

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
    "tag_2",
    "tag_3",
    "duration_rule_band",
    "tab_duration_cross",
    "user_active_degree",
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
] + [f"onehot_feat{index}" for index in range(12)]

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


@dataclass
class SubmissionSplits:
    train: pd.DataFrame
    score: pd.DataFrame
    score_users: np.ndarray
    score_videos: np.ndarray


LOG_DTYPES = {
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


def _read_log_files(data_dir: Path, columns: list[str]) -> pd.DataFrame:
    dtype = {column: LOG_DTYPES[column] for column in columns}
    first = pd.read_csv(
        data_dir / "log_standard_4_08_to_4_21_pure.csv",
        usecols=columns,
        dtype=dtype,
    )
    second = pd.read_csv(
        data_dir / "log_standard_4_22_to_5_08_pure.csv",
        usecols=columns,
        dtype=dtype,
    )
    return pd.concat([first, second], ignore_index=True)


def _read_logs(data_dir: Path) -> pd.DataFrame:
    logs = _read_log_files(data_dir, LOG_COLUMNS)
    # Hard privacy boundary: dates after the public validation period are
    # discarded immediately and cannot enter feature engineering or metrics.
    logs = logs.loc[logs["date"] <= VALID_END].copy()
    if int(logs["date"].max()) > VALID_END:
        raise RuntimeError("Held-out test dates crossed the development boundary")
    return logs


def _read_video_features(data_dir: Path) -> pd.DataFrame:
    video = pd.read_csv(
        data_dir / "video_features_basic_pure.csv",
        usecols=VIDEO_COLUMNS,
    )
    tags = video["tag"].fillna("UNK").astype(str).str.split(",", expand=True)
    video["primary_tag"] = tags[0].fillna("UNK")
    video["tag_2"] = tags[1].fillna("UNK") if tags.shape[1] > 1 else "UNK"
    video["tag_3"] = tags[2].fillna("UNK") if tags.shape[1] > 2 else "UNK"
    video["upload_dt"] = pd.to_datetime(video["upload_dt"], errors="coerce")
    return video.drop(columns="tag")


def _read_user_features(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "user_features_pure.csv", usecols=USER_COLUMNS)


def _factorize_joint(train: pd.DataFrame, valid: pd.DataFrame, column: str) -> None:
    combined = pd.concat([train[column], valid[column]], ignore_index=True)
    codes, _ = pd.factorize(combined.fillna("__UNK__"), sort=True)
    train[column] = codes[: len(train)].astype(np.int32)
    valid[column] = codes[len(train) :].astype(np.int32)


def _merge_and_engineer(
    data_dir: Path, train: pd.DataFrame, valid: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    video = _read_video_features(data_dir)
    user = _read_user_features(data_dir)
    train = train.merge(video, on="video_id", how="left", validate="many_to_one")
    valid = valid.merge(video, on="video_id", how="left", validate="many_to_one")
    train = train.merge(user, on="user_id", how="left", validate="many_to_one")
    valid = valid.merge(user, on="user_id", how="left", validate="many_to_one")

    train["_row_id"] = np.arange(len(train), dtype=np.int32)
    valid["_row_id"] = np.arange(len(valid), dtype=np.int32)

    duration_edges = np.unique(
        np.quantile(train["duration_ms"].to_numpy(), np.linspace(0.0, 1.0, 21)[1:-1])
    )
    duration_rule_edges = (
        np.asarray(
            [3, 7, 13, 18, 21, 32, 48, 69, 95, 125, 175, 250, 400],
            dtype=np.float32,
        )
        * 1000
    )
    for frame in (train, valid):
        frame["duration_bucket"] = np.searchsorted(
            duration_edges, frame["duration_ms"].to_numpy(), side="right"
        ).astype(np.int16)
        frame["duration_rule_band"] = np.searchsorted(
            duration_rule_edges,
            frame["duration_ms"].to_numpy(),
            side="right",
        ).astype(np.int16)
        frame["tab_duration_cross"] = (
            frame["tab"].astype(str) + "_" + frame["duration_rule_band"].astype(str)
        )
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
    return train, valid


def load_temporal_splits(
    data_dir: Path,
    train_end: int,
    valid_start: int,
    valid_end: int,
    expected_rows: dict[str, int] | None = None,
) -> DatasetSplits:
    logs = _read_logs(Path(data_dir))
    train = logs.loc[logs["date"] <= train_end].copy()
    valid = logs.loc[(logs["date"] >= valid_start) & (logs["date"] <= valid_end)].copy()
    del logs
    actual = {"train": len(train), "valid": len(valid)}
    if expected_rows is not None and actual != expected_rows:
        raise RuntimeError(f"Unexpected split sizes: {actual}")

    valid_users = valid["user_id"].to_numpy(copy=True)
    valid_labels = valid["long_view"].to_numpy(dtype=np.int8, copy=True)
    train, valid = _merge_and_engineer(Path(data_dir), train, valid)
    return DatasetSplits(
        train=train,
        valid=valid,
        valid_users=valid_users,
        valid_labels=valid_labels,
    )


def load_development_splits(data_dir: Path) -> DatasetSplits:
    return load_temporal_splits(
        data_dir,
        train_end=TRAIN_END,
        valid_start=VALID_START,
        valid_end=VALID_END,
        expected_rows=EXPECTED_ROWS,
    )


def load_submission_splits(data_dir: Path) -> SubmissionSplits:
    """Load final-fit rows and hidden-test context without reading test outcomes."""

    data_dir = Path(data_dir)
    labeled = _read_log_files(data_dir, LOG_COLUMNS)
    train = labeled.loc[labeled["date"] <= FINAL_TRAIN_END].copy()
    del labeled
    context = _read_log_files(data_dir, CONTEXT_COLUMNS)
    score = context.loc[
        (context["date"] >= TEST_START) & (context["date"] <= TEST_END)
    ].copy()
    del context
    score_users = score["user_id"].to_numpy(copy=True)
    score_videos = score["video_id"].to_numpy(copy=True)
    train, score = _merge_and_engineer(data_dir, train, score)
    return SubmissionSplits(
        train=train,
        score=score,
        score_users=score_users,
        score_videos=score_videos,
    )
