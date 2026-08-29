"""Ordered historical features that simulate what was known at recommendation time."""

from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import BASE_FEATURES, CATEGORICAL_FEATURES, DatasetSplits

RATE_SPECS = [
    (("video_id",), "long_view", 20.0, "video_lv"),
    (("author_id",), "long_view", 50.0, "author_lv"),
    (("duration_bucket",), "long_view", 200.0, "duration_lv"),
    (("video_id", "tab"), "long_view", 20.0, "video_tab_lv"),
    (("user_id", "video_id"), "long_view", 5.0, "user_video_lv"),
    (("user_id", "author_id"), "long_view", 10.0, "user_author_lv"),
    (("user_id", "duration_bucket"), "long_view", 10.0, "user_duration_lv"),
    (("user_id", "primary_tag"), "long_view", 10.0, "user_tag_lv"),
    (("video_id",), "is_click", 20.0, "video_click"),
    (("author_id",), "is_click", 50.0, "author_click"),
    (("user_id", "video_id"), "is_click", 5.0, "user_video_click"),
]


@dataclass
class FeatureMatrices:
    train_x: pd.DataFrame
    train_y: np.ndarray
    valid_x: pd.DataFrame
    valid_y: np.ndarray
    valid_users: np.ndarray
    categorical_features: list[str]
    feature_names: list[str]


def _ordered_rate_feature(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    keys: tuple[str, ...],
    target: str,
    alpha: float,
    prefix: str,
) -> list[str]:
    prior = float(train[target].mean())
    group_key: str | list[str] = keys[0] if len(keys) == 1 else list(keys)
    grouped = train.groupby(group_key, sort=False, observed=True)[target]
    previous_sum = grouped.cumsum() - train[target]
    previous_count = train.groupby(group_key, sort=False, observed=True).cumcount()
    rate_name = f"{prefix}_rate"
    count_name = f"{prefix}_log_count"
    train[rate_name] = (
        (previous_sum + alpha * prior) / (previous_count + alpha)
    ).astype(np.float32)
    train[count_name] = np.log1p(previous_count).astype(np.float32)

    table = (
        train.groupby(list(keys), sort=False, observed=True)[target]
        .agg(["sum", "count"])
        .reset_index()
    )
    table[rate_name] = (
        (table["sum"] + alpha * prior) / (table["count"] + alpha)
    ).astype(np.float32)
    table[count_name] = np.log1p(table["count"]).astype(np.float32)
    original_order = valid["_row_id"].to_numpy(copy=True)
    valid_with_features = valid.merge(
        table[list(keys) + [rate_name, count_name]],
        on=list(keys),
        how="left",
        sort=False,
        validate="many_to_one",
    )
    valid_with_features = valid_with_features.sort_values("_row_id", kind="stable")
    if not np.array_equal(valid_with_features["_row_id"].to_numpy(), original_order):
        raise RuntimeError(f"Feature merge changed validation row order for {prefix}")
    valid[rate_name] = valid_with_features[rate_name].fillna(prior).astype(np.float32)
    valid[count_name] = valid_with_features[count_name].fillna(0).astype(np.float32)
    return [rate_name, count_name]


def _duration_preference(train: pd.DataFrame, valid: pd.DataFrame) -> list[str]:
    global_positive_mean = float(
        train.loc[train["long_view"] == 1, "log_duration"].mean()
    )
    positive_duration = train["log_duration"] * train["long_view"]
    group = train.groupby("user_id", sort=False, observed=True)
    previous_positive_sum = (
        positive_duration.groupby(train["user_id"], sort=False).cumsum()
        - positive_duration
    )
    previous_positive_count = group["long_view"].cumsum() - train["long_view"]
    preference = (previous_positive_sum + 5.0 * global_positive_mean) / (
        previous_positive_count + 5.0
    )
    train["user_positive_duration_mean"] = preference.astype(np.float32)
    train["duration_preference_distance"] = np.abs(
        train["log_duration"] - train["user_positive_duration_mean"]
    ).astype(np.float32)

    summary = (
        train.assign(_positive_duration=positive_duration)
        .groupby("user_id", sort=False, observed=True)
        .agg(
            _positive_duration_sum=("_positive_duration", "sum"),
            _positive_count=("long_view", "sum"),
        )
        .reset_index()
    )
    summary["user_positive_duration_mean"] = (
        summary["_positive_duration_sum"] + 5.0 * global_positive_mean
    ) / (summary["_positive_count"] + 5.0)
    mapped = valid[["_row_id", "user_id"]].merge(
        summary[["user_id", "user_positive_duration_mean"]],
        on="user_id",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    mapped = mapped.sort_values("_row_id", kind="stable")
    valid["user_positive_duration_mean"] = (
        mapped["user_positive_duration_mean"]
        .fillna(global_positive_mean)
        .astype(np.float32)
        .to_numpy()
    )
    valid["duration_preference_distance"] = np.abs(
        valid["log_duration"] - valid["user_positive_duration_mean"]
    ).astype(np.float32)
    return ["user_positive_duration_mean", "duration_preference_distance"]


def build_features(splits: DatasetSplits) -> FeatureMatrices:
    train = splits.train
    valid = splits.valid
    engineered: list[str] = []
    for keys, target, alpha, prefix in RATE_SPECS:
        engineered.extend(
            _ordered_rate_feature(train, valid, keys, target, alpha, prefix)
        )
        gc.collect()
    engineered.extend(_duration_preference(train, valid))

    feature_names = BASE_FEATURES + engineered
    for column in feature_names:
        train[column] = train[column].replace([np.inf, -np.inf], np.nan).fillna(0)
        valid[column] = valid[column].replace([np.inf, -np.inf], np.nan).fillna(0)
    return FeatureMatrices(
        train_x=train[feature_names].copy(),
        train_y=train["long_view"].to_numpy(dtype=np.int8, copy=True),
        valid_x=valid[feature_names].copy(),
        valid_y=splits.valid_labels,
        valid_users=splits.valid_users,
        categorical_features=[
            name for name in CATEGORICAL_FEATURES if name in feature_names
        ],
        feature_names=feature_names,
    )
