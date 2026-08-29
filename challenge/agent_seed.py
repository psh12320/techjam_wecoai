"""Official FM seed adapted to AIDE's validation-prediction contract."""

import csv
import sys

import numpy as np

sys.path.insert(0, "input")
import baseline as B
from data import encode, load
from evaluate import evaluate

splits = load("input")
encoded, dimension = encode(splits)
train_x, train_y, _ = encoded["train"]
valid_x, valid_y, valid_users = encoded["valid"]
model = B.FM(dimension, k=16, lr=0.001, seed=0)
rng = np.random.default_rng(0)
best_primary = -1.0
best_state = None
bad_epochs = 0
for epoch in range(1, 41):
    order = rng.permutation(len(train_y))
    for offset in range(0, len(order), 8192):
        batch = order[offset : offset + 8192]
        model.step(train_x[batch], train_y[batch])
    prediction = model.predict(valid_x)
    metric = evaluate(valid_users, valid_y, prediction)
    print(
        f"epoch={epoch} GAUC={metric['GAUC']:.6f} "
        f"nDCG@5={metric['nDCG@5']:.6f} primary={metric['primary']:.6f}"
    )
    if metric["primary"] > best_primary + 1e-5:
        best_primary = metric["primary"]
        bad_epochs = 0
        best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
    else:
        bad_epochs += 1
        if bad_epochs >= 4:
            break

model.V, model.W, model.b = best_state
scores = model.predict(valid_x)
with open("working/validation_predictions.csv", "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["row_id", "score"])
    for row_id, score in enumerate(scores):
        writer.writerow([row_id, f"{float(score):.9g}"])
