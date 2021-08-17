import numpy as np
import torch
from sklearn.metrics import accuracy_score

from pygln import utils
from pygln.pytorch import GLN

print("Getting dataset")
X_train, y_train, X_test, y_test = utils.get_fashionmnist(deskewed=False)

model = GLN(
    layer_sizes=[16, 16, 1],
    input_size=X_train.shape[1],
    num_classes=10,
    pred_clipping=1e-5,
    learning_rate=1e-3,
    bias=False,
    context_bias=False,
)

print("Training")
for n in range(X_train.shape[0]):
    model.predict(X_train[n : n + 1], target=y_train[n : n + 1])

print("Testing")
preds = []
for n in range(X_test.shape[0]):
    preds.append(model.predict(X_test[n]))

print("Accuracy:", accuracy_score(y_test, np.vstack(preds)))
