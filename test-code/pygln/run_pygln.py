import numpy as np
from sklearn.metrics import accuracy_score

from pygln import utils
from pygln.pytorch import GLN

print('Getting dataset')
X_train, y_train, X_test, y_test = utils.get_mnist(deskewed=True)

model = GLN(layer_sizes=[4, 4, 1], input_size=X_train.shape[1],
            num_classes=10)

print('Training')
for n in range(X_train.shape[0]):
    model.predict(X_train[n:n+1], target=y_train[n:n+1])

print('Testing')
preds = []
for n in range(X_test.shape[0]):
    preds.append(model.predict(X_test[n]))

print('Accuracy:', accuracy_score(y_test, np.vstack(preds)))