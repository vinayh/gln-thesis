# Copyright 2021 DeepMind Technologies Limited. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn import model_selection
from typing import List, Optional

do_classification = True  # if False, does regression

if do_classification:
  # features, targets = datasets.load_breast_cancer(return_X_y=True)
  features, targets = datasets.fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, cache=True)
else:
  features, targets = datasets.load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    features, targets, test_size=0.2, random_state=0)
n_features = x_train.shape[-1]

# Input features are centered and scaled to unit variance:
feature_encoder = preprocessing.StandardScaler()
x_train = feature_encoder.fit_transform(x_train)
x_test = feature_encoder.transform(x_test)

if not do_classification:
  # Continuous targets are centered and scaled to unit variance:
  target_encoder = preprocessing.StandardScaler()
  y_train = np.squeeze(target_encoder.fit_transform(y_train[:, np.newaxis]))
  y_test = np.squeeze(target_encoder.transform(y_test[:, np.newaxis]))

def step_square_loss(inputs: np.ndarray,
                     weights: List[np.ndarray],
                     hyperplanes: List[np.ndarray],
                     hyperplane_bias_magnitude: Optional[float] = 1.,
                     learning_rate: Optional[float] = 1e-5,
                     target: Optional[float] = None,
                     update: bool = False,
                     ):
  """Implements a DGN inference/update using square loss."""
  r_in = inputs
  side_info = np.hstack([hyperplane_bias_magnitude, inputs])

  for w, h in zip(weights, hyperplanes):  # loop over layers
    r_in = np.hstack([1., r_in])  # add biases
    gate_values = np.heaviside(h.dot(side_info), 0).astype(bool)
    effective_weights = gate_values.dot(w).sum(axis=1)
    r_out = effective_weights.dot(r_in)

    if update:
      grad = (r_out[:, None] - target) * r_in[None]
      w -= learning_rate * gate_values[:, :, None] * grad[:, None]

    r_in = r_out
  loss = (target - r_out)**2 / 2
  return r_out, loss

def sigmoid(x):  # numerically stable sigmoid
  return np.exp(-np.logaddexp(0, -x))

def inverse_sigmoid(x):
  return np.log(x/(1-x))

def step_bernoulli(inputs: np.ndarray,
                   weights: List[np.ndarray],
                   hyperplanes: List[np.ndarray],
                   hyperplane_bias_magnitude: Optional[float] = 1.,
                   learning_rate: Optional[float] = 1e-5,
                   epsilon: float = 0.01,
                   target: Optional[float] = None,
                   update: bool = False,
                   ):
  """Implements a DGN inference/update using Bernoulli log loss."""
  r_in = np.clip(sigmoid(inputs), epsilon, 1-epsilon)
  side_info = np.hstack([hyperplane_bias_magnitude, inputs])

  for w, h in zip(weights, hyperplanes):  # loop over layers
    r_in = np.hstack([sigmoid(1.), r_in])  # add biases
    h_in = inverse_sigmoid(r_in)
    gate_values = np.heaviside(h.dot(side_info), 0).astype(bool)
    effective_weights = gate_values.dot(w).sum(axis=1)
    h_out = effective_weights.dot(h_in)
    r_out_unclipped = sigmoid(h_out)
    r_out = np.clip(r_out_unclipped, epsilon, 1 - epsilon)
    if update:
      update_indicator = np.abs(target - r_out_unclipped) > epsilon
      grad = (r_out[:, None] - target) * h_in[None]  * update_indicator[:, None]
      w -= learning_rate * gate_values[:, :, None] * grad[:, None]
    r_in = r_out
  loss = - (target * np.log(r_out) + (1 - target) * np.log(1 - r_out))
  return r_out, loss

def forward_pass(step_fn, x, y, weights, hyperplanes, learning_rate, update):
  losses, outputs = np.zeros(len(y)), np.zeros(len(y))
  y = np.float64(y)
  for i, (x_i, y_i) in enumerate(zip(x, y)):
    print(i)
    outputs[i], losses[i] = step_fn(x_i, weights, hyperplanes, target=y_i,
                                    learning_rate=learning_rate, update=update)
  return np.mean(losses), outputs

# number of neurons per layer, the last element must be 1
n_neurons = np.array([100, 10, 1])
n_branches = 20  # number of dendritic brancher per neuron

n_inputs = np.hstack([n_features + 1, n_neurons[:-1] + 1])  # 1 for the bias
dgn_weights = [np.zeros((n_neuron, n_branches, n_input))
               for n_neuron, n_input in zip(n_neurons, n_inputs)]

# Fixing random seed for reproducibility:
np.random.seed(12345)
dgn_hyperplanes = [
    np.random.normal(0, 1, size=(n_neuron, n_branches, n_features + 1))
    for n_neuron in n_neurons]
# By default, the weight parameters are drawn from a normalised Gaussian:
dgn_hyperplanes = [
    h_ / np.linalg.norm(h_[:, :, :-1], axis=(1, 2))[:, None, None]
    for h_ in dgn_hyperplanes]

if do_classification:
  eta = 1e-4
  n_epochs = 3
  step = step_bernoulli
else:
  eta = 1e-5
  n_epochs = 10
  step = step_square_loss

if do_classification:
  step = step_bernoulli
else:
  step = step_square_loss

print('Training on {} problem for {} epochs with learning rate {}.'.format(
    ['regression', 'classification'][do_classification], n_epochs, eta))
print('This may take a minute. Please be patient...')

for epoch in range(0, n_epochs + 1):
  train_loss, train_pred = forward_pass(
      step, x_train, y_train, dgn_weights,
      dgn_hyperplanes, eta, update=(epoch > 0))

  test_loss, test_pred = forward_pass(
      step, x_test, y_test, dgn_weights,
      dgn_hyperplanes, eta, update=False)
  to_print = 'epoch: {}, test loss: {:.3f} (train: {:.3f})'.format(
      epoch, test_loss, train_loss)

  if do_classification:
    accuracy_train = np.mean(np.round(train_pred) == y_train)
    accuracy = np.mean(np.round(test_pred) == y_test)
    to_print += ', test accuracy: {:.3f} (train: {:.3f})'.format(
        accuracy, accuracy_train)
  print(to_print)
