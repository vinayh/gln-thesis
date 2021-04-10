import numpy as np
import torch
from pygln import utils
from pygln.pytorch import GLN
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

print('Loading/deskewing MNIST')
X_train, y_train, X_test, y_test = utils.get_mnist()

print('Fitting PCA')
pca = PCA(n_components=3)
pca.fit(X_train)
pca_result = pca.transform(X_train)

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df.loc[rndperm,:],
    legend="full",
    alpha=0.3
)

# num_contexts = 1
# context_map_size = 4
# boolean_converter = torch.nn.Parameter(torch.as_tensor(
#                 np.array([[2**i] for i in range(context_map_size)])),
#                                                    requires_grad=False)

# context_maps = torch.as_tensor(
#                 np.random.normal(size=(num_contexts, context_map_size, X_train.shape[1])),
#                 dtype=torch.float32)

# distances = X_train.matmul(context_maps)

# mapped_context_binary = (distances > 0).int()
# current_context_indices = torch.sum(mapped_context_binary * boolean_converter,
#                                     dim=-2)
