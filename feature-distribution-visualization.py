import numpy as np
import umap
import matplotlib.pyplot as plt

labels = np.load('./wrn-28-id-label-list-cifar10-for-empirical-validation-w-unitization-check-8-31.npy')
# labels = np.load('./wrn-28-id-label-list-cifar10-for-empirical-validation-wo-unitization-check-8-31.npy')
# feat = np.load('./wrn-28-id-feat-list-cifar10-for-empirical-validation-w-unitization-check-8-31.npy')
# logits = np.load('./wrn-28-id-logits-list-cifar10-for-empirical-validation-w-unitization-8-31.npy')
softmax = np.load('./wrn-28-id-softmax-list-cifar10-for-empirical-validation-w-unitization-8-31.npy')
# disunit_feat = np.load('./wrn-28-id-feat-list-cifar10-for-empirical-validation-wo-unitization-check-8-31.npy')
umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
umap_embedding = umap_reducer.fit_transform(softmax)

plt.figure(figsize=(12, 10))

scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=labels, cmap='Spectral', s=10)

unique_labels = np.unique(labels)
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=14, markerfacecolor=c) for label, c in zip(unique_labels, scatter.cmap(scatter.norm(unique_labels)))], title='Classes', loc='upper left')

plt.savefig('./wrn-28-cifar10-w-unit-softmax-distribution-umap-8-31.png', dpi=700)
